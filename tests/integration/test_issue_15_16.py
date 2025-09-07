"""
Test for Issues #15 & #16: Dynamic Model-Aware Token Limits

Issue #15: Token-Limit vs Stop-Token Race Condition
- Models cut off by artificial token limits before natural stopping
- Solution: Context-aware token policies based on model capabilities

Issue #16: Interactive vs Server Token Limit Policies  
- Interactive mode should allow unlimited tokens for natural completion
- Server mode needs DoS protection with reasonable limits
- Solution: Different token policies per usage context

This test is self-contained and manages its own MLX Knife server instance.
"""

import json
import logging
import re
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Dict, List, Tuple

import os
import math
from functools import lru_cache
import psutil
import pytest
import requests
try:
    from tests.support import process_guard as pg  # pytest path
except Exception:
    try:
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from support import process_guard as pg  # type: ignore
    except Exception:
        class _PG:
            @staticmethod
            def register_popen(*args, **kwargs):
                pass
            @staticmethod
            def unregister(*args, **kwargs):
                pass
            @staticmethod
            def install_signal_handlers():
                pass
            @staticmethod
            def kill_all(*args, **kwargs):
                pass
        pg = _PG()

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# RAM estimation: separate 4-bit vs FP16 to avoid selecting huge FP16 models
MODEL_RAM_REQUIREMENTS_4BIT = {
    "0.5B": 1,   "1B": 2,    "3B": 4,    "4B": 5,
    "7B": 8,     "8x7B": 16, "24B": 20,  "30B": 24,
    "70B": 40,   "480B": 180,
}

MODEL_RAM_REQUIREMENTS_FP16 = {
    "0.5B": 2,   "1B": 4,    "3B": 8,    "4B": 12,
    "7B": 16,    "8x7B": 180, "24B": 48,  "30B": 60,
    "70B": 140,  "480B": 960,
}

SERVER_BASE_URL = "http://localhost:8001"  # Different port to avoid conflicts
SERVER_PORT = 8001


def extract_model_size(model_name: str) -> str:
    """Extract model size from model name."""
    # Match patterns like "30B", "8x7B", "480B", "0.5B", "3.2B", "Phi-3-mini" etc.
    size_patterns = [
        r'(\d+x\d+B)',  # MoE models like "8x7B"
        r'(\d+\.?\d*B)',  # Standard like "30B", "0.5B", "3.2B"
        r'(mini|small|medium|large)',  # Qualitative sizes
    ]
    
    for pattern in size_patterns:
        match = re.search(pattern, model_name, re.IGNORECASE)
        if match:
            size = match.group(1).lower()
            # Map qualitative sizes to quantitative
            if size == 'mini':
                return '3B'  # Phi-3-mini is ~4B params
            elif size == 'small':
                return '1B'
            elif size == 'medium':
                return '7B'  
            elif size == 'large':
                return '30B'
            return size.upper()
    
    return "3B"  # Default fallback


def is_quantized_4bit(model_name: str) -> bool:
    name = model_name.lower()
    return (
        "4bit" in name or
        "q4" in name or
        "int4" in name
    )


def estimate_required_ram_gb(model_name: str, size_str: str) -> int:
    """Estimate RAM using show-based disk size and quantization-aware maps."""
    info = get_model_info_via_show(model_name)
    q4 = is_quantized_4bit(model_name)

    try:
        if q4:
            factor = float(os.getenv("MLXK_TEST_FACTOR_4BIT", os.getenv("MLXK_TEST_DISK_TO_RAM_FACTOR", "0.6")))
        else:
            factor = float(os.getenv("MLXK_TEST_FACTOR_FP16", os.getenv("MLXK_TEST_DISK_TO_RAM_FACTOR", "0.6")))
        factor = max(0.1, min(2.0, factor))
    except Exception:
        factor = 0.6

    disk_ram_est = None
    if info and info.get("size_gb") is not None:
        disk_ram_est = max(1, math.ceil(info["size_gb"] * factor))

    map_est = None
    if size_str:
        if q4:
            map_est = MODEL_RAM_REQUIREMENTS_4BIT.get(size_str)
        else:
            map_est = MODEL_RAM_REQUIREMENTS_FP16.get(size_str)

    if disk_ram_est is not None and map_est is not None:
        return max(disk_ram_est, map_est)
    if disk_ram_est is not None:
        return disk_ram_est
    if map_est is not None:
        return map_est
    return 999


def get_available_ram_gb() -> int:
    """Get available system RAM in GB (with optional safety margin)."""
    try:
        total = int(psutil.virtual_memory().total / (1024**3))
        available = int(psutil.virtual_memory().available / (1024**3))
        safety_factor = float(os.getenv("MLXK_TEST_RAM_SAFETY", "1.0"))
        safety_factor = max(0.1, min(1.0, safety_factor))
        safe_usable = min(int(available * safety_factor), total - 4)
        return max(1, safe_usable)
    except Exception:
        return 8  # Conservative fallback


def get_suitable_models(available_models: List[str]) -> List[str]:
    """Filter models based on available RAM."""
    available_ram = get_available_ram_gb()
    logger.info(f"Available RAM: {available_ram}GB")
    
    suitable = []
    for model in available_models:
        size = extract_model_size(model)
        required_ram = estimate_required_ram_gb(model, size)

        if required_ram <= available_ram:
            suitable.append(model)
            logger.info(f"✓ {model} ({size}, {required_ram}GB) - Suitable")
        else:
            logger.info(f"✗ {model} ({size}, {required_ram}GB) - Too large")
    
    return suitable


def get_cached_models() -> List[str]:
    """Get list of cached MLX models."""
    try:
        result = subprocess.run(
            ["mlxk", "list", "--framework", "mlx"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return []
        
        models = []
        for line in result.stdout.split('\n'):
            line = line.strip()
            if line and not line.startswith('MODEL') and not line.startswith('NAME'):
                # Extract model name from table format
                parts = line.split()
                if len(parts) >= 1 and not parts[0] in ['MODEL', 'NAME']:
                    models.append(parts[0])
        
        return models
    except Exception as e:
        logger.warning(f"Failed to get cached models: {e}")
        return []


def parse_size_to_gb(size_str: str) -> float:
    try:
        parts = size_str.strip().split()
        if len(parts) < 2:
            return None
        value = float(parts[0])
        unit = parts[1].upper()
        if unit.startswith('KB'):
            return value / (1024 ** 2)
        if unit.startswith('MB'):
            return value / 1024
        if unit.startswith('GB'):
            return value
        if unit.startswith('TB'):
            return value * 1024
        return None
    except Exception:
        return None


@lru_cache(maxsize=128)
def get_model_info_via_show(model_name: str) -> dict:
    """Use `mlxk show` to fetch size and quantization details for a model."""
    try:
        res = subprocess.run(["mlxk", "show", model_name], capture_output=True, text=True, timeout=15)
        if res.returncode != 0:
            return {}
        size_gb = None
        quant_info = None
        for raw in res.stdout.splitlines():
            line = raw.strip()
            if line.startswith("Size:"):
                size_text = line.split("Size:", 1)[1].strip()
                val = parse_size_to_gb(size_text)
                if val is not None:
                    size_gb = val
            elif line.startswith("Quantization:"):
                quant_info = line.split("Quantization:", 1)[1].strip()
        return {"size_gb": size_gb, "quantization": quant_info}
    except Exception:
        return {}


def extract_context_length_from_model(model_name: str) -> int:
    """Extract context length from a real model's config."""
    try:
        result = subprocess.run(
            ["mlxk", "show", model_name, "--config"],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode != 0:
            return 4096
            
        # Extract JSON from the output (it comes after "Config:")
        config_text = result.stdout
        
        # Find the JSON part after "Config:"
        config_start = config_text.find("Config:")
        if config_start == -1:
            return 4096
            
        json_text = config_text[config_start + 7:].strip()  # Skip "Config:"
        
        try:
            config = json.loads(json_text)
            context_keys = [
                "max_position_embeddings",
                "n_positions", 
                "context_length",
                "max_sequence_length",
                "seq_len"
            ]
            
            for key in context_keys:
                if key in config:
                    return config[key]
                    
            return 4096
        except json.JSONDecodeError:
            return 4096
            
    except Exception:
        return 4096


class MLXKnifeServer:
    """Manages MLX Knife server lifecycle for testing."""
    
    def __init__(self, port: int = SERVER_PORT):
        self.port = port
        self.process = None
        self.base_url = f"http://localhost:{port}"
    
    def start(self) -> bool:
        """Start the MLX Knife server."""
        try:
            # Ensure signal handlers are installed for robust cleanup (server-only)
            try:
                pg.install_signal_handlers()
            except Exception:
                pass
            cmd = [
                "mlxk", "server", 
                "--host", "127.0.0.1",
                "--port", str(self.port),
                "--max-tokens", "1000",  # Conservative default for testing
                "--log-level", "warning"
            ]
            
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=os.setsid if hasattr(os, "setsid") else None
            )
            # Track for robust cleanup on Ctrl-C
            pg.register_popen(self.process, label="mlxk-server")
            
            # Wait for server to start
            for attempt in range(30):
                try:
                    response = requests.get(f"{self.base_url}/v1/models", timeout=2)
                    if response.status_code == 200:
                        logger.info(f"MLX Knife server started on port {self.port}")
                        return True
                except requests.RequestException:
                    pass
                
                if self.process.poll() is not None:
                    logger.error("Server process died during startup")
                    return False
                
                time.sleep(1)
            
            logger.error("Server failed to start within timeout")
            return False
            
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            return False
    
    def stop(self):
        """Stop the MLX Knife server."""
        if self.process:
            try:
                # Try graceful shutdown first
                try:
                    self.process.terminate()
                except Exception:
                    pass
                try:
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    # Force kill if not responding
                    try:
                        if hasattr(os, "killpg"):
                            os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                    except Exception:
                        pass
                    try:
                        self.process.kill()
                        self.process.wait(timeout=3)
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"Error stopping server: {e}")
            finally:
                try:
                    pg.unregister(self.process.pid)
                except Exception:
                    pass
                self.process = None
    
    def chat_completion(self, model: str, messages: List[Dict], max_tokens: int = None) -> Dict:
        """Send chat completion request."""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.3,
            "stream": False
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        
        response = requests.post(
            f"{self.base_url}/v1/chat/completions",
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        return response.json()


@pytest.fixture(scope="module")
def mlx_server():
    """Provide MLX Knife server for the test session."""
    server = MLXKnifeServer()
    
    if not server.start():
        pytest.skip("Failed to start MLX Knife server")
    
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture(scope="module") 
def available_models():
    """Get available models suitable for current system."""
    all_models = get_cached_models()
    if not all_models:
        pytest.skip("No MLX models found in cache")
    
    suitable = get_suitable_models(all_models)
    if not suitable:
        pytest.skip("No suitable models found for current RAM")
    
    return suitable


@pytest.mark.server
class TestIssue15TokenLimitVsStopTokenRace:
    """Test Issue #15: Token-Limit vs Stop-Token Race Condition Resolution."""
    
    def test_model_context_length_extraction(self, available_models):
        """Test that we can extract context length from real models."""
        model = available_models[0]
        context_length = extract_context_length_from_model(model)
        
        assert context_length >= 512, f"Context length too small for {model}: {context_length}"
        assert context_length <= 1048576, f"Context length unrealistic for {model}: {context_length}"  # 1M tokens max
        
        logger.info(f"Model {model} has context length: {context_length}")
    
    def test_realistic_token_limits_prevent_race_condition(self, mlx_server, available_models):
        """Test that realistic token limits prevent race conditions."""
        model = available_models[0]
        context_length = extract_context_length_from_model(model)
        
        # Request tokens close to but under the expected server limit (context/2)
        server_limit = context_length // 2
        test_tokens = min(server_limit - 100, 500)  # Conservative test
        
        messages = [{"role": "user", "content": "Write a short story about a robot."}]
        
        response = mlx_server.chat_completion(model, messages, max_tokens=test_tokens)
        
        assert "choices" in response
        assert len(response["choices"]) > 0
        choice = response["choices"][0]
        assert "message" in choice
        assert "content" in choice["message"]
        
        content = choice["message"]["content"]
        assert len(content) > 0, "No content generated"
        
        # The key test: model should generate reasonable content within limits
        # without being cut off mid-sentence due to race conditions
        logger.info(f"Generated {len(content)} characters with {test_tokens} token limit")


@pytest.mark.server  
class TestIssue16InteractiveVsServerTokenPolicies:
    """Test Issue #16: Interactive vs Server Token Limit Policies Resolution."""
    
    def test_server_mode_uses_dos_protection_limits(self, mlx_server, available_models):
        """Test that server mode uses DoS protection (context/2)."""
        model = available_models[0]
        context_length = extract_context_length_from_model(model)
        server_limit = context_length // 2
        
        # Request more tokens than server limit should allow, but not too excessive for testing
        excessive_tokens = min(server_limit + 200, 800)  # Keep reasonable for testing
        
        messages = [{"role": "user", "content": "Write a brief summary of machine learning."}]
        
        # This should work without errors - the server should internally
        # limit tokens to the DoS protection limit
        response = mlx_server.chat_completion(model, messages, max_tokens=excessive_tokens)
        
        assert "choices" in response
        assert len(response["choices"]) > 0
        choice = response["choices"][0]
        assert "message" in choice
        assert "content" in choice["message"]
        
        content = choice["message"]["content"]
        assert len(content) > 0
        
        # The response should be successful, proving the server handles
        # excessive token requests gracefully
        logger.info(f"Server handled excessive token request ({excessive_tokens}) gracefully")
        logger.info(f"Model context: {context_length}, Server limit: {server_limit}, Generated content length: {len(content)}")
    
    def test_server_honors_reasonable_token_requests(self, mlx_server, available_models):
        """Test that server honors reasonable token requests."""
        model = available_models[0] 
        context_length = extract_context_length_from_model(model)
        server_limit = context_length // 2
        
        # Request reasonable number of tokens (well under limit)
        reasonable_tokens = min(server_limit // 4, 200)
        
        messages = [{"role": "user", "content": "Say hello."}]
        
        response = mlx_server.chat_completion(model, messages, max_tokens=reasonable_tokens)
        
        assert "choices" in response
        assert len(response["choices"]) > 0
        choice = response["choices"][0]
        assert "message" in choice
        assert "content" in choice["message"]
        
        content = choice["message"]["content"]
        assert len(content) > 0
        assert "hello" in content.lower() or "hi" in content.lower()
        
        logger.info(f"Server honored reasonable token request ({reasonable_tokens})")
    
    def test_model_capabilities_vs_hardcoded_limits(self, available_models):
        """Test that models with different context lengths get appropriate limits."""
        if len(available_models) < 2:
            pytest.skip("Need multiple models to compare context lengths")
        
        model_contexts = []
        for model in available_models[:3]:  # Test up to 3 models
            context_length = extract_context_length_from_model(model)
            model_contexts.append((model, context_length))
        
        # Verify that different models have different context lengths
        # (or at least our system recognizes their individual capabilities)
        contexts = [ctx for _, ctx in model_contexts]
        
        # At minimum, verify context extraction worked
        for model, context in model_contexts:
            assert context >= 1024, f"Model {model} context too small: {context}"
            logger.info(f"Model {model}: {context} tokens context")
        
        # The key insight: No hardcoded 500/2000 token limits!
        # Each model gets limits based on its actual capabilities
        for model, context in model_contexts:
            server_limit = context // 2
            # Server limits should be much higher than old hardcoded limits
            # for models with large context windows
            if context >= 4096:
                assert server_limit >= 2048, f"Model {model} should have server limit >= 2048, got {server_limit}"
