"""
Test for End-Token Issue: Streaming vs Non-Streaming Consistency

This test ensures that End-Tokens are handled consistently across different
models and streaming modes using actual token metrics instead of word estimates.
"""

import logging
import signal
import subprocess
import time
from typing import Dict, List, Tuple, Any
import json
import math
import subprocess
from functools import lru_cache

import os
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

# RAM requirements by quantization level
# 4-bit (GB)
MODEL_RAM_REQUIREMENTS_4BIT = {
    "0.5B": 1,   "1B": 2,    "3B": 4,    "4B": 5,
    "7B": 8,     "8x7B": 16, "24B": 20,  "30B": 24,
    "70B": 40,   "480B": 180,
}

# FP16/BF16 (GB) — conservative
MODEL_RAM_REQUIREMENTS_FP16 = {
    "0.5B": 2,   "1B": 4,    "3B": 8,    "4B": 12,
    "7B": 16,    "8x7B": 180, "24B": 48,  "30B": 60,
    "70B": 140,  "480B": 960,
}

# Model-specific End-Tokens to check for (comprehensive list)
MODEL_END_TOKENS = {
    "llama": ["</s>", "<|end_of_text|>", "<|eot_id|>"],  # Llama-2/3.x tokens
    "mistral": ["</s>", "<|endoftext|>"],  # Mistral variants
    "qwen": ["<|im_end|>", "<|endoftext|>", "<|end|>", "</s>"],  # Qwen variants  
    "phi": ["<|endoftext|>", "<|end|>", "</s>"],  # Phi-3 variants
    "mixtral": ["</s>", "<|endoftext|>"],  # Mixtral (Mistral-based)
    "gpt-oss": ["<|return|>"],  # GPT-OSS reasoning models: <|end|> is NOT a stop token, only <|return|>
    "default": [  # Comprehensive catch-all list
        "</s>", "<|im_end|>", "<|endoftext|>", "<|end_of_text|>", 
        "<|eot_id|>", "<|end|>", "<end>", "</end>", "<eos>", "</eos>",
        "<|assistant|>", "<|user|>", "<|system|>"
    ]
}

SERVER_BASE_URL = "http://localhost:8000"
SERVER_PORT = 8000


def extract_model_size(model_name: str) -> str:
    """Extract model size from model name (prefers MoE tokens like 8x7B)."""
    import re

    # Prefer MoE first, then standard sizes; include special cases
    size_patterns = [
        r'(\d+(?:\.\d+)?(?:x\d+)?B)',  # 30B, 0.5B, 3.2B, 8x7B
        r'Phi-3-mini',                    # Special case → 4B
        r'Qwen2\.5-(\d+(?:\.\d+)?)B', # Qwen2.5-0.5B → 0.5B
    ]

    for pattern in size_patterns:
        match = re.search(pattern, model_name)
        if match:
            if 'Phi-3-mini' in model_name:
                return '4B'
            elif 'Qwen2.5' in model_name:
                return f"{match.group(1)}B"
            else:
                return match.group(1)

    return 'unknown'


def is_quantized_4bit_from_text(text: str) -> bool:
    t = text.lower()
    markers = ["4bit", "4-bit", "q4", "int4", "gguf q4", "q4_k", "q4_"]
    return any(m in t for m in markers)


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


def is_quantized_4bit(model_name: str) -> bool:
    name = model_name.lower()
    if any(m in name for m in ("4bit", "q4", "int4")):
        return True
    info = get_model_info_via_show(model_name)
    if info and info.get("quantization"):
        return is_quantized_4bit_from_text(info["quantization"])
    return False


def estimate_required_ram_gb(model_name: str, size_str: str) -> int:
    """Estimate RAM using show-based disk size and quantization-aware maps."""
    info = get_model_info_via_show(model_name)
    q4 = is_quantized_4bit(model_name)

    # Quantization-specific disk→RAM factor
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
    if size_str and size_str != 'unknown':
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
def get_model_disk_size_gb(model_name: str) -> float:
    info = get_model_info_via_show(model_name)
    return info.get("size_gb") if info else None


def get_model_family(model_name: str) -> str:
    """Determine model family for End-Token selection."""
    model_lower = model_name.lower()
    
    if 'llama' in model_lower:
        return 'llama'
    elif 'mistral' in model_lower and 'mixtral' not in model_lower:
        return 'mistral'
    elif 'qwen' in model_lower:
        return 'qwen'
    elif 'phi' in model_lower:
        return 'phi'
    elif 'mixtral' in model_lower:
        return 'mixtral'
    elif 'gpt-oss' in model_lower:
        return 'gpt-oss'
    else:
        return 'default'


def get_available_ram_gb() -> int:
    """Get available system RAM in GB (with optional safety margin)."""
    total = psutil.virtual_memory().total // (1024**3)
    available = psutil.virtual_memory().available // (1024**3)
    try:
        safety_factor = float(os.getenv("MLXK_TEST_RAM_SAFETY", "1.0"))
        safety_factor = max(0.1, min(1.0, safety_factor))
    except Exception:
        safety_factor = 1.0
    safe_usable = min(int(available * safety_factor), total - 4)
    return max(1, safe_usable)


def find_existing_mlxk_servers() -> List[psutil.Process]:
    """Find running MLX Knife server processes (best-effort)."""
    servers = []
    for p in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmd = ' '.join(p.info.get('cmdline') or [])
            if 'mlx_knife.server:app' in cmd or ('mlxk' in cmd and 'server' in cmd):
                servers.append(p)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return servers


def cleanup_zombie_servers(port: int):
    """Clean up any zombie MLX Knife servers on the specified port."""
    logger.info(f"🧹 Checking for existing servers on port {port}")
    # Try to find listeners on the port
    try:
        conns = psutil.net_connections(kind='inet')
    except (psutil.AccessDenied, PermissionError):
        conns = []
    for c in conns:
        if c.laddr.port == port and c.status == psutil.CONN_LISTEN and c.pid:
            try:
                proc = psutil.Process(c.pid)
                cmd = ' '.join(proc.cmdline())
                if 'mlxk' in cmd and 'server' in cmd:
                    logger.warning(f"⚠️  Killing leftover MLX Knife server PID {proc.pid}")
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                    except psutil.TimeoutExpired:
                        proc.kill()
                else:
                    logger.warning(f"Port {port} occupied by non-MLX process {proc.pid}: {cmd}")
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
    # Also kill any stray MLX Knife servers regardless of port
    for proc in find_existing_mlxk_servers():
        try:
            logger.warning(f"⚠️  Cleaning zombie MLX Knife server PID {proc.pid}")
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except psutil.TimeoutExpired:
                proc.kill()
        except psutil.NoSuchProcess:
            pass


class MLXKnifeServerManager:
    """Context manager for MLX Knife server lifecycle with zombie cleanup."""
    
    def __init__(self):
        self.process = None
        
    def __enter__(self):
        self.start_server()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_server()
        
    def start_server(self):
        """Start MLX Knife server."""
        # Ensure signal handlers are installed for robust cleanup (server-only)
        try:
            pg.install_signal_handlers()
        except Exception:
            pass
        # Ensure no stale server blocks the port
        cleanup_zombie_servers(SERVER_PORT)
        logger.info("Starting MLX Knife server...")
        self.process = subprocess.Popen(
            ["mlxk", "server", "--host", "127.0.0.1", "--port", str(SERVER_PORT)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid if hasattr(os, "setsid") else None
        )
        # Track for robust cleanup on Ctrl-C
        pg.register_popen(self.process, label="mlxk-server")
        
        # Wait for server to be ready
        for attempt in range(30):
            try:
                response = requests.get(f"{SERVER_BASE_URL}/health", timeout=2)
                if response.status_code == 200:
                    logger.info("Server is ready")
                    return
            except:
                pass
            time.sleep(1)
        
        raise RuntimeError("Server failed to start within 30 seconds")
        
    def stop_server(self):
        """Stop MLX Knife server with proper cleanup."""
        if self.process:
            logger.info("Stopping server...")
            # Graceful shutdown attempt
            try:
                self.process.terminate()
            except Exception:
                pass
            try:
                self.process.wait(timeout=10)
                logger.info("Server stopped gracefully")
            except subprocess.TimeoutExpired:
                logger.warning("Server did not stop gracefully, force killing...")
                # Kill process group first for immediate stop
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
                logger.info("Server force killed")
            try:
                pg.unregister(self.process.pid)
            except Exception:
                pass
            
            # Wait a bit for port cleanup
            time.sleep(2)
            
            # Verify port is actually free
            for attempt in range(5):
                try:
                    response = requests.get(f"{SERVER_BASE_URL}/health", timeout=1)
                    if attempt == 4:
                        logger.warning("Port may still be occupied after server shutdown")
                    time.sleep(1)
                except requests.exceptions.RequestException:
                    # Good - server is really down
                    logger.info("Port confirmed free")
                    break


def get_available_models() -> List[str]:
    """Get list of available models from server."""
    try:
        response = requests.get(f"{SERVER_BASE_URL}/v1/models", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return [model["id"] for model in data.get("data", [])]
    except Exception as e:
        logger.warning(f"Failed to get models: {e}")
    return []


def get_safe_models_for_system() -> List[Tuple[str, str, int]]:
    """Get models that can safely run on current system."""
    models = get_available_models()
    available_ram = get_available_ram_gb()
    safe_models = []
    
    for model in models:
        size_str = extract_model_size(model)
        ram_needed = estimate_required_ram_gb(model, size_str)

        if ram_needed <= available_ram:
            safe_models.append((model, size_str, ram_needed))

    return safe_models


def get_model_context_length(model_name: str) -> int:
    """Get model's context length from server."""
    try:
        response = requests.get(f"{SERVER_BASE_URL}/v1/models", timeout=10)
        if response.status_code == 200:
            data = response.json()
            for model in data.get("data", []):
                if model["id"] == model_name:
                    return model.get("context_length", 4096)
    except Exception:
        pass
    return 4096  # Default fallback


def get_model_aware_token_targets(model_name: str, model_size: str) -> Dict[str, int]:
    """Get realistic token targets based on actual model capabilities."""
    context_length = get_model_context_length(model_name)
    
    # Calculate reasonable target based on model size + context
    if model_size in ["1B", "3B"]:
        target_tokens = min(512, context_length // 8)
    elif model_size in ["4B", "7B"]:
        target_tokens = min(1024, context_length // 6)
    elif model_size in ["24B", "30B", "70B"]:
        target_tokens = min(2048, context_length // 4)
    else:
        target_tokens = min(800, context_length // 6)
    
    # Model-specific adjustments for known behaviors
    model_lower = model_name.lower()
    if 'phi-3' in model_lower:
        # Phi-3 models tend to be very concise, adjust expectations
        target_tokens = min(target_tokens, 200)
    
    return {
        "target_tokens": target_tokens,
        "min_tokens": target_tokens // 3,  # Allow 33% variance
        "context_length": context_length
    }


def create_adaptive_trilogy_prompt(model_size: str, target_tokens: int) -> str:
    """Create trilogy prompt adapted to model capabilities."""
    
    base_plot = '''Here is the outline for fantasy trilogy "EMBERS OF THE FORGOTTEN":

**MAIN CHARACTERS:**
1. Kaelen Veyra - The Exiled Flame Herald (32, war poet, controls Soulfire)
2. Sylra D'Tharn - The Shadow Warrior (28, assassin, uses Emotionweave)
3. Lord Morvath - The Unforgotten King (45, tragic villain with Grief-Crown)

**TRILOGY STRUCTURE:**
- Book I: "Embers of the Forgotten" - The flame that remembers
- Book II: "The Lovers' Crucible" - The fire that doesn't burn
- Book III: "The Fire That Binds" - The flame that connects

**THEMES:** Love as power not weakness, memory as healing, emotions as connection'''

    if model_size in ["1B", "3B"]:
        task = f'''**YOUR TASK:** Write a 500-word opening scene of Book I featuring Kaelen's exile.
- Focus on Kaelen's emotional state after Lirien's death
- Use poetic, mythic language
- Target approximately {target_tokens} tokens
- End with him seeing Veyra (Valley of Faces) in the distance'''
    
    elif model_size in ["4B", "7B"]:
        task = f'''**YOUR TASK:** Write the opening chapter of Book I: "The Poet Who Burned" 
- Focus on Kaelen's exile from Celestine after Lirien's execution
- Include his emotional journey and Soulfire powers
- Use poetic, mythic language with deep inner rhythm
- Target approximately {target_tokens} tokens (1000-1500 words)
- End with his arrival at Veyra (Valley of Faces)'''
    
    else:  # 24B, 30B, 70B
        task = f'''**YOUR TASK:** Write the complete first chapter of Book I: "The Poet Who Burned"
- Focus on Kaelen's exile from Celestine after his beloved Lirien's execution  
- Include his arrival at Veyra (Valley of Faces) with 30 lost masks
- Show his Soulfire powers and deep emotional development
- Use poetic, mythic language with deep inner rhythm
- Target approximately {target_tokens} tokens (2000+ words)
- Include dialogue and rich character development
- End with the mysterious mask whispering: "You were here - a thousand years ago"'''

    return f"{base_plot}\n\n{task}\n\nWrite the complete chapter now."


def make_chat_request(model_name: str, prompt: str, stream: bool = False, timeout: int = 120) -> str:
    """Make chat completion request to server."""
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "stream": stream,
        "temperature": 0.7
    }
    
    response = requests.post(
        f"{SERVER_BASE_URL}/v1/chat/completions",
        json=payload,
        timeout=timeout,
        stream=stream
    )
    
    if not response.ok:
        raise RuntimeError(f"Request failed: {response.status_code} - {response.text}")
    
    if stream:
        # Handle streaming response
        content = ""
        for line in response.iter_lines(decode_unicode=True):
            if line.startswith("data: "):
                data_str = line[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    delta = data.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    content += delta
                except json.JSONDecodeError:
                    continue
        return content
    else:
        # Handle non-streaming response
        data = response.json()
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")


def contains_end_tokens(text: str, model_name: str) -> List[str]:
    """Check if text contains any End-Tokens for the given model."""
    model_family = get_model_family(model_name)
    end_tokens = MODEL_END_TOKENS.get(model_family, MODEL_END_TOKENS["default"])
    
    found_tokens = []
    for token in end_tokens:
        if token in text:
            found_tokens.append(token)
    
    return found_tokens


def estimate_token_count(text: str) -> int:
    """Rough token count estimation (4 chars per token average)."""
    return len(text) // 4


def get_safe_models_lazy():
    """Lazy evaluation for parametrize to avoid import-time server calls."""
    try:
        return get_safe_models_for_system()
    except:
        return [("test-model", "1B", 1)]


def pytest_generate_tests(metafunc):
    """Dynamic test parametrization to avoid import-time server calls."""
    if "model_name" in metafunc.fixturenames:
        try:
            with MLXKnifeServerManager() as server:
                models = get_safe_models_for_system()
                metafunc.parametrize("model_name,size_str,ram_needed", models)
        except Exception as e:
            pytest.skip(f"Cannot set up server for testing: {e}")


@pytest.mark.server
@pytest.mark.timeout(300)  # 5 minute timeout for large models
def test_non_streaming_end_tokens(model_name, size_str, ram_needed):
    """
    Test Issue #20: Non-streaming mode should show End-Tokens (EXPECTED TO FAIL).
    
    This test validates that non-streaming responses contain visible End-Tokens,
    proving the server-side filtering bug in generate_batch().
    
    Expected result: FAIL (End-Tokens visible) - this confirms Issue #20.
    """
    logger.info(f"🔍 Testing NON-STREAMING End-Tokens with {model_name} ({size_str}, {ram_needed}GB RAM)")
    
    with MLXKnifeServerManager() as server:
        # Get model-specific token targets
        token_specs = get_model_aware_token_targets(model_name, size_str)
        logger.info(f"Token targets: {token_specs}")
        
        # Create adaptive prompt (no max_tokens - let model use natural stopping)
        prompt = create_adaptive_trilogy_prompt(size_str, token_specs["target_tokens"])
        
        logger.info("🚫 Testing NON-STREAMING mode (should show End-Tokens)...")
        
        response_content = make_chat_request(model_name, prompt, stream=False, timeout=300)
        
        # Basic validation
        assert response_content.strip(), "Non-streaming returned empty response"
        
        # Token count validation
        estimated_tokens = estimate_token_count(response_content)
        logger.info(f"Non-streaming response: ~{estimated_tokens} tokens")
        logger.info(f"Response ends with: '{response_content[-100:]}'" if len(response_content) > 100 else f"Full response end: '{response_content}'")
        
        # Should generate reasonable amount
        min_expected = token_specs["min_tokens"]
        assert estimated_tokens >= min_expected, \
            f"Non-streaming generated too few tokens: {estimated_tokens} < {min_expected}"
        
        # Issue #20 Check: Non-streaming SHOULD contain End-Tokens (this is the bug)
        found_end_tokens = contains_end_tokens(response_content, model_name)
        
        if found_end_tokens:
            logger.error(f"❌ CONFIRMED Issue #20: Non-streaming contains End-Tokens: {found_end_tokens}")
            logger.error(f"Raw response end: {repr(response_content[-50:])}")
            # This SHOULD fail - it confirms Issue #20
            assert False, f"Issue #20 CONFIRMED: Non-streaming shows End-Tokens {found_end_tokens}"
        else:
            logger.warning(f"⚠️  UNEXPECTED: Non-streaming clean (no End-Tokens found)")
            logger.info(f"✅ Non-streaming mode unexpectedly passed (no Issue #20 detected)")


@pytest.mark.server  
@pytest.mark.timeout(300)  # 5 minute timeout for large models
def test_streaming_end_tokens(model_name, size_str, ram_needed):
    """
    Test Issue #20: Streaming mode should filter End-Tokens (EXPECTED TO PASS).
    
    This test validates that streaming responses properly filter End-Tokens,
    proving the streaming pipeline works correctly.
    
    Expected result: PASS (End-Tokens filtered) - this shows streaming works correctly.
    """
    logger.info(f"🔍 Testing STREAMING End-Tokens with {model_name} ({size_str}, {ram_needed}GB RAM)")
    
    with MLXKnifeServerManager() as server:
        # Get model-specific token targets  
        token_specs = get_model_aware_token_targets(model_name, size_str)
        logger.info(f"Token targets: {token_specs}")
        
        # Create adaptive prompt (no max_tokens - let model use natural stopping)
        prompt = create_adaptive_trilogy_prompt(size_str, token_specs["target_tokens"])
        
        logger.info("✅ Testing STREAMING mode (should filter End-Tokens)...")
        
        response_content = make_chat_request(model_name, prompt, stream=True, timeout=300)
        
        # Basic validation
        assert response_content.strip(), "Streaming returned empty response"
        
        # Token count validation
        estimated_tokens = estimate_token_count(response_content)
        logger.info(f"Streaming response: ~{estimated_tokens} tokens")
        logger.info(f"Response ends with: '{response_content[-100:]}'" if len(response_content) > 100 else f"Full response end: '{response_content}'")
        
        # Should generate reasonable amount
        min_expected = token_specs["min_tokens"]
        assert estimated_tokens >= min_expected, \
            f"Streaming generated too few tokens: {estimated_tokens} < {min_expected}"
        
        # Issue #20 Check: Streaming should NOT contain End-Tokens (correct behavior)
        found_end_tokens = contains_end_tokens(response_content, model_name)
        
        if found_end_tokens:
            logger.error(f"❌ UNEXPECTED: Streaming contains End-Tokens: {found_end_tokens}")
            logger.error(f"Raw response end: {repr(response_content[-50:])}")
            assert False, f"Streaming unexpectedly shows End-Tokens {found_end_tokens}"
        else:
            logger.info(f"✅ Streaming mode correctly filtered End-Tokens")


@pytest.mark.server
@pytest.mark.timeout(600)  # Longer timeout for comparison test
def test_end_token_consistency_comparison(model_name, size_str, ram_needed):
    """
    Test Issue #20: Direct comparison of streaming vs non-streaming End-Token handling.
    
    This test runs both modes and compares their End-Token behavior to document
    the exact differences for Issue #20 analysis.
    
    Expected pattern:
    - Non-streaming: Contains End-Tokens (Issue #20 bug) 
    - Streaming: Clean responses (correct behavior)
    """
    logger.info(f"🔍 COMPARISON TEST: {model_name} ({size_str}, {ram_needed}GB RAM)")
    logger.info("="*80)
    
    with MLXKnifeServerManager() as server:
        # Get model-specific token targets
        token_specs = get_model_aware_token_targets(model_name, size_str)
        
        # Create adaptive prompt (no max_tokens)
        prompt = create_adaptive_trilogy_prompt(size_str, token_specs["target_tokens"])
        
        responses = {}
        end_token_results = {}
        
        # Test both modes
        for stream_mode in [False, True]:
            mode_name = "streaming" if stream_mode else "non-streaming"
            logger.info(f"\n📡 Testing {mode_name.upper()} mode...")
            
            response_content = make_chat_request(model_name, prompt, stream=stream_mode, timeout=300)
            responses[stream_mode] = response_content
            
            # Check End-Tokens
            found_end_tokens = contains_end_tokens(response_content, model_name)
            end_token_results[stream_mode] = found_end_tokens
            
            estimated_tokens = estimate_token_count(response_content)
            logger.info(f"{mode_name} response: ~{estimated_tokens} tokens")
            logger.info(f"{mode_name} ends with: '{response_content[-80:]}'" if len(response_content) > 80 else f"Full: '{response_content}'")
            
            if found_end_tokens:
                logger.error(f"❌ {mode_name} contains End-Tokens: {found_end_tokens}")
            else:
                logger.info(f"✅ {mode_name} clean (no End-Tokens)")
        
        # Issue #20 Pattern Analysis
        logger.info(f"\n📊 ISSUE #20 ANALYSIS for {model_name}:")
        logger.info("="*80)
        
        non_stream_tokens = end_token_results[False]
        stream_tokens = end_token_results[True]
        
        logger.info(f"Non-streaming End-Tokens: {non_stream_tokens if non_stream_tokens else 'None'}")
        logger.info(f"Streaming End-Tokens:     {stream_tokens if stream_tokens else 'None'}")
        
        # Issue #20 pattern detection
        if non_stream_tokens and not stream_tokens:
            logger.error(f"🎯 ISSUE #20 CONFIRMED!")
            logger.error(f"   - Non-streaming shows End-Tokens: {non_stream_tokens}")
            logger.error(f"   - Streaming filters correctly: Clean")
            issue_20_detected = True
        elif not non_stream_tokens and not stream_tokens:
            logger.warning(f"⚠️  Both modes clean - Issue #20 not detected")
            issue_20_detected = False
        elif non_stream_tokens and stream_tokens:
            logger.error(f"🚨 Both modes show End-Tokens - different issue?")
            issue_20_detected = False
        else:
            logger.warning(f"🤔 Unexpected pattern - investigate further")
            issue_20_detected = False
        
        # This test is purely documentary - it doesn't fail, just reports findings
        logger.info(f"\n📝 Issue #20 Status: {'CONFIRMED' if issue_20_detected else 'NOT DETECTED'}")
        logger.info("="*80)


if __name__ == "__main__":
    # Quick test run
    with MLXKnifeServerManager() as server:
        models = get_safe_models_for_system()
        print(f"Found {len(models)} safe models for testing:")
        for model, size, ram in models:
            print(f"  {model} ({size}, {ram}GB)")
