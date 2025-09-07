"""
Test for Issue #14: Interactive Chat Self-Conversation Bug

This test ensures that models don't continue conversations autonomously
by generating "You:", "Human:", "Assistant:" markers after their response.

This test is self-contained and manages its own MLX Knife server instance.
"""

import logging
import os
import re
import math
import subprocess
from functools import lru_cache
import signal
import subprocess
import time
from typing import List, Tuple

import psutil
import pytest
import requests
try:
    from tests.support import process_guard as pg  # pytest-run path
except Exception:
    try:
        # Direct-script fallback: add tests/ to sys.path and import support
        import sys
        from pathlib import Path
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from support import process_guard as pg  # type: ignore
    except Exception:
        # No-op fallback
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

"""Model RAM estimation helpers.

We need to avoid loading extremely large FP16 models during server tests.
Previously, we applied 4-bit RAM heuristics to all models by parsing only the
size string (e.g., "8x7B"). This incorrectly marked non-quantized models like
"Mixtral-8x7B-Instruct-v0.1" as fitting in 16GB, leading to massive swap usage.

Fix: detect 4-bit quantization in the model name and use separate maps for
4-bit vs FP16 estimates. Non-quantized Mixtral-8x7B is treated as ~180GB to
ensure it is skipped on typical machines.
"""

# 4-bit quantized models (GB)
MODEL_RAM_REQUIREMENTS_4BIT = {
    "0.5B": 1,   "1B": 2,    "3B": 4,    "4B": 5,
    "7B": 8,     "8x7B": 16, "24B": 20,  "30B": 24,
    "70B": 40,   "480B": 180,
}

# Approximate FP16/BF16 models (GB) — conservative, intentionally high
MODEL_RAM_REQUIREMENTS_FP16 = {
    "0.5B": 2,   "1B": 4,    "3B": 8,    "4B": 12,
    "7B": 16,    "8x7B": 180, "24B": 48,  "30B": 60,
    "70B": 140,  "480B": 960,
}

# Self-conversation patterns to detect Issue #14
SELF_CONVERSATION_PATTERNS = [
    r'\nYou:',
    r'\nHuman:',
    r'\nAssistant:',
    r'\nUser:',
    r'\n\nYou:',
    r'\n\nHuman:',
    r'\n\nAssistant:',
    r'\n\nUser:',
]

SERVER_BASE_URL = "http://localhost:8000"
SERVER_PORT = 8000


def extract_model_size(model_name: str) -> str:
    """Extract model size from model name."""
    # Match patterns like "30B", "8x7B", "480B", "0.5B", "3.2B", "Phi-3-mini" etc.
    size_patterns = [
        r'(\d+(?:\.\d+)?(?:x\d+)?B)',  # 30B, 0.5B, 3.2B, 8x7B, 480B
        r'Phi-3-mini',  # Special case: Phi-3-mini = ~4B
        r'Qwen2\.5-(\d+(?:\.\d+)?)B', # Qwen2.5-0.5B
    ]
    
    for pattern in size_patterns:
        match = re.search(pattern, model_name)
        if match:
            if 'Phi-3-mini' in model_name:
                return '4B'  # Phi-3-mini is ~4B parameters
            elif 'Qwen2.5' in model_name:
                return f"{match.group(1)}B"  # Extract from Qwen2.5-0.5B
            else:
                return match.group(1)
    
    return "unknown"


def is_quantized_4bit_from_text(text: str) -> bool:
    t = text.lower()
    markers = ["4bit", "4-bit", "q4", "int4", "gguf q4", "q4_k", "k_m q4", "q4_"]
    return any(m in t for m in markers)


def is_quantized_4bit(model_name: str) -> bool:
    """Detect 4-bit quantization using name and `mlxk show` output if available."""
    # Quick name-based check first
    name = model_name.lower()
    if any(m in name for m in ("4bit", "q4", "int4")):
        return True
    # Try to refine using show output
    info = get_model_info_via_show(model_name)
    if info and info.get("quantization"):
        return is_quantized_4bit_from_text(info["quantization"])
    return False


def estimate_required_ram_gb(model_name: str, size_str: str) -> int:
    """Estimate RAM using a combination of show-based disk size and size maps.

    Strategy:
    - Prefer `mlxk show` disk size and convert to RAM via quantization-specific factor.
    - If a size token is known, also compute map-based estimate and take the max for safety.
    - If no disk info and no size token, return a high sentinel to skip.
    """
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
    else:
        # Fallback to list-based size if show failed
        disk_gb = get_model_disk_size_gb(model_name)
        if disk_gb is not None:
            disk_ram_est = max(1, math.ceil(disk_gb * factor))

    map_est = None
    if size_str != "unknown":
        if q4:
            map_est = MODEL_RAM_REQUIREMENTS_4BIT.get(size_str)
        else:
            map_est = MODEL_RAM_REQUIREMENTS_FP16.get(size_str)

    # Combine estimates conservatively
    if disk_ram_est is not None and map_est is not None:
        return max(disk_ram_est, map_est)
    if disk_ram_est is not None:
        return disk_ram_est
    if map_est is not None:
        return map_est
    return 999


def parse_size_to_gb(size_str: str) -> float:
    """Parse a human size like '579.2 MB' or '8.5 GB' to GB as float."""
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
    """Use `mlxk show <model>` to obtain size and quantization info.

    Returns a dict like {"size_gb": float|None, "quantization": str|None}.
    """
    try:
        res = subprocess.run(["mlxk", "show", model_name], capture_output=True, text=True, timeout=15)
        if res.returncode != 0:
            return {}
        size_gb = None
        quant_info = None
        for raw in res.stdout.splitlines():
            line = raw.strip()
            if line.startswith("Size:"):
                # Format: Size: 579.2 MB
                size_text = line.split("Size:", 1)[1].strip()
                val = parse_size_to_gb(size_text)
                if val is not None:
                    size_gb = val
            elif line.startswith("Quantization:"):
                quant_info = line.split("Quantization:", 1)[1].strip()
        return {"size_gb": size_gb, "quantization": quant_info}
    except Exception:
        return {}


def get_model_disk_size_gb(model_name: str) -> float:
    info = get_model_info_via_show(model_name)
    return info.get("size_gb") if info else None


def get_available_models() -> List[str]:
    """Get list of available models from MLX Knife server."""
    try:
        response = requests.get(f"{SERVER_BASE_URL}/v1/models", timeout=10)
        response.raise_for_status()
        data = response.json()
        return [model["id"] for model in data["data"]]
    except Exception as e:
        pytest.skip(f"Cannot connect to MLX Knife server: {e}")


def get_safe_models_for_system() -> List[Tuple[str, str, int]]:
    """Get models that fit safely in available system RAM."""
    total_ram_gb = psutil.virtual_memory().total // (1024**3)
    available_ram_gb = psutil.virtual_memory().available // (1024**3)

    # Safety margin: configurable via MLXK_TEST_RAM_SAFETY (default 0.8)
    try:
        safety_factor = float(os.getenv("MLXK_TEST_RAM_SAFETY", "0.8"))
        safety_factor = max(0.1, min(1.0, safety_factor))
    except Exception:
        safety_factor = 0.8

    # Keep 4GB headroom as hard minimum
    max_usable_gb = min(available_ram_gb * safety_factor, total_ram_gb - 4)
    
    logger.info(f"System RAM: {total_ram_gb}GB total, {available_ram_gb}GB available")
    logger.info(f"Safe limit for model testing: {max_usable_gb:.1f}GB")
    
    safe_models = []
    all_models = get_available_models()
    
    for model in all_models:
        size_str = extract_model_size(model)
        required_ram = estimate_required_ram_gb(model, size_str)

        if required_ram <= max_usable_gb:
            safe_models.append((model, size_str, required_ram))
            logger.info(f"✅ {model} ({size_str}) - fits in {required_ram}GB")
        else:
            logger.warning(f"⏭️  Skipping {model} ({size_str}) - needs {required_ram}GB, have {max_usable_gb:.1f}GB")
    
    if not safe_models:
        pytest.skip("No models fit in available system RAM")
    
    return safe_models


def has_self_conversation_markers(text: str) -> bool:
    """Check if text contains self-conversation markers indicating Issue #14."""
    for pattern in SELF_CONVERSATION_PATTERNS:
        if re.search(pattern, text):
            return True
    return False


def chat_completion_request(model_name: str, prompt: str, max_tokens: int = 150) -> str:
    """Send chat completion request to MLX Knife server."""
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False
    }
    
    try:
        response = requests.post(
            f"{SERVER_BASE_URL}/v1/chat/completions",
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        pytest.fail(f"Chat completion failed for {model_name}: {e}")


@pytest.mark.server
def test_issue_14_self_conversation_regression_original(mlx_server, model_name: str, size_str: str, ram_needed: int):
    """
    Test Issue #14: Ensure models don't continue conversations autonomously.
    
    This test verifies that models stop cleanly after their response without
    generating additional conversation turns like "You:", "Human:", etc.
    """
    logger.info(f"🦫 Testing Issue #14 with {model_name} ({size_str}, {ram_needed}GB)")
    
    # Use constrained prompt to encourage natural stopping
    test_prompt = "Write a short story about a friendly dragon in exactly 50 words."
    
    start_time = time.time()
    response = chat_completion_request(model_name, test_prompt, max_tokens=100)
    duration = time.time() - start_time
    
    logger.info(f"⏱️  Response time: {duration:.2f}s")
    logger.info(f"📝 Response preview: {response[:100]}...")
    
    # Check for Issue #14: self-conversation markers
    if has_self_conversation_markers(response):
        # Log the problematic response for debugging
        logger.error(f"❌ Self-conversation detected in {model_name}:")
        logger.error(f"Full response: {repr(response)}")
        pytest.fail(f"Issue #14 regression: {model_name} shows self-conversation markers")
    
    logger.info(f"✅ {model_name}: No self-conversation detected - Issue #14 fix working!")


def find_existing_mlxk_servers() -> List[psutil.Process]:
    """Find any existing MLX Knife server processes."""
    servers = []
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if proc.info['cmdline'] and any('mlxk' in arg and 'server' in arg for arg in proc.info['cmdline']):
                servers.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return servers


def cleanup_zombie_servers(port: int):
    """Clean up any zombie MLX Knife servers on the specified port."""
    logger.info(f"🧹 Checking for existing servers on port {port}")
    
    # Check for processes using the port - handle macOS permission issues
    try:
        connections = psutil.net_connections(kind='inet')
    except (psutil.AccessDenied, PermissionError) as e:
        logger.warning(f"⚠️  Cannot scan network connections (permission denied): {e}")
        logger.info("🔧 Falling back to process-based cleanup only")
        connections = []
    
    for conn in connections:
        if conn.laddr.port == port and conn.status == psutil.CONN_LISTEN:
            try:
                proc = psutil.Process(conn.pid)
                logger.warning(f"⚠️  Found process {proc.pid} listening on port {port}: {proc.cmdline()}")
                
                if 'mlxk' in ' '.join(proc.cmdline()) and 'server' in ' '.join(proc.cmdline()):
                    logger.info(f"🛑 Terminating existing MLX Knife server {proc.pid}")
                    proc.terminate()
                    try:
                        proc.wait(timeout=5)
                        logger.info(f"✅ Server {proc.pid} terminated gracefully")
                    except psutil.TimeoutExpired:
                        logger.warning(f"⚡ Force killing server {proc.pid}")
                        proc.kill()
                        proc.wait()
                else:
                    logger.error(f"❌ Port {port} is occupied by non-MLX process {proc.pid}")
                    raise RuntimeError(f"Port {port} is busy with: {proc.cmdline()}")
                    
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    
    # Also check for any MLX Knife server processes (even if not on our port)
    existing_servers = find_existing_mlxk_servers()
    for server in existing_servers:
        logger.warning(f"⚠️  Found zombie MLX Knife server: {server.pid}")
        try:
            server.terminate()
            server.wait(timeout=5)
            logger.info(f"✅ Cleaned up zombie server {server.pid}")
        except (psutil.TimeoutExpired, psutil.NoSuchProcess):
            try:
                server.kill()
                logger.info(f"⚡ Force killed zombie server {server.pid}")
            except psutil.NoSuchProcess:
                pass


class MLXKnifeServerManager:
    """Context manager for MLX Knife server lifecycle with zombie cleanup."""
    
    def __init__(self, port: int = 8000):
        self.port = port
        self.process = None
        self.base_url = f"http://localhost:{port}"
    
    def start_server(self) -> bool:
        """Start MLX Knife server and wait for it to be ready."""
        try:
            # Ensure signal handlers are installed for robust cleanup (server-only)
            try:
                pg.install_signal_handlers()
            except Exception:
                pass
            # First, clean up any zombies or port conflicts
            cleanup_zombie_servers(self.port)
            
            # Check if server is already running (after cleanup)
            if self.is_server_running():
                logger.info("🟢 MLX Knife server already running")
                return True
            
            logger.info(f"🚀 Starting MLX Knife server on port {self.port}")
            
            # Start server process - use sys.executable to ensure same Python env
            import sys
            self.process = subprocess.Popen(
                [sys.executable, "-m", "mlx_knife.cli", "server", "--port", str(self.port)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                preexec_fn=os.setsid if hasattr(os, "setsid") else None
            )
            # Track for robust cleanup on Ctrl-C or failures
            pg.register_popen(self.process, label="mlxk-server")
            
            logger.info(f"📋 Started process PID: {self.process.pid}")
            
            # Give it a moment to fail fast if there's an immediate error
            time.sleep(1)
            if self.process.poll() is not None:
                stdout, stderr = self.process.communicate()
                logger.error(f"❌ Server failed immediately:")
                logger.error(f"stdout: {stdout}")
                logger.error(f"stderr: {stderr}")
                return False
            
            # Wait for server to be ready (max 30 seconds)
            for _ in range(60):  # 30 seconds, 0.5s intervals
                if self.is_server_running():
                    logger.info("✅ MLX Knife server is ready")
                    return True
                time.sleep(0.5)
            
            # Timeout - get final output
            stdout, stderr = "", ""
            if self.process:
                try:
                    if self.process.poll() is None:
                        stdout, stderr = self.process.communicate(timeout=2)
                    else:
                        stdout, stderr = self.process.communicate()
                except subprocess.TimeoutExpired:
                    stdout, stderr = "timeout", "timeout"
            
            logger.error("❌ Server failed to start within timeout")
            logger.error(f"Final stdout: {stdout}")
            logger.error(f"Final stderr: {stderr}")
            self.stop_server()
            return False
            
        except Exception as e:
            import traceback
            logger.error(f"❌ Failed to start server: {e}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            self.stop_server()
            return False
    
    def stop_server(self):
        """Stop MLX Knife server if running."""
        if self.process:
            logger.info("🛑 Stopping MLX Knife server")
            try:
                self.process.terminate()
            except Exception:
                pass
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                logger.warning("⚠️  Server didn't stop gracefully, killing...")
                # Try process group kill first
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
            try:
                pg.unregister(self.process.pid)
            except Exception:
                pass
            self.process = None
    
    def is_server_running(self) -> bool:
        """Check if server is running and healthy."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def __enter__(self):
        if not self.start_server():
            pytest.skip("Failed to start MLX Knife server")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_server()


@pytest.fixture(scope="module")
def mlx_server():
    """Pytest fixture to manage MLX Knife server for all tests in module."""
    with MLXKnifeServerManager(SERVER_PORT) as server:
        yield server


@pytest.mark.server
def test_server_health(mlx_server):
    """Verify MLX Knife server is running and healthy."""
    assert mlx_server.is_server_running(), "MLX Knife server is not healthy"
    logger.info("🟢 MLX Knife server is healthy")


@pytest.mark.server
def test_issue_14_self_conversation_regression(mlx_server, model_name: str, size_str: str, ram_needed: int):
    """
    Test Issue #14: Ensure models don't continue conversations autonomously.
    
    This test verifies that models stop cleanly after their response without
    generating additional conversation turns like "You:", "Human:", etc.
    """
    logger.info(f"🦫 Testing Issue #14 with {model_name} ({size_str}, {ram_needed}GB)")
    
    # Use constrained prompt to encourage natural stopping
    test_prompt = "Write a short story about a friendly dragon in exactly 50 words."
    
    start_time = time.time()
    response = chat_completion_request(model_name, test_prompt, max_tokens=100)
    duration = time.time() - start_time
    
    logger.info(f"⏱️  Response time: {duration:.2f}s")
    logger.info(f"📝 Response preview: {response[:100]}...")
    
    # Check for Issue #14: self-conversation markers
    if has_self_conversation_markers(response):
        # Log the problematic response for debugging
        logger.error(f"❌ Self-conversation detected in {model_name}:")
        logger.error(f"Full response: {repr(response)}")
        pytest.fail(f"Issue #14 regression: {model_name} shows self-conversation markers")
    
    logger.info(f"✅ {model_name}: No self-conversation detected - Issue #14 fix working!")


def get_safe_models_lazy():
    """Lazy evaluation for parametrize to avoid import-time server calls."""
    try:
        return get_safe_models_for_system()
    except:
        # Fallback for when server isn't running yet
        return [("test-model", "1B", 1)]


# Dynamic test generation at runtime instead of import time
def pytest_generate_tests(metafunc):
    """Dynamic test parametrization to avoid import-time server calls."""
    if "model_name" in metafunc.fixturenames:
        # Only get models when actually running tests, not during import
        try:
            with MLXKnifeServerManager() as server:
                models = get_safe_models_for_system()
                metafunc.parametrize("model_name,size_str,ram_needed", models)
        except Exception as e:
            pytest.skip(f"Cannot set up server for testing: {e}")


if __name__ == "__main__":
    # Quick smoke test - start server first
    print("🦫 MLX Knife Issue #14 Test - Smoke Test")
    print("=" * 50)
    
    # Test server start directly without context manager
    manager = MLXKnifeServerManager()
    success = manager.start_server()
    
    print(f"🏁 Server start result: {success}")
    
    if success:
        try:
            models = get_safe_models_for_system()
            print(f"\n📊 Safe models for this system: {len(models)}")
            
            total_ram = psutil.virtual_memory().total // (1024**3)
            available_ram = psutil.virtual_memory().available // (1024**3)
            print(f"💾 System RAM: {total_ram}GB total, {available_ram}GB available")
            print()
            
            for model, size, ram in models:
                print(f"  🎯 {model}")
                print(f"     └─ Size: {size}, RAM needed: {ram}GB")
            
            print(f"\n🚀 Ready to run: pytest tests/integration/test_issue_14.py -v")
        
        finally:
            manager.stop_server()
    
    else:
        print("💡 Check the logs above for server start failure details")
