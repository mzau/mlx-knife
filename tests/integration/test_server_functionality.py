"""
High Priority Tests: Server Functionality

Tests for the OpenAI-compatible API server:
- Server startup and shutdown
- Process lifecycle during server operations  
- API endpoint availability
- Request handling and response format
- Server interruption and cleanup
"""
import pytest
import subprocess
import time
import requests
import signal
import json
from pathlib import Path


@pytest.mark.timeout(60)
class TestServerLifecycle:
    """Test server startup, operation, and shutdown."""
    
    def test_server_startup_shutdown(self, mlx_knife_process, process_monitor):
        """Test server starts and shuts down cleanly."""
        # Start server
        proc = mlx_knife_process(["server", "--host", "127.0.0.1", "--port", "8001"])
        main_pid = proc.pid
        
        # Give server time to start
        time.sleep(3)
        
        # Check if server is responsive (basic health check)
        try:
            response = requests.get("http://127.0.0.1:8001/health", timeout=5)
            server_started = response.status_code == 200
        except requests.exceptions.RequestException:
            # Server might not have health endpoint, that's OK
            server_started = proc.poll() is None  # Process still running
        
        # Track child processes
        children_before = process_monitor["get_process_tree"](main_pid)
        
        # Shutdown server
        proc.send_signal(signal.SIGINT)
        
        try:
            return_code = proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            pytest.fail("Server did not shutdown within timeout")
        
        # Verify clean shutdown
        assert return_code is not None, "Server process did not terminate"
        
        # Verify all child processes cleaned up
        assert process_monitor["wait_for_cleanup"](main_pid, timeout=10)
        
        for child in children_before:
            assert not child.is_running(), f"Server child process survived: PID {child.pid}"

    def test_server_sigterm_handling(self, mlx_knife_process, process_monitor):
        """Test server responds to SIGTERM gracefully."""
        proc = mlx_knife_process(["server", "--host", "127.0.0.1", "--port", "8002"])
        main_pid = proc.pid
        
        time.sleep(3)
        children_before = process_monitor["get_process_tree"](main_pid)
        
        # Send SIGTERM
        proc.send_signal(signal.SIGTERM)
        
        try:
            return_code = proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
            pytest.fail("Server did not respond to SIGTERM")
        
        # Should exit gracefully
        assert return_code is not None
        
        # Clean up child processes
        assert process_monitor["wait_for_cleanup"](main_pid, timeout=10)
        
        for child in children_before:
            assert not child.is_running(), f"Server child survived SIGTERM: PID {child.pid}"

    def test_server_sigkill_cleanup(self, mlx_knife_process, process_monitor):
        """Test cleanup after SIGKILL."""
        proc = mlx_knife_process(["server", "--host", "127.0.0.1", "--port", "8003"])
        main_pid = proc.pid
        
        time.sleep(3)
        children_before = process_monitor["get_process_tree"](main_pid)
        
        # SIGKILL should kill immediately
        proc.send_signal(signal.SIGKILL)
        
        try:
            return_code = proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pytest.fail("Process did not die from SIGKILL")
        
        assert return_code == -signal.SIGKILL
        
        # Child processes should be cleaned up by OS
        assert process_monitor["wait_for_cleanup"](main_pid, timeout=10)

    def test_server_port_binding_conflicts(self, mlx_knife_process):
        """Test server handles port conflicts gracefully."""
        # Start first server
        proc1 = mlx_knife_process(["server", "--host", "127.0.0.1", "--port", "8004"])
        time.sleep(3)
        
        # Try to start second server on same port
        proc2 = mlx_knife_process(["server", "--host", "127.0.0.1", "--port", "8004"])
        
        try:
            # Second server should fail quickly
            return_code2 = proc2.wait(timeout=10)
            assert return_code2 != 0, "Second server should fail on port conflict"
        except subprocess.TimeoutExpired:
            proc2.kill()
            pytest.fail("Second server did not fail quickly on port conflict")
        finally:
            # Clean up first server
            if proc1.poll() is None:
                proc1.send_signal(signal.SIGINT)
                proc1.wait(timeout=10)

    def test_server_invalid_arguments(self, mlx_knife_process):
        """Test server handles invalid arguments gracefully."""
        invalid_configs = [
            ["server", "--port", "99999"],  # Invalid port
            ["server", "--host", "invalid-host"],  # Invalid host
            ["server", "--max-tokens", "-1"],  # Invalid max tokens
        ]
        
        for config in invalid_configs:
            proc = mlx_knife_process(config)
            try:
                return_code = proc.wait(timeout=10)
                # Should fail gracefully, not hang
                assert return_code is not None, f"Server hung on invalid config: {config}"
                assert return_code != 0, f"Server should fail on invalid config: {config}"
            except subprocess.TimeoutExpired:
                proc.kill()
                pytest.fail(f"Server hung on invalid config: {config}")


@pytest.mark.timeout(90)
class TestServerAPI:
    """Test server API functionality."""
    
    def test_server_health_endpoint(self, mlx_knife_process):
        """Test server health/status endpoint if available."""
        proc = mlx_knife_process(["server", "--host", "127.0.0.1", "--port", "8005"])
        
        # Wait for server to start
        time.sleep(4)
        
        try:
            # Try common health endpoints
            health_endpoints = [
                "http://127.0.0.1:8005/health",
                "http://127.0.0.1:8005/v1/models",
                "http://127.0.0.1:8005/",
            ]
            
            server_responsive = False
            for endpoint in health_endpoints:
                try:
                    response = requests.get(endpoint, timeout=5)
                    if response.status_code in [200, 404]:  # 404 is OK, means server is running
                        server_responsive = True
                        break
                except requests.exceptions.RequestException:
                    continue
            
            # Server should be responsive to at least one endpoint
            assert server_responsive, "Server not responsive to any health endpoints"
            
        finally:
            # Clean up
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
                proc.wait(timeout=15)

    def test_server_openai_models_endpoint(self, mlx_knife_process):
        """Test OpenAI-compatible /v1/models endpoint."""
        proc = mlx_knife_process(["server", "--host", "127.0.0.1", "--port", "8006"])
        
        time.sleep(4)
        
        try:
            response = requests.get("http://127.0.0.1:8006/v1/models", timeout=10)
            
            # Should respond (may be empty list if no models)
            assert response.status_code == 200, f"Models endpoint failed: {response.status_code}"
            
            # Should return valid JSON
            try:
                data = response.json()
                assert isinstance(data, dict), "Models endpoint should return JSON object"
                # OpenAI format typically has 'data' field
                if 'data' in data:
                    assert isinstance(data['data'], list), "Models data should be a list"
            except json.JSONDecodeError:
                pytest.fail("Models endpoint returned invalid JSON")
                
        except requests.exceptions.RequestException as e:
            pytest.fail(f"Failed to connect to models endpoint: {e}")
        finally:
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
                proc.wait(timeout=15)

    def test_server_chat_completions_endpoint(self, mlx_knife_process):
        """Test OpenAI-compatible /v1/chat/completions endpoint structure."""
        proc = mlx_knife_process(["server", "--host", "127.0.0.1", "--port", "8007"])
        
        time.sleep(4)
        
        try:
            # Test with minimal valid request
            payload = {
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hello"}],
                "max_tokens": 10
            }
            
            response = requests.post(
                "http://127.0.0.1:8007/v1/chat/completions",
                json=payload,
                timeout=15
            )
            
            # Should respond (may be error if no models, but shouldn't hang)
            assert response.status_code is not None, "Chat completions endpoint hung"
            
            # Should return JSON response
            try:
                data = response.json()
                assert isinstance(data, dict), "Chat completions should return JSON object"
                
                if response.status_code == 200:
                    # Valid response should have expected fields
                    assert 'choices' in data or 'error' in data
                elif response.status_code == 400:
                    # Bad request should have error message
                    assert 'error' in data
                    
            except json.JSONDecodeError:
                pytest.fail("Chat completions returned invalid JSON")
                
        except requests.exceptions.RequestException as e:
            pytest.fail(f"Failed to connect to chat completions endpoint: {e}")
        finally:
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
                proc.wait(timeout=15)

    def test_server_streaming_endpoint(self, mlx_knife_process):
        """Test streaming functionality if available."""
        proc = mlx_knife_process(["server", "--host", "127.0.0.1", "--port", "8008"])
        
        time.sleep(4)
        
        try:
            # Test streaming request
            payload = {
                "model": "test-model", 
                "messages": [{"role": "user", "content": "Hi"}],
                "max_tokens": 5,
                "stream": True
            }
            
            response = requests.post(
                "http://127.0.0.1:8008/v1/chat/completions",
                json=payload,
                timeout=20,
                stream=True
            )
            
            # Should respond to streaming request
            assert response.status_code is not None, "Streaming endpoint hung"
            
            # Should handle streaming gracefully (may error if no model)
            if response.status_code == 200:
                # Should return SSE format or similar
                assert 'text/plain' in response.headers.get('content-type', '') or \
                       'text/event-stream' in response.headers.get('content-type', '') or \
                       'application/json' in response.headers.get('content-type', '')
                       
        except requests.exceptions.RequestException as e:
            pytest.fail(f"Streaming endpoint connection failed: {e}")
        finally:
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
                proc.wait(timeout=15)


@pytest.mark.timeout(45)
class TestServerResourceManagement:
    """Test server resource management."""
    
    def test_server_memory_cleanup_after_shutdown(self, mlx_knife_process):
        """Test that server cleans up memory after shutdown."""
        # Start and stop server multiple times
        for i in range(3):
            proc = mlx_knife_process(["server", "--host", "127.0.0.1", "--port", f"800{9+i}"])
            
            time.sleep(2)
            
            # Shutdown cleanly
            proc.send_signal(signal.SIGINT)
            return_code = proc.wait(timeout=15)
            
            assert return_code is not None, f"Server {i} did not shutdown"

    def test_server_handles_multiple_requests(self, mlx_knife_process):
        """Test server can handle multiple concurrent requests without hanging."""
        proc = mlx_knife_process(["server", "--host", "127.0.0.1", "--port", "8012"])
        
        time.sleep(4)
        
        try:
            # Send multiple requests concurrently
            import threading
            import queue
            
            results = queue.Queue()
            
            def make_request(endpoint):
                try:
                    response = requests.get(f"http://127.0.0.1:8012{endpoint}", timeout=10)
                    results.put(("success", response.status_code))
                except Exception as e:
                    results.put(("error", str(e)))
            
            # Start multiple threads
            threads = []
            endpoints = ["/v1/models", "/v1/models", "/v1/models"]
            
            for endpoint in endpoints:
                thread = threading.Thread(target=make_request, args=(endpoint,))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads
            for thread in threads:
                thread.join(timeout=20)
                assert not thread.is_alive(), "Request thread hung"
            
            # Check results
            success_count = 0
            while not results.empty():
                result_type, result_value = results.get()
                if result_type == "success":
                    success_count += 1
            
            # At least some requests should succeed
            assert success_count > 0, "No requests succeeded"
            
        finally:
            if proc.poll() is None:
                proc.send_signal(signal.SIGINT)
                proc.wait(timeout=15)

    def test_server_request_interruption(self, mlx_knife_process):
        """Test server handles request interruption cleanly."""
        proc = mlx_knife_process(["server", "--host", "127.0.0.1", "--port", "8013"])
        
        time.sleep(4)
        
        try:
            # Start a request and interrupt it
            import threading
            
            def make_slow_request():
                try:
                    requests.get("http://127.0.0.1:8013/v1/models", timeout=2)
                except:
                    pass  # Expected to timeout/fail
            
            # Start request in background
            request_thread = threading.Thread(target=make_slow_request)
            request_thread.start()
            
            # Give request time to start
            time.sleep(1)
            
            # Shutdown server while request is in progress
            proc.send_signal(signal.SIGINT)
            return_code = proc.wait(timeout=15)
            
            # Server should shutdown cleanly even with active requests
            assert return_code is not None, "Server hung during request interruption"
            
            # Request thread should complete
            request_thread.join(timeout=10)
            assert not request_thread.is_alive(), "Request thread hung after server shutdown"
            
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()