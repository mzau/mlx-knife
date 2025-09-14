"""Lightweight test stub for mlx.core to avoid native deps in unit tests.

Only implements the minimal API surface used by the 2.0 unit tests and runner:
- zeros(n)
- array(x)
- clear_cache()
- get_active_memory()
"""

class _Array:
    def __init__(self, data):
        self._data = data

    def item(self):
        # mimic behavior of mx.array([...]).item() -> first element
        if isinstance(self._data, (list, tuple)):
            return self._data[0]
        return self._data


def zeros(n):
    # Return a simple Python list as a stand-in
    return [0] * (n if isinstance(n, int) else 1)


def array(x):
    # Wrap in simple array-like with .item()
    return _Array(x if isinstance(x, (list, tuple)) else [x])


def clear_cache():
    # No-op for tests
    return None


def get_active_memory():
    # Return a tiny deterministic value (bytes)
    return 0

