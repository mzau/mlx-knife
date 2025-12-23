"""MLX-Knife - HuggingFace model management for MLX."""

# Suppress urllib3 LibreSSL warning on macOS system Python 3.9
# (must run before any imports that may indirectly import urllib3)
import warnings

# Issue parity with 1.1.0 (Issue #22)
warnings.filterwarnings('ignore', message='urllib3 v2 only supports OpenSSL 1.1.1+')

__version__ = "2.0.4b3"
