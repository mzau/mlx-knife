"""
Test for --log-json CLI flag (ADR-004 improvement).

Minimal test: Verify flag sets MLXK2_LOG_JSON environment variable.
"""

import os
import sys
from unittest.mock import patch, MagicMock


def test_serve_log_json_flag_sets_env_var():
    """--log-json flag should set MLXK2_LOG_JSON=1 environment variable."""
    # Mock start_server to prevent actual server start
    with patch('mlxk2.operations.serve.start_server') as mock_start_server:
        # Simulate CLI invocation: mlxk2 serve --log-json
        test_args = ['mlxk2', 'serve', '--log-json']

        with patch.object(sys, 'argv', test_args):
            # Clear MLXK2_LOG_JSON before test
            os.environ.pop('MLXK2_LOG_JSON', None)

            # Import and run CLI
            from mlxk2.cli import main

            try:
                main()
            except SystemExit:
                pass  # Ignore exit (server would run indefinitely)

            # Verify environment variable was set
            assert os.environ.get('MLXK2_LOG_JSON') == '1', \
                "MLXK2_LOG_JSON should be set to '1' when --log-json flag is present"

            # Verify start_server was called
            assert mock_start_server.called, "start_server should have been called"


def test_serve_without_log_json_flag():
    """Without --log-json, MLXK2_LOG_JSON should remain unset."""
    with patch('mlxk2.operations.serve.start_server') as mock_start_server:
        test_args = ['mlxk2', 'serve']

        with patch.object(sys, 'argv', test_args):
            # Clear MLXK2_LOG_JSON before test
            os.environ.pop('MLXK2_LOG_JSON', None)

            from mlxk2.cli import main

            try:
                main()
            except SystemExit:
                pass

            # Verify environment variable was NOT set
            assert os.environ.get('MLXK2_LOG_JSON') != '1', \
                "MLXK2_LOG_JSON should not be set without --log-json flag"
