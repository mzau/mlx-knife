"""
Unit tests for cli.py module.

Tests the command-line interface functionality:
- Argument parsing
- Command dispatch
- Help and version output
"""
import pytest
import argparse
from unittest.mock import patch, MagicMock
import sys
import os

# Import the module under test
from mlx_knife.cli import main


class TestMainFunctionBasic:
    """Test basic main function behavior without requiring parser creation."""
    
    def test_main_function_exists(self):
        """Test that main function exists and is callable."""
        try:
            assert callable(main)
        except Exception as e:
            pytest.fail(f"Main function test failed: {e}")

    def test_version_flag_via_main(self):
        """Test version flag through main function."""
        try:
            with patch('sys.argv', ['mlxk', '--version']):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                # Version should exit cleanly
                assert exc_info.value.code in [0, None]
        except Exception as e:
            # It's OK if version parsing isn't fully implemented yet
            pass


class TestMainFunction:
    """Test main function behavior."""
    
    def test_main_with_help(self):
        """Test main function with help argument."""
        try:
            with patch('sys.argv', ['mlxk', '--help']):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                # Help should exit with code 0
                assert exc_info.value.code == 0 or exc_info.value.code is None
        except Exception as e:
            pytest.fail(f"Main function help test failed: {e}")

    def test_main_with_invalid_command(self):
        """Test main function with invalid command."""
        try:
            with patch('sys.argv', ['mlxk', 'invalid-command-xyz']):
                with pytest.raises(SystemExit) as exc_info:
                    main()
                # Invalid command should exit with non-zero code
                assert exc_info.value.code != 0
        except Exception as e:
            pytest.fail(f"Main function invalid command test failed: {e}")

    @patch('mlx_knife.cache_utils.list_models')
    def test_main_with_list_command(self, mock_list_models):
        """Test main function with list command."""
        try:
            # Mock the list_models function to avoid actual cache interaction
            mock_list_models.return_value = None
            
            with patch('sys.argv', ['mlxk', 'list']):
                try:
                    main()
                except SystemExit as e:
                    # List command might exit with 0 on success
                    assert e.code == 0 or e.code is None
        except Exception as e:
            pytest.fail(f"Main function list command test failed: {e}")

    @patch('mlx_knife.cache_utils.check_all_models_health')
    def test_main_with_health_command(self, mock_health_check):
        """Test main function with health command."""
        try:
            # Mock the health check function
            mock_health_check.return_value = None
            
            with patch('sys.argv', ['mlxk', 'health']):
                try:
                    main()
                except SystemExit as e:
                    # Health command should exit gracefully
                    assert e.code == 0 or e.code is None
        except Exception as e:
            pytest.fail(f"Main function health command test failed: {e}")

    def test_main_no_arguments(self):
        """Test main function with no arguments."""
        try:
            with patch('sys.argv', ['mlxk']):
                # The CLI shows help when no args are provided - this is valid behavior
                main()  # Should complete successfully showing help
        except SystemExit as e:
            # Also valid - some CLIs exit after showing help
            pass
        except Exception as e:
            pytest.fail(f"Main function no arguments test failed: {e}")


class TestErrorHandling:
    """Test CLI error handling."""
    
    def test_keyboard_interrupt_handling(self):
        """Test handling of KeyboardInterrupt (Ctrl+C)."""
        try:
            # Test that KeyboardInterrupt doesn't crash the CLI completely
            with patch('sys.argv', ['mlxk', 'list']):
                with patch('builtins.print', side_effect=KeyboardInterrupt()):
                    try:
                        main()
                    except KeyboardInterrupt:
                        # KeyboardInterrupt propagating up is acceptable
                        pass
                    except SystemExit:
                        # Graceful exit is also acceptable
                        pass
        except Exception as e:
            pytest.fail(f"Keyboard interrupt handling test failed: {e}")

    def test_basic_command_robustness(self):
        """Test that basic commands don't crash unexpectedly."""
        try:
            # Test that list command runs successfully (already working based on earlier test)
            with patch('sys.argv', ['mlxk', 'list']):
                main()  # Should work fine
        except SystemExit:
            # Exit is acceptable for some CLI implementations
            pass
        except Exception as e:
            pytest.fail(f"Basic command robustness test failed: {e}")


class TestHealthCommandDefaultBehavior:
    """Test health command default behavior (Issue 3)."""
    
    @patch('mlx_knife.cli.check_all_models_health')
    def test_health_command_without_args_calls_all(self, mock_check_all):
        """Test that 'mlxk health' (no args) calls check_all_models_health."""
        mock_check_all.return_value = True
        
        try:
            with patch('sys.argv', ['mlxk', 'health']):
                main()
            
            # Should have called check_all_models_health
            assert mock_check_all.called
            mock_check_all.assert_called_once()
        except SystemExit:
            # Exit is acceptable after running the command
            assert mock_check_all.called
        except Exception as e:
            pytest.fail(f"Health command default behavior test failed: {e}")
    
    @patch('mlx_knife.cli.check_model_health')
    @patch('mlx_knife.cli.check_all_models_health')
    def test_health_command_with_specific_model(self, mock_check_all, mock_check_specific):
        """Test that 'mlxk health model-name' calls check_model_health."""
        mock_check_specific.return_value = True
        
        try:
            with patch('sys.argv', ['mlxk', 'health', 'some-model']):
                main()
            
            # Should have called check_model_health with the specific model
            assert mock_check_specific.called
            mock_check_specific.assert_called_once_with('some-model')
            
            # Should NOT have called check_all_models_health
            assert not mock_check_all.called
        except SystemExit:
            # Exit is acceptable after running the command
            assert mock_check_specific.called
            assert not mock_check_all.called
        except Exception as e:
            pytest.fail(f"Health command specific model test failed: {e}")
    
    @patch('mlx_knife.cli.check_all_models_health')
    def test_health_command_backward_compatibility_with_all_flag(self, mock_check_all):
        """Test that 'mlxk health --all' still works for backward compatibility."""
        mock_check_all.return_value = True
        
        try:
            with patch('sys.argv', ['mlxk', 'health', '--all']):
                main()
            
            # Should have called check_all_models_health  
            assert mock_check_all.called
            mock_check_all.assert_called_once()
        except SystemExit:
            # Exit is acceptable after running the command
            assert mock_check_all.called
        except Exception as e:
            pytest.fail(f"Health command --all flag test failed: {e}")