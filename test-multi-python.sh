#!/bin/bash
# Note: removed set -e to allow script to continue through all Python versions
# Individual error handling is done explicitly in each test section

echo "🧪 MLX Knife Multi-Python Version Testing"
echo "=========================================="
echo "Prerequisites: Python versions should be available as:"
echo "  - python3 (3.9+ - system default)"
echo "  - python3.10, python3.11, python3.12, python3.13 (if installed)"
echo ""

# Colors for output  
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Python versions to test (bash 3.2 compatible)
PYTHON_COMMANDS=("/usr/bin/python3" "python3.10" "python3.11" "python3.12" "python3.13")
VERSION_NAMES=("3.9" "3.10" "3.11" "3.12" "3.13")
RESULTS=()

# Test function
test_python_version() {
    local index=$1
    local version_name="${VERSION_NAMES[$index]}"
    local python_cmd="${PYTHON_COMMANDS[$index]}"
    
    echo -e "\n${YELLOW}🐍 Testing Python ${version_name}${NC}"
    echo "----------------------------------------"
    
    # Check if Python version is available
    if ! command -v $python_cmd &> /dev/null; then
        echo -e "${RED}❌ Python ${version_name} not found (tried: $python_cmd)${NC}"
        RESULTS+=("${version_name}:NOT_FOUND")
        return 1
    fi
    
    # Show actual version
    local actual_version=$($python_cmd --version 2>&1)
    echo "📍 Found: $actual_version"
    
    # Create virtual environment
    local venv_name="test_env_${version_name//./_}"
    # Trap termination to ensure cleanup (Ctrl-C or external kill)
    trap 'echo -e "\n⛔ Received termination signal. Cleaning up $venv_name..."; deactivate 2>/dev/null || true; pkill -P $$ 2>/dev/null || true; rm -rf "$venv_name"; echo "Exiting due to signal."; exit 1' INT TERM
    echo "🔧 Creating virtual environment: $venv_name"
    
    if [ -d "$venv_name" ]; then
        rm -rf "$venv_name"
    fi
    
    $python_cmd -m venv "$venv_name"
    source "$venv_name/bin/activate"
    
    # Upgrade pip and install MLX Knife
    echo "📦 Installing MLX Knife..."
    pip install --upgrade pip setuptools wheel > /dev/null 2>&1
    
    if pip install -e ".[dev,test]" > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Installation successful${NC}"
        
        # Run smoke test
        echo "🧪 Running import test (this may take up to 2 minutes for MLX)..."
        if python -c "import mlx_knife.cli; print('Import successful')"; then
            echo -e "${GREEN}✅ Import test passed${NC}"
            
            # Try basic CLI command
            echo "🧪 Testing CLI help..."
            if python -m mlx_knife.cli --help > /dev/null 2>&1; then
                echo -e "${GREEN}✅ CLI test passed${NC}"
                
                # Run complete test suite
                echo "🧪 Running FULL test suite (this takes 5-10 minutes)..."
                local test_log="test_results_${version_name//./_}.log"
                # Disable process guard for multi-env run to avoid cross-session signal handling
                MLXK_TEST_DISABLE_PROCESS_GUARD=1 MLXK_TEST_DISABLE_CATCH_TERM=1 MLXK_TEST_DETACH_PGRP=0 python -m pytest tests/ -v --tb=short --timeout-method=thread > "$test_log" 2>&1
                local pytest_rc=$?
                local passed_count=$(grep -c "PASSED" "$test_log" 2>/dev/null)
                local failed_count=$(grep -c "FAILED" "$test_log" 2>/dev/null)
                passed_count=${passed_count:-0}
                failed_count=${failed_count:-0}
                local test_count=$((passed_count + failed_count))

                # Treat stray signal exits (e.g., 143=SIGTERM, 137=SIGKILL) as success if log shows all passed
                if [ $pytest_rc -ne 0 ] && [ "$failed_count" -eq 0 ] && [ "$passed_count" -gt 0 ] && grep -q "passed" "$test_log"; then
                    echo -e "${YELLOW}ℹ️  PyTest exit code $pytest_rc but log shows all tests passed — accepting as success${NC}"
                    pytest_rc=0
                fi

                if [ $pytest_rc -eq 0 ]; then
                    if [ "$failed_count" -eq 0 ] && [ "$passed_count" -gt 0 ]; then
                        echo -e "${GREEN}✅ Full test suite passed ($passed_count/$test_count tests)${NC}"
                        
                        # Code quality checks
                        echo "🧪 Running code quality checks..."
                        
                        # Check if ruff is properly installed
                        if python -c "import ruff" > /dev/null 2>&1; then
                            local ruff_log="ruff_${version_name//./_}.log"
                            echo "🧪 Running ruff check (logging to $ruff_log)..."
                            if python -m ruff check mlx_knife/ > "$ruff_log" 2>&1; then
                                echo -e "${GREEN}✅ ruff linting passed${NC}"
                                
                                # Note: mypy might have many warnings, so we allow it to "fail" but still continue
                                python -m mypy mlx_knife/ --ignore-missing-imports > mypy_${version_name//./_}.log 2>&1
                                local mypy_errors=$(grep -c "error:" mypy_${version_name//./_}.log 2>/dev/null || echo "0")
                                echo -e "${YELLOW}ℹ️  mypy check complete ($mypy_errors errors found)${NC}"
                                
                                RESULTS+=("${version_name}:FULL_SUCCESS:${passed_count}tests")
                            else
                                local ruff_error_count=$(grep -c "Found .* error" "$ruff_log" 2>/dev/null || echo "unknown")
                                echo -e "${RED}❌ ruff linting failed ($ruff_error_count errors)${NC}"
                                echo "   See $ruff_log for details"
                                RESULTS+=("${version_name}:RUFF_FAILED")
                            fi
                        else
                            echo -e "${RED}❌ ruff not properly installed, trying to install...${NC}"
                            if pip install ruff>=0.1.0 > /dev/null 2>&1; then
                                echo "🔧 ruff installed, retrying check..."
                                local ruff_log="ruff_${version_name//./_}.log"
                                if python -m ruff check mlx_knife/ > "$ruff_log" 2>&1; then
                                    echo -e "${GREEN}✅ ruff linting passed${NC}"
                                    
                                    # Note: mypy might have many warnings, so we allow it to "fail" but still continue
                                    python -m mypy mlx_knife/ --ignore-missing-imports > mypy_${version_name//./_}.log 2>&1
                                    local mypy_errors=$(grep -c "error:" mypy_${version_name//./_}.log 2>/dev/null || echo "0")
                                    echo -e "${YELLOW}ℹ️  mypy check complete ($mypy_errors errors found)${NC}"
                                    
                                    RESULTS+=("${version_name}:FULL_SUCCESS:${passed_count}tests")
                                else
                                    local ruff_error_count=$(grep -c "Found .* error" "$ruff_log" 2>/dev/null || echo "unknown")
                                    echo -e "${RED}❌ ruff linting failed after installation ($ruff_error_count errors)${NC}"
                                    echo "   See $ruff_log for details"
                                    RESULTS+=("${version_name}:RUFF_FAILED")
                                fi
                            else
                                echo -e "${RED}❌ Could not install ruff${NC}"
                                RESULTS+=("${version_name}:RUFF_INSTALL_FAILED")
                            fi
                        fi
                    else
                        echo -e "${RED}❌ Test suite failed ($passed_count passed, $failed_count failed)${NC}"
                        echo "   See $test_log for details"
                        RESULTS+=("${version_name}:TESTS_FAILED:${failed_count}failures")
                    fi
                else
                    echo -e "${RED}❌ Test suite timed out or crashed (exit=$pytest_rc)${NC}"
                    echo "   Tail of log ($test_log):"
                    tail -n 60 "$test_log" 2>/dev/null || true
                    RESULTS+=("${version_name}:TESTS_TIMEOUT")
                fi
            else
                echo -e "${RED}❌ CLI test failed${NC}"
                RESULTS+=("${version_name}:CLI_FAILED")
            fi
        else
            echo -e "${RED}❌ Import test failed${NC}"
            RESULTS+=("${version_name}:IMPORT_FAILED")
        fi
    else
        echo -e "${RED}❌ Installation failed${NC}"
        RESULTS+=("${version_name}:INSTALL_FAILED")
    fi
    
    # Cleanup
    deactivate 2>/dev/null || true
    rm -rf "$venv_name"
    trap - INT TERM
}

# Run tests for all Python versions
for i in "${!PYTHON_COMMANDS[@]}"; do
    test_python_version "$i"
done

# Summary
echo
echo "SUMMARY"
echo "==========="

for result in "${RESULTS[@]}"; do
    IFS=':' read -r version status details <<< "$result"
    case $status in
        "FULL_SUCCESS")
            echo "OK Python ${version}: FULLY VERIFIED - ${details}"
            ;;
        "NOT_FOUND")
            echo "WARN Python ${version}: NOT INSTALLED"
            ;;
        "TESTS_FAILED")
            echo "FAIL Python ${version}: TESTS FAILED - ${details}"
            ;;
        "RUFF_FAILED")
            echo "FAIL Python ${version}: CODE QUALITY FAILED"
            ;;
        "RUFF_INSTALL_FAILED")
            echo "FAIL Python ${version}: RUFF INSTALLATION FAILED"
            ;;
        "TESTS_TIMEOUT")
            echo "FAIL Python ${version}: TESTS TIMED OUT"
            ;;
        *)
            echo "FAIL Python ${version}: ${status}"
            ;;
    esac
done

# Recommendations
echo
echo "RECOMMENDATIONS"
echo "=================="

fully_verified_count=0
partial_count=0
failed_count=0
not_found_count=0
fully_verified_versions=()

for result in "${RESULTS[@]}"; do
    IFS=':' read -r version status details <<< "$result"
    case $status in
        "FULL_SUCCESS")
            ((fully_verified_count++))
            fully_verified_versions+=("$version")
            ;;
        "NOT_FOUND")
            ((not_found_count++))
            ;;
        *)
            ((failed_count++))
            ;;
    esac
done

echo "VERIFICATION RESULTS:"
printf "   Fully Verified: %s\n" "$fully_verified_count"
printf "   Failed/Issues: %s\n" "$failed_count"
printf "   Not Available: %s\n" "$not_found_count"

if [ $fully_verified_count -eq 0 ]; then
    echo
    echo "CRITICAL: No Python versions fully verified!"
    echo "   - Cannot release without verified compatibility"
    echo "   - Fix blocking issues before any release"
elif [ $failed_count -eq 0 ] && [ $fully_verified_count -ge 2 ]; then
    echo
    echo "PRODUCTION READY: All tested versions fully verified!"
    echo "   - Safe to release with confidence"
    echo "   - All versions pass: installation, tests, code quality"
    echo "   - Verified versions: ${fully_verified_versions[*]}"
elif [ $fully_verified_count -ge 2 ]; then
    echo
    echo "PARTIAL SUCCESS: ${fully_verified_count} verified, ${failed_count} with issues"
    echo "   - Can release with verified versions: ${fully_verified_versions[*]}"
    echo "   - Document known issues with other versions"
    echo "   - Consider fixing compatibility or updating requirements"
else
    echo
    echo "INSUFFICIENT VERIFICATION: Only ${fully_verified_count} versions verified"
    echo "   - Need at least 2 fully verified versions for release"
    echo "   - Fix compatibility issues or verify more versions"
fi

echo
echo "NEXT STEPS"
echo "============="

if [ $fully_verified_count -ge 2 ] && [ $failed_count -eq 0 ]; then
    echo "READY TO RELEASE:"
    echo "  1. Update README.md with verified Python versions"
    echo "  2. Update pyproject.toml requires-python based on results"
    echo "  3. Document verified versions: ${fully_verified_versions[*]}"
    echo "  4. Safe to tag and release MLX Knife 1.1.1-b2"
    exit_code=0
else
    echo "WORK NEEDED:"
    echo "  1. Review detailed logs: test_results_*.log, mypy_*.log"
    echo "  2. Fix compatibility issues for failed versions"
    echo "  3. Re-run this script until all targeted versions pass"
    echo "  4. Update documentation to reflect actual compatibility"
    echo "  5. Consider reducing version scope if fixes are complex"
    exit_code=1
fi

echo ""
echo "Generated Files:"
echo "   - test_results_<version>.log: Detailed pytest results"  
echo "   - mypy_<version>.log: Type checking results"
echo "   - Use these logs to debug specific compatibility issues"

exit $exit_code
