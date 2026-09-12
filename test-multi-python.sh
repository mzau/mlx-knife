#!/bin/bash
# Note: removed set -e to allow script to continue through all Python versions
# Individual error handling is done explicitly in each test section

echo "🧪 MLX Knife 2.0 (mlxk2) Multi-Python Version Testing"
echo "=========================================="
echo "Prerequisites: Python versions should be available as:"
echo "  - python3.10, python3.11, python3.12, python3.13, python3.14 (full support: text + vision + audio)"
echo "Note: Python 3.9 not supported (MLX 0.30+ requires 3.10+)"
echo "Note: an interpreter that is not on PATH is reported as 'Not Available', not as a failure"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Python versions to test (bash 3.2 compatible)
# Note: Python 3.9 dropped (MLX 0.30+ requires 3.10+)
# Note: versioned installs do not always expose the interpreter under these names;
#       a missing one is counted as "Not Available" and does not fail the run.
PYTHON_COMMANDS=("python3.10" "python3.11" "python3.12" "python3.13" "python3.14")
VERSION_NAMES=("3.10" "3.11" "3.12" "3.13" "3.14")
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
    echo "🔧 Creating virtual environment: $venv_name"
    
    if [ -d "$venv_name" ]; then
        rm -rf "$venv_name"
    fi
    
    $python_cmd -m venv "$venv_name"
    source "$venv_name/bin/activate"
    
    # Upgrade pip and install MLX Knife
    echo "📦 Installing MLX Knife (2.0) ..."
    local install_log="install_${version_name//./_}.log"
    pip install --upgrade pip setuptools wheel > "$install_log" 2>&1

    # Install with vision + audio support (all supported versions are 3.10+)
    local install_extras=".[test,vision,audio]"
    echo "   Including vision + audio support (Python $version_name)"

    if pip install -e "$install_extras" >> "$install_log" 2>&1; then
        echo -e "${GREEN}✅ Installation successful${NC}"
        echo "🧰 Ensuring tooling (ruff, mypy)..."
        pip install -q "ruff>=0.1.0" "mypy>=1.5.0" >> "$install_log" 2>&1 || true
        
        # Run smoke test
        echo "🧪 Running import test (mlxk2)..."
        if python -c "import mlxk2, mlxk2.cli; print('Import successful')"; then
            echo -e "${GREEN}✅ Import test passed${NC}"
            
            # Try basic CLI command
            echo "🧪 Testing CLI version (JSON)..."
            if python -m mlxk2.cli --version --json > /dev/null 2>&1; then
                echo -e "${GREEN}✅ CLI test (version) passed${NC}"
                
                # Run complete test suite (exclude ALL live tests requiring models/network)
                echo "🧪 Running 2.0 test suite..."
                local test_log="test_results_${version_name//./_}.log"
                if python -m pytest tests_2.0/ -m "not live" -v --tb=short > "$test_log" 2>&1; then
                    local passed_count=$(grep -c "PASSED" "$test_log" 2>/dev/null)
                    local failed_count=$(grep -c "FAILED" "$test_log" 2>/dev/null)
                    passed_count=${passed_count:-0}
                    failed_count=${failed_count:-0}
                    local test_count=$((passed_count + failed_count))
                    
                    if [ "$failed_count" -eq 0 ] && [ "$passed_count" -gt 0 ]; then
                        echo -e "${GREEN}✅ Full test suite passed ($passed_count/$test_count tests)${NC}"
                        
                        # Code quality checks
                        echo "🧪 Running code quality checks (mlxk2)..."
                        
                        local ruff_log="ruff_${version_name//./_}.log"
                        echo "🧪 Running ruff check on mlxk2 (logging to $ruff_log)..."
                        if python -m ruff check mlxk2/ > "$ruff_log" 2>&1; then
                            echo -e "${GREEN}✅ ruff linting passed${NC}"
                            
                            # mypy carries a pre-existing error count; the gate is on growth only,
                            # against the total in scripts/mypy-baseline.txt (see that file).
                            python -m mypy mlxk2/ > mypy_${version_name//./_}.log 2>&1
                            local mypy_errors=$(grep -c "error:" mypy_${version_name//./_}.log 2>/dev/null || echo "0")
                            local mypy_baseline=$(awk '/^total /{print $2}' scripts/mypy-baseline.txt 2>/dev/null)

                            if [ -n "$mypy_baseline" ] && [ "$mypy_errors" -gt "$mypy_baseline" ]; then
                                echo -e "${RED}❌ mypy grew: $mypy_errors errors, baseline $mypy_baseline${NC}"
                                echo "   See mypy_${version_name//./_}.log — fix the new ones, or refresh"
                                echo "   scripts/mypy-baseline.txt deliberately if the growth is intended"
                                RESULTS+=("${version_name}:MYPY_GREW:${mypy_errors}errors")
                            else
                                if [ -n "$mypy_baseline" ] && [ "$mypy_errors" -lt "$mypy_baseline" ]; then
                                    # Fewer can also mean a leaner venv, not cleaner code: with
                                    # ignore_missing_imports an absent package silences its callers.
                                    echo -e "${GREEN}✅ mypy $mypy_errors errors, below the baseline of $mypy_baseline — refresh scripts/mypy-baseline.txt after a deliberate cleanup${NC}"
                                else
                                    echo -e "${YELLOW}ℹ️  mypy $mypy_errors errors (baseline ${mypy_baseline:-none}, gate on growth only)${NC}"
                                fi
                                RESULTS+=("${version_name}:FULL_SUCCESS:${passed_count}tests")
                            fi
                        else
                            local ruff_error_count=$(grep -c "Found .* error" "$ruff_log" 2>/dev/null || echo "unknown")
                            echo -e "${RED}❌ ruff linting failed ($ruff_error_count errors)${NC}"
                            echo "   See $ruff_log for details"
                            RESULTS+=("${version_name}:RUFF_FAILED")
                        fi
                    else
                        echo -e "${RED}❌ Test suite failed ($passed_count passed, $failed_count failed)${NC}"
                        echo "   See $test_log for details"
                        RESULTS+=("${version_name}:TESTS_FAILED:${failed_count}failures")
                    fi
                else
                    echo -e "${RED}❌ Test suite timed out or crashed${NC}"
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
        echo "   See $install_log for details"
        RESULTS+=("${version_name}:INSTALL_FAILED")
    fi
    
    # Cleanup
    deactivate 2>/dev/null || true
    rm -rf "$venv_name"
}

# Run tests for all Python versions
for i in "${!PYTHON_COMMANDS[@]}"; do
    test_python_version "$i"
done

# Summary
echo -e "\n${YELLOW}📊 SUMMARY${NC}"
echo "==========="

for result in "${RESULTS[@]}"; do
    IFS=':' read -r version status details <<< "$result"
    case $status in
        "FULL_SUCCESS")
            echo -e "${GREEN}✅ Python $version: FULLY VERIFIED ($details)${NC}"
            ;;
        "NOT_FOUND")
            echo -e "${YELLOW}⚠️  Python $version: NOT INSTALLED${NC}"
            ;;
        "TESTS_FAILED")
            echo -e "${RED}❌ Python $version: TESTS FAILED ($details)${NC}"
            ;;
        "RUFF_FAILED")
            echo -e "${RED}❌ Python $version: CODE QUALITY FAILED${NC}"
            ;;
        "RUFF_INSTALL_FAILED")
            echo -e "${RED}❌ Python $version: RUFF INSTALLATION FAILED${NC}"
            ;;
        "TESTS_TIMEOUT")
            echo -e "${RED}❌ Python $version: TESTS TIMED OUT${NC}"
            ;;
        *)
            echo -e "${RED}❌ Python $version: $status${NC}"
            ;;
    esac
done

# Recommendations
echo -e "\n${YELLOW}💡 RECOMMENDATIONS${NC}"
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

echo -e "${YELLOW}📊 VERIFICATION RESULTS:${NC}"
echo "   Fully Verified: $fully_verified_count"
echo "   Failed/Issues: $failed_count" 
echo "   Not Available: $not_found_count"

if [ $fully_verified_count -eq 0 ]; then
    echo -e "\n${RED}🚨 CRITICAL: No Python versions fully verified!${NC}"
    echo "   → Cannot release without verified compatibility"
    echo "   → Fix blocking issues before any release"
elif [ $failed_count -eq 0 ] && [ $fully_verified_count -ge 2 ]; then
    echo -e "\n${GREEN}🎉 PRODUCTION READY: All tested versions fully verified!${NC}"
    echo "   → Safe to release with confidence"
    echo "   → All versions pass: installation, tests, code quality"
    echo "   → Verified versions: ${fully_verified_versions[*]}"
elif [ $fully_verified_count -ge 2 ]; then
    echo -e "\n${YELLOW}⚖️  PARTIAL SUCCESS: $fully_verified_count verified, $failed_count with issues${NC}"
    echo "   → Can release with verified versions: ${fully_verified_versions[*]}"
    echo "   → Document known issues with other versions"
    echo "   → Consider fixing compatibility or updating requirements"
else
    echo -e "\n${RED}⚠️  INSUFFICIENT VERIFICATION: Only $fully_verified_count version(s) verified${NC}"
    echo "   → Need at least 2 fully verified versions for release"
    echo "   → Fix compatibility issues or verify more versions"
fi

echo -e "\n${YELLOW}📝 NEXT STEPS${NC}"
echo "============="

if [ $fully_verified_count -ge 2 ] && [ $failed_count -eq 0 ]; then
    echo "✅ READY TO RELEASE:"
    echo "   1. Update README.md with verified Python versions"
    echo "   2. Update pyproject.toml requires-python based on results"
    echo "   3. Document verified versions: ${fully_verified_versions[*]}"
    echo "   4. Safe to tag and release (alpha.1)"
    exit_code=0
else
    echo "🔧 WORK NEEDED:"
    echo "   1. Review detailed logs: test_results_*.log, mypy_*.log"
    echo "   2. Fix compatibility issues for failed versions"
    echo "   3. Re-run this script until all targeted versions pass"
    echo "   4. Update documentation to reflect actual compatibility"
    echo "   5. Consider reducing version scope if fixes are complex"
    exit_code=1
fi

echo ""
echo -e "${YELLOW}📁 Generated Files:${NC}"
echo "   - test_results_<version>.log: Detailed pytest results"  
echo "   - mypy_<version>.log: Type checking results"
echo "   - Use these logs to debug specific compatibility issues"

exit $exit_code
