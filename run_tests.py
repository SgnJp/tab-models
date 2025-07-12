#!/usr/bin/env python3
"""
Simple test runner for tab-models library
"""

import sys
import os
import subprocess
import argparse


def run_tests(test_path=None, verbose=False):
    """Run tests using pytest"""
    cmd = ["python", "-m", "pytest"]
    
    if verbose:
        cmd.append("-v")
    
    if test_path:
        cmd.append(test_path)
    else:
        cmd.append("tests/")
    
    print(f"Running tests with command: {' '.join(cmd)}")
    print("-" * 50)
    
    try:
        result = subprocess.run(cmd, check=True)
        print("-" * 50)
        print("All tests passed! ✅")
        return True
    except subprocess.CalledProcessError as e:
        print("-" * 50)
        print(f"Tests failed with exit code {e.returncode} ❌")
        return False


def main():
    parser = argparse.ArgumentParser(description="Run tests for tab-models library")
    parser.add_argument("--test-file", "-t", help="Run specific test file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Check if we're in the right directory
    if not os.path.exists("tests/"):
        print("Error: tests/ directory not found. Make sure you're in the project root.")
        sys.exit(1)
    
    success = run_tests(args.test_file, args.verbose)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main() 