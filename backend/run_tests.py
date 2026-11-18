"""
Master test runner for DocuMind backend.
Executes all test suites in tests/ directory and prints a structured summary.
"""

import os
import sys
import unittest

# Ensure backend root is on sys.path
backend_dir = os.path.dirname(os.path.abspath(__file__))
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)


def main():
    loader = unittest.TestLoader()
    tests_dir = os.path.join(backend_dir, "tests")
    suite = loader.discover(start_dir=tests_dir, pattern="test_*.py", top_level_dir=backend_dir)

    print("======================================================================")
    print("                     DocuMind AI Backend Test Suite                   ")
    print("======================================================================")

    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    total = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = total - failures - errors

    print("======================================================================")
    print(f"Results: {passed}/{total} passed | {failures} failures | {errors} errors")
    print("======================================================================")

    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
