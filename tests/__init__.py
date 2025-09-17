"""Test initialization file for tests package."""

# This file makes the tests directory a Python package
# and can contain test configuration and fixtures

import os
import sys
from pathlib import Path

# Add src directory to Python path for testing
test_dir = Path(__file__).parent
src_dir = test_dir.parent / "src"
sys.path.insert(0, str(src_dir))

# Test configuration
TEST_DATA_DIR = test_dir / "data"
TEST_OUTPUTS_DIR = test_dir / "outputs"

# Create test directories if they don't exist
TEST_DATA_DIR.mkdir(exist_ok=True)
TEST_OUTPUTS_DIR.mkdir(exist_ok=True)