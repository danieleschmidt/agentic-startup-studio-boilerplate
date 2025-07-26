#!/usr/bin/env python3
"""
Test suite for cookiecutter.json configuration validation.
This follows TDD principles - tests first, then implementation.
"""

import json
import os
from pathlib import Path

def test_cookiecutter_json_exists():
    """Test that cookiecutter.json file exists in the repository root."""
    cookiecutter_path = Path("cookiecutter.json")
    assert cookiecutter_path.exists(), "cookiecutter.json must exist in repository root"
    print("✓ cookiecutter.json exists")

def test_cookiecutter_json_is_valid_json():
    """Test that cookiecutter.json contains valid JSON."""
    with open("cookiecutter.json", "r") as f:
        try:
            config = json.load(f)
        except json.JSONDecodeError as e:
            raise AssertionError(f"cookiecutter.json contains invalid JSON: {e}")
    
    assert isinstance(config, dict), "cookiecutter.json must contain a JSON object"
    print("✓ cookiecutter.json contains valid JSON")
    return config

def test_cookiecutter_has_required_fields(config):
    """Test that cookiecutter.json contains all required template variables."""
    required_fields = [
        "project_name",
        "project_slug", 
        "author_name",
        "author_email",
        "description",
        "version"
    ]
    
    for field in required_fields:
        assert field in config, f"Required field '{field}' missing from cookiecutter.json"
        assert config[field], f"Required field '{field}' cannot be empty"
    print("✓ All required fields present")

def test_cookiecutter_has_sensible_defaults(config):
    """Test that cookiecutter.json provides sensible default values."""
    # Check project_slug is derived from project_name
    assert "{{" in config.get("project_slug", ""), "project_slug should be templated from project_name"
    
    # Check version follows semver
    version = config.get("version", "")
    assert version.startswith("0."), "Version should start with 0. for initial release"
    
    # Check description is meaningful
    description = config.get("description", "")
    assert len(description) > 10, "Description should be meaningful (>10 chars)"
    print("✓ Sensible defaults provided")

def test_cookiecutter_agentic_specific_fields(config):
    """Test that cookiecutter.json includes agentic startup specific fields."""
    agentic_fields = [
        "use_crewai",
        "use_fastapi", 
        "use_react_shadcn",
        "include_terraform",
        "include_keycloak_auth"
    ]
    
    for field in agentic_fields:
        assert field in config, f"Agentic field '{field}' missing from cookiecutter.json"
    print("✓ Agentic-specific fields present")

def run_all_tests():
    """Run all tests in sequence."""
    try:
        test_cookiecutter_json_exists()
        config = test_cookiecutter_json_is_valid_json()
        test_cookiecutter_has_required_fields(config)
        test_cookiecutter_has_sensible_defaults(config)
        test_cookiecutter_agentic_specific_fields(config)
        print("\n🎉 All tests passed!")
        return True
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        return False

if __name__ == "__main__":
    run_all_tests()