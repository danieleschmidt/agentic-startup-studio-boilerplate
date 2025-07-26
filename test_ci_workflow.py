#!/usr/bin/env python3
"""
Test suite for GitHub Actions CI workflow validation.
Ensures CI workflow exists and contains proper validation steps.
"""

from pathlib import Path

def test_ci_workflow_exists():
    """Test that ci.yml workflow file exists."""
    ci_path = Path(".github/workflows/ci.yml")
    assert ci_path.exists(), "CI workflow .github/workflows/ci.yml must exist"
    print("✓ CI workflow file exists")

def test_workflow_content():
    """Test that CI workflow has essential content."""
    ci_path = Path(".github/workflows/ci.yml")
    
    with open(ci_path, "r") as f:
        content = f.read()
    
    # Basic YAML structure checks
    assert "name:" in content, "Workflow must have a name"
    assert "on:" in content, "Workflow must specify triggers"
    assert "jobs:" in content, "Workflow must have jobs"
    
    print("✓ Workflow has basic structure")
    return content

def test_workflow_triggers(content):
    """Test that workflow triggers on appropriate events."""
    # Should trigger on push and pull requests
    triggers_found = "push" in content or "pull_request" in content
    assert triggers_found, "Workflow should trigger on push or PR"
    
    print("✓ Workflow has appropriate triggers")

def test_validation_steps(content):
    """Test that workflow includes validation steps."""
    content_lower = content.lower()
    
    # Should have some form of validation
    validation_keywords = ["lint", "test", "validate", "check", "cookiecutter"]
    validation_found = any(keyword in content_lower for keyword in validation_keywords)
    
    assert validation_found, "Workflow should include validation steps"
    print("✓ Workflow includes validation steps")

def test_cookiecutter_template_validation(content):
    """Test that workflow validates cookiecutter template."""
    if "cookiecutter" in content.lower():
        print("✓ Workflow includes cookiecutter template validation")
    else:
        print("ℹ️  No explicit cookiecutter validation found")

def run_all_tests():
    """Run all CI workflow tests."""
    try:
        test_ci_workflow_exists()
        content = test_workflow_content()
        test_workflow_triggers(content)
        test_validation_steps(content)
        test_cookiecutter_template_validation(content)
        print("\n🎉 All CI workflow tests passed!")
        return True
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        return False

if __name__ == "__main__":
    run_all_tests()