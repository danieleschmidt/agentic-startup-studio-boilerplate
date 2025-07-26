#!/usr/bin/env python3
"""
Test suite for CONTRIBUTING.md and CODE_OF_CONDUCT.md validation.
Ensures community documentation exists and is properly structured.
"""

from pathlib import Path

def test_contributing_exists():
    """Test that CONTRIBUTING.md exists."""
    contributing_path = Path("CONTRIBUTING.md")
    assert contributing_path.exists(), "CONTRIBUTING.md must exist"
    print("✓ CONTRIBUTING.md exists")

def test_code_of_conduct_exists():
    """Test that CODE_OF_CONDUCT.md exists."""
    coc_path = Path("CODE_OF_CONDUCT.md")
    assert coc_path.exists(), "CODE_OF_CONDUCT.md must exist"
    print("✓ CODE_OF_CONDUCT.md exists")

def test_contributing_content():
    """Test that CONTRIBUTING.md has essential content."""
    with open("CONTRIBUTING.md", "r") as f:
        content = f.read().lower()
    
    essential_sections = [
        "contributing",
        "development",
        "pull request",
        "issue"
    ]
    
    for section in essential_sections:
        assert section in content, f"CONTRIBUTING.md should mention '{section}'"
    
    print("✓ CONTRIBUTING.md has essential sections")

def test_code_of_conduct_content():
    """Test that CODE_OF_CONDUCT.md follows standard format."""
    with open("CODE_OF_CONDUCT.md", "r") as f:
        content = f.read().lower()
    
    essential_elements = [
        "code of conduct",
        "behavior",
        "community",
        "enforcement"
    ]
    
    for element in essential_elements:
        assert element in content, f"CODE_OF_CONDUCT.md should mention '{element}'"
    
    print("✓ CODE_OF_CONDUCT.md has essential elements")

def test_readme_links_work():
    """Test that README links to these documents correctly."""
    with open("README.md", "r") as f:
        readme_content = f.read()
    
    # Check that README references exist
    assert "CONTRIBUTING.md" in readme_content, "README should link to CONTRIBUTING.md"
    assert "CODE_OF_CONDUCT.md" in readme_content, "README should link to CODE_OF_CONDUCT.md"
    
    print("✓ README properly links to community documents")

def run_all_tests():
    """Run all community documentation tests."""
    try:
        test_contributing_exists()
        test_code_of_conduct_exists()
        test_contributing_content()
        test_code_of_conduct_content()
        test_readme_links_work()
        print("\n🎉 All community documentation tests passed!")
        return True
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        return False

if __name__ == "__main__":
    run_all_tests()