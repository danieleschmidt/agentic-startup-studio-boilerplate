#!/usr/bin/env python3
"""
Test suite for CHANGELOG.md validation.
Ensures changelog follows Keep a Changelog format.
"""

from pathlib import Path
import re

def test_changelog_exists():
    """Test that CHANGELOG.md exists."""
    changelog_path = Path("CHANGELOG.md")
    assert changelog_path.exists(), "CHANGELOG.md must exist"
    print("✓ CHANGELOG.md exists")

def test_changelog_format():
    """Test that CHANGELOG follows Keep a Changelog format."""
    with open("CHANGELOG.md", "r") as f:
        content = f.read()
    
    # Should have proper title
    assert "# Changelog" in content or "# CHANGELOG" in content, "Must have proper changelog title"
    
    # Should reference Keep a Changelog
    assert "keepachangelog.com" in content.lower(), "Should reference Keep a Changelog format"
    
    # Should use semantic versioning
    assert "semver.org" in content.lower(), "Should reference Semantic Versioning"
    
    print("✓ Changelog follows proper format")

def test_version_entries():
    """Test that changelog has version entries."""
    with open("CHANGELOG.md", "r") as f:
        content = f.read()
    
    # Should have version sections
    version_pattern = r'## \[?(\d+\.\d+\.\d+|\w+)\]?'
    versions = re.findall(version_pattern, content)
    
    assert len(versions) > 0, "Should have at least one version entry"
    
    # Should have Unreleased section
    assert "unreleased" in content.lower(), "Should have Unreleased section"
    
    print(f"✓ Found {len(versions)} version entries")

def test_initial_version():
    """Test that initial version v0.1.0 is documented."""
    with open("CHANGELOG.md", "r") as f:
        content = f.read()
    
    # Should have 0.1.0 version
    assert "0.1.0" in content, "Should document initial version 0.1.0"
    
    print("✓ Initial version 0.1.0 is documented")

def run_all_tests():
    """Run all changelog tests."""
    try:
        test_changelog_exists()
        test_changelog_format()
        test_version_entries()
        test_initial_version()
        print("\n🎉 All changelog tests passed!")
        return True
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        return False

if __name__ == "__main__":
    run_all_tests()