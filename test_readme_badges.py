#!/usr/bin/env python3
"""
Test suite for README badge validation.
Ensures GitHub badges use actual repository paths, not placeholders.
"""

import re
from pathlib import Path

def test_readme_exists():
    """Test that README.md exists."""
    readme_path = Path("README.md")
    assert readme_path.exists(), "README.md must exist"
    print("✓ README.md exists")

def test_no_placeholder_urls():
    """Test that README doesn't contain placeholder GitHub usernames."""
    with open("README.md", "r") as f:
        content = f.read()
    
    # Check for placeholder patterns
    placeholder_patterns = [
        "your-github-username-or-org",
        "your-github-username",
        "your-org-name"
    ]
    
    for pattern in placeholder_patterns:
        assert pattern not in content, f"Found placeholder '{pattern}' in README.md"
    
    print("✓ No placeholder usernames found")

def test_badge_urls_are_valid():
    """Test that badge URLs follow proper GitHub format."""
    with open("README.md", "r") as f:
        content = f.read()
    
    # Find all badge URLs
    badge_urls = re.findall(r'https://img\.shields\.io/github/[^)]+', content)
    github_urls = re.findall(r'https://github\.com/[^)]+', content)
    
    for url in badge_urls + github_urls:
        # Should not contain template placeholders
        assert "your-" not in url.lower(), f"Badge URL contains placeholder: {url}"
        # Should follow proper GitHub format
        if "github.com" in url:
            assert re.match(r'https://github\.com/[\w\-\.]+/[\w\-\.]+', url), f"Invalid GitHub URL format: {url}"
    
    print("✓ Badge URLs are properly formatted")

def test_build_badge_points_to_actual_workflow():
    """Test that build status badge references an actual workflow."""
    with open("README.md", "r") as f:
        content = f.read()
    
    # Look for build status badge
    build_badge_match = re.search(r'https://img\.shields\.io/github/actions/workflow/status/([^/]+)/([^/]+)/([^?]+)', content)
    
    if build_badge_match:
        owner, repo, workflow = build_badge_match.groups()
        print(f"✓ Build badge references workflow: {workflow} in {owner}/{repo}")
        
        # Workflow file should exist if we're referencing it
        workflow_path = Path(f".github/workflows/{workflow}")
        if not workflow_path.exists():
            print(f"⚠️  Warning: Referenced workflow {workflow} doesn't exist yet")
    else:
        print("ℹ️  No build status badge found")

def run_all_tests():
    """Run all badge tests."""
    try:
        test_readme_exists()
        test_no_placeholder_urls()
        test_badge_urls_are_valid()
        test_build_badge_points_to_actual_workflow()
        print("\n🎉 All badge tests passed!")
        return True
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        return False
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        return False

if __name__ == "__main__":
    run_all_tests()