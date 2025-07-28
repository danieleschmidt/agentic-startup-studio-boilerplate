#!/usr/bin/env python3
"""
Automated metrics collection script for Agentic Startup Studio Boilerplate
Collects metrics from various sources and updates project-metrics.json
"""

import json
import os
import subprocess
import requests
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MetricsCollector:
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.metrics_file = self.repo_path / ".github" / "project-metrics.json"
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.repo_name = os.getenv("GITHUB_REPOSITORY", "danieleschmidt/agentic-startup-studio-boilerplate")
        
    def collect_all_metrics(self) -> Dict[str, Any]:
        """Collect all available metrics"""
        logger.info("Starting metrics collection...")
        
        metrics = {
            "last_updated": datetime.now(timezone.utc).isoformat(),
            "git_metrics": self.collect_git_metrics(),
            "code_metrics": self.collect_code_metrics(),
            "test_metrics": self.collect_test_metrics(),
            "security_metrics": self.collect_security_metrics(),
            "docker_metrics": self.collect_docker_metrics(),
            "github_metrics": self.collect_github_metrics(),
            "dependencies": self.collect_dependency_metrics()
        }
        
        logger.info("Metrics collection completed")
        return metrics
    
    def collect_git_metrics(self) -> Dict[str, Any]:
        """Collect Git repository metrics"""
        logger.info("Collecting Git metrics...")
        
        try:
            # Get commit count
            commit_count = subprocess.check_output(
                ["git", "rev-list", "--count", "HEAD"],
                cwd=self.repo_path,
                text=True
            ).strip()
            
            # Get contributors count
            contributors = subprocess.check_output(
                ["git", "shortlog", "-sn", "--all"],
                cwd=self.repo_path,
                text=True
            ).strip().split('\n')
            
            # Get recent commit activity (last 30 days)
            recent_commits = subprocess.check_output(
                ["git", "rev-list", "--count", "--since='30 days ago'", "HEAD"],
                cwd=self.repo_path,
                text=True
            ).strip()
            
            # Get branch information
            branches = subprocess.check_output(
                ["git", "branch", "-r"],
                cwd=self.repo_path,
                text=True
            ).strip().split('\n')
            
            return {
                "total_commits": int(commit_count),
                "contributors_count": len(contributors),
                "commits_last_30_days": int(recent_commits),
                "remote_branches": len([b for b in branches if b.strip() and "HEAD" not in b]),
                "last_commit_date": self.get_last_commit_date()
            }
        except subprocess.CalledProcessError as e:
            logger.error(f"Error collecting Git metrics: {e}")
            return {}
    
    def get_last_commit_date(self) -> str:
        """Get the date of the last commit"""
        try:
            return subprocess.check_output(
                ["git", "log", "-1", "--format=%ci"],
                cwd=self.repo_path,
                text=True
            ).strip()
        except subprocess.CalledProcessError:
            return ""
    
    def collect_code_metrics(self) -> Dict[str, Any]:
        """Collect code quality metrics"""
        logger.info("Collecting code metrics...")
        
        metrics = {
            "lines_of_code": self.count_lines_of_code(),
            "file_counts": self.count_files_by_type(),
            "complexity": self.analyze_complexity()
        }
        
        return metrics
    
    def count_lines_of_code(self) -> Dict[str, int]:
        """Count lines of code by language"""
        counts = {"python": 0, "javascript": 0, "typescript": 0, "total": 0}
        
        # Python files
        for py_file in self.repo_path.rglob("*.py"):
            if not any(part.startswith('.') for part in py_file.parts):
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        lines = len([line for line in f if line.strip() and not line.strip().startswith('#')])
                        counts["python"] += lines
                except Exception:
                    continue
        
        # JavaScript files
        for js_file in self.repo_path.rglob("*.js"):
            if not any(part.startswith('.') for part in js_file.parts):
                try:
                    with open(js_file, 'r', encoding='utf-8') as f:
                        lines = len([line for line in f if line.strip() and not line.strip().startswith('//')])
                        counts["javascript"] += lines
                except Exception:
                    continue
        
        # TypeScript files
        for ts_file in self.repo_path.rglob("*.ts"):
            if not any(part.startswith('.') for part in ts_file.parts):
                try:
                    with open(ts_file, 'r', encoding='utf-8') as f:
                        lines = len([line for line in f if line.strip() and not line.strip().startswith('//')])
                        counts["typescript"] += lines
                except Exception:
                    continue
        
        counts["total"] = counts["python"] + counts["javascript"] + counts["typescript"]
        return counts
    
    def count_files_by_type(self) -> Dict[str, int]:
        """Count files by type"""
        counts = {}
        extensions = ['.py', '.js', '.ts', '.tsx', '.jsx', '.json', '.yaml', '.yml', '.md', '.dockerfile']
        
        for ext in extensions:
            count = len(list(self.repo_path.rglob(f"*{ext}")))
            if count > 0:
                counts[ext.lstrip('.')] = count
        
        return counts
    
    def analyze_complexity(self) -> Dict[str, Any]:
        """Analyze code complexity using available tools"""
        complexity = {}
        
        # Try to run radon for Python complexity
        try:
            result = subprocess.check_output(
                ["radon", "cc", ".", "-j"],
                cwd=self.repo_path,
                text=True,
                stderr=subprocess.DEVNULL
            )
            complexity["python_complexity"] = json.loads(result)
        except (subprocess.CalledProcessError, FileNotFoundError, json.JSONDecodeError):
            complexity["python_complexity"] = "not_available"
        
        return complexity
    
    def collect_test_metrics(self) -> Dict[str, Any]:
        """Collect test coverage and results"""
        logger.info("Collecting test metrics...")
        
        metrics = {
            "test_files": self.count_test_files(),
            "coverage": self.get_coverage_info()
        }
        
        return metrics
    
    def count_test_files(self) -> Dict[str, int]:
        """Count test files"""
        test_counts = {
            "unit_tests": len(list(self.repo_path.rglob("test_*.py"))),
            "integration_tests": len(list((self.repo_path / "tests/integration").rglob("*.py"))) if (self.repo_path / "tests/integration").exists() else 0,
            "e2e_tests": len(list((self.repo_path / "tests/e2e").rglob("*.py"))) if (self.repo_path / "tests/e2e").exists() else 0,
            "performance_tests": len(list((self.repo_path / "tests/performance").rglob("*.py"))) if (self.repo_path / "tests/performance").exists() else 0
        }
        
        test_counts["total_tests"] = sum(test_counts.values())
        return test_counts
    
    def get_coverage_info(self) -> Dict[str, Any]:
        """Get test coverage information"""
        coverage_file = self.repo_path / "coverage.xml"
        if coverage_file.exists():
            try:
                # Parse coverage.xml for coverage percentage
                with open(coverage_file, 'r') as f:
                    content = f.read()
                    # Simple regex to extract coverage percentage
                    import re
                    match = re.search(r'line-rate="([0-9.]+)"', content)
                    if match:
                        return {"coverage_percentage": float(match.group(1)) * 100}
            except Exception:
                pass
        
        return {"coverage_percentage": "unknown"}
    
    def collect_security_metrics(self) -> Dict[str, Any]:
        """Collect security metrics"""
        logger.info("Collecting security metrics...")
        
        metrics = {
            "security_files": self.check_security_files(),
            "vulnerabilities": self.check_vulnerability_reports()
        }
        
        return metrics
    
    def check_security_files(self) -> Dict[str, bool]:
        """Check for security-related files"""
        security_files = {
            "security_md": (self.repo_path / "SECURITY.md").exists(),
            "bandit_config": (self.repo_path / ".bandit").exists() or (self.repo_path / "pyproject.toml").exists(),
            "safety_config": (self.repo_path / ".safety-policy.json").exists(),
            "pre_commit_config": (self.repo_path / ".pre-commit-config.yaml").exists()
        }
        
        return security_files
    
    def check_vulnerability_reports(self) -> Dict[str, Any]:
        """Check for vulnerability reports"""
        reports = {}
        
        # Check for bandit report
        bandit_report = self.repo_path / "bandit-report.json"
        if bandit_report.exists():
            try:
                with open(bandit_report, 'r') as f:
                    data = json.load(f)
                    reports["bandit"] = {
                        "issues_count": len(data.get("results", [])),
                        "confidence_high": len([r for r in data.get("results", []) if r.get("issue_confidence") == "HIGH"])
                    }
            except Exception:
                reports["bandit"] = "error_reading_report"
        
        # Check for safety report
        safety_report = self.repo_path / "safety-report.json"
        if safety_report.exists():
            try:
                with open(safety_report, 'r') as f:
                    data = json.load(f)
                    reports["safety"] = {
                        "vulnerabilities_count": len(data.get("vulnerabilities", []))
                    }
            except Exception:
                reports["safety"] = "error_reading_report"
        
        return reports
    
    def collect_docker_metrics(self) -> Dict[str, Any]:
        """Collect Docker-related metrics"""
        logger.info("Collecting Docker metrics...")
        
        metrics = {
            "dockerfile_present": (self.repo_path / "Dockerfile").exists(),
            "docker_compose_present": (self.repo_path / "docker-compose.yml").exists(),
            "dockerignore_present": (self.repo_path / ".dockerignore").exists(),
            "multi_stage_build": self.check_multi_stage_dockerfile()
        }
        
        return metrics
    
    def check_multi_stage_dockerfile(self) -> bool:
        """Check if Dockerfile uses multi-stage builds"""
        dockerfile = self.repo_path / "Dockerfile"
        if dockerfile.exists():
            try:
                with open(dockerfile, 'r') as f:
                    content = f.read()
                    return content.count("FROM ") > 1
            except Exception:
                pass
        return False
    
    def collect_github_metrics(self) -> Dict[str, Any]:
        """Collect GitHub API metrics"""
        logger.info("Collecting GitHub metrics...")
        
        if not self.github_token:
            logger.warning("GITHUB_TOKEN not provided, skipping GitHub API metrics")
            return {"error": "no_github_token"}
        
        headers = {
            "Authorization": f"token {self.github_token}",
            "Accept": "application/vnd.github.v3+json"
        }
        
        try:
            # Repository information
            repo_url = f"https://api.github.com/repos/{self.repo_name}"
            repo_response = requests.get(repo_url, headers=headers, timeout=10)
            
            if repo_response.status_code == 200:
                repo_data = repo_response.json()
                
                # Pull requests
                pr_url = f"https://api.github.com/repos/{self.repo_name}/pulls?state=all&per_page=100"
                pr_response = requests.get(pr_url, headers=headers, timeout=10)
                pr_data = pr_response.json() if pr_response.status_code == 200 else []
                
                # Issues
                issues_url = f"https://api.github.com/repos/{self.repo_name}/issues?state=all&per_page=100"
                issues_response = requests.get(issues_url, headers=headers, timeout=10)
                issues_data = issues_response.json() if issues_response.status_code == 200 else []
                
                # Filter out PRs from issues (GitHub treats PRs as issues)
                actual_issues = [issue for issue in issues_data if not issue.get("pull_request")]
                
                return {
                    "stars": repo_data.get("stargazers_count", 0),
                    "forks": repo_data.get("forks_count", 0),
                    "watchers": repo_data.get("subscribers_count", 0),
                    "open_issues": repo_data.get("open_issues_count", 0),
                    "pull_requests": {
                        "total": len(pr_data),
                        "open": len([pr for pr in pr_data if pr.get("state") == "open"]),
                        "merged": len([pr for pr in pr_data if pr.get("merged_at")])
                    },
                    "issues": {
                        "total": len(actual_issues),
                        "open": len([issue for issue in actual_issues if issue.get("state") == "open"])
                    }
                }
            else:
                logger.error(f"GitHub API error: {repo_response.status_code}")
                return {"error": f"api_error_{repo_response.status_code}"}
                
        except requests.RequestException as e:
            logger.error(f"Error fetching GitHub metrics: {e}")
            return {"error": "request_failed"}
    
    def collect_dependency_metrics(self) -> Dict[str, Any]:
        """Collect dependency metrics"""
        logger.info("Collecting dependency metrics...")
        
        metrics = {}
        
        # Python dependencies
        requirements_file = self.repo_path / "requirements.txt"
        if requirements_file.exists():
            try:
                with open(requirements_file, 'r') as f:
                    deps = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                    metrics["python_dependencies"] = len(deps)
            except Exception:
                metrics["python_dependencies"] = "error_reading"
        
        # Node.js dependencies
        package_json = self.repo_path / "package.json"
        if package_json.exists():
            try:
                with open(package_json, 'r') as f:
                    data = json.load(f)
                    deps = data.get("dependencies", {})
                    dev_deps = data.get("devDependencies", {})
                    metrics["nodejs_dependencies"] = len(deps)
                    metrics["nodejs_dev_dependencies"] = len(dev_deps)
            except Exception:
                metrics["nodejs_dependencies"] = "error_reading"
        
        return metrics
    
    def update_metrics_file(self, new_metrics: Dict[str, Any]) -> None:
        """Update the project metrics file"""
        logger.info("Updating metrics file...")
        
        # Load existing metrics
        existing_metrics = {}
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    existing_metrics = json.load(f)
            except Exception as e:
                logger.warning(f"Error loading existing metrics: {e}")
        
        # Merge new metrics with existing ones
        existing_metrics.update(new_metrics)
        existing_metrics["collection_timestamp"] = datetime.now(timezone.utc).isoformat()
        
        # Write updated metrics
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.metrics_file, 'w') as f:
            json.dump(existing_metrics, f, indent=2, sort_keys=True)
        
        logger.info(f"Metrics updated in {self.metrics_file}")
    
    def generate_summary_report(self, metrics: Dict[str, Any]) -> str:
        """Generate a summary report of collected metrics"""
        report = []
        report.append("# Metrics Collection Summary")
        report.append(f"Generated: {metrics.get('last_updated', 'Unknown')}")
        report.append("")
        
        # Git metrics
        if "git_metrics" in metrics:
            git = metrics["git_metrics"]
            report.append("## Git Metrics")
            report.append(f"- Total commits: {git.get('total_commits', 'N/A')}")
            report.append(f"- Contributors: {git.get('contributors_count', 'N/A')}")
            report.append(f"- Recent commits (30 days): {git.get('commits_last_30_days', 'N/A')}")
            report.append("")
        
        # Code metrics
        if "code_metrics" in metrics:
            code = metrics["code_metrics"]
            if "lines_of_code" in code:
                loc = code["lines_of_code"]
                report.append("## Code Metrics")
                report.append(f"- Total lines of code: {loc.get('total', 'N/A')}")
                report.append(f"  - Python: {loc.get('python', 'N/A')}")
                report.append(f"  - JavaScript: {loc.get('javascript', 'N/A')}")
                report.append(f"  - TypeScript: {loc.get('typescript', 'N/A')}")
                report.append("")
        
        # Test metrics
        if "test_metrics" in metrics:
            test = metrics["test_metrics"]
            if "test_files" in test:
                tf = test["test_files"]
                report.append("## Test Metrics")
                report.append(f"- Total test files: {tf.get('total_tests', 'N/A')}")
                report.append(f"  - Unit tests: {tf.get('unit_tests', 'N/A')}")
                report.append(f"  - Integration tests: {tf.get('integration_tests', 'N/A')}")
                report.append(f"  - E2E tests: {tf.get('e2e_tests', 'N/A')}")
                report.append("")
        
        # GitHub metrics
        if "github_metrics" in metrics and "error" not in metrics["github_metrics"]:
            gh = metrics["github_metrics"]
            report.append("## GitHub Metrics")
            report.append(f"- Stars: {gh.get('stars', 'N/A')}")
            report.append(f"- Forks: {gh.get('forks', 'N/A')}")
            report.append(f"- Open issues: {gh.get('issues', {}).get('open', 'N/A')}")
            report.append(f"- Open PRs: {gh.get('pull_requests', {}).get('open', 'N/A')}")
            report.append("")
        
        return "\n".join(report)

def main():
    """Main function to collect and update metrics"""
    collector = MetricsCollector()
    
    try:
        # Collect all metrics
        metrics = collector.collect_all_metrics()
        
        # Update metrics file
        collector.update_metrics_file(metrics)
        
        # Generate and log summary
        summary = collector.generate_summary_report(metrics)
        logger.info("Metrics collection completed successfully")
        print(summary)
        
    except Exception as e:
        logger.error(f"Error during metrics collection: {e}")
        raise

if __name__ == "__main__":
    main()