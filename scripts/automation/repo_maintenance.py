#!/usr/bin/env python3
"""
Repository maintenance automation script
Performs routine maintenance tasks like dependency updates, cleanup, and health checks
"""

import os
import subprocess
import json
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Any
import logging
import requests

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RepositoryMaintenance:
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.repo_name = os.getenv("GITHUB_REPOSITORY", "danieleschmidt/agentic-startup-studio-boilerplate")
        
    def run_all_maintenance_tasks(self) -> Dict[str, Any]:
        """Run all maintenance tasks"""
        logger.info("Starting repository maintenance...")
        
        results = {
            "timestamp": datetime.now().isoformat(),
            "tasks": {}
        }
        
        maintenance_tasks = [
            ("dependency_updates", self.check_dependency_updates),
            ("security_scan", self.run_security_scan),
            ("cleanup", self.cleanup_repository),
            ("health_check", self.repository_health_check),
            ("documentation_check", self.check_documentation_health),
            ("backup_verification", self.verify_backups),
            ("performance_check", self.check_performance_metrics)
        ]
        
        for task_name, task_func in maintenance_tasks:
            logger.info(f"Running {task_name}...")
            try:
                results["tasks"][task_name] = task_func()
                logger.info(f"✅ {task_name} completed successfully")
            except Exception as e:
                logger.error(f"❌ {task_name} failed: {e}")
                results["tasks"][task_name] = {"status": "failed", "error": str(e)}
        
        # Generate maintenance report
        self.generate_maintenance_report(results)
        
        logger.info("Repository maintenance completed")
        return results
    
    def check_dependency_updates(self) -> Dict[str, Any]:
        """Check for available dependency updates"""
        logger.info("Checking for dependency updates...")
        
        results = {
            "python_updates": self.check_python_dependencies(),
            "nodejs_updates": self.check_nodejs_dependencies(),
            "docker_updates": self.check_docker_base_images()
        }
        
        return results
    
    def check_python_dependencies(self) -> Dict[str, Any]:
        """Check Python dependency updates"""
        requirements_file = self.repo_path / "requirements.txt"
        if not requirements_file.exists():
            return {"status": "no_requirements_file"}
        
        try:
            # Use pip-outdated or similar tool
            result = subprocess.run(
                ["pip", "list", "--outdated", "--format=json"],
                capture_output=True,
                text=True,
                cwd=self.repo_path
            )
            
            if result.returncode == 0:
                outdated_packages = json.loads(result.stdout)
                return {
                    "status": "success",
                    "outdated_count": len(outdated_packages),
                    "packages": outdated_packages[:10]  # Limit to first 10
                }
            else:
                return {"status": "error", "error": result.stderr}
                
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def check_nodejs_dependencies(self) -> Dict[str, Any]:
        """Check Node.js dependency updates"""
        package_json = self.repo_path / "package.json"
        if not package_json.exists():
            return {"status": "no_package_json"}
        
        try:
            result = subprocess.run(
                ["npm", "outdated", "--json"],
                capture_output=True,
                text=True,
                cwd=self.repo_path
            )
            
            # npm outdated returns non-zero when there are outdated packages
            if result.stdout:
                try:
                    outdated_packages = json.loads(result.stdout)
                    return {
                        "status": "success",
                        "outdated_count": len(outdated_packages),
                        "packages": list(outdated_packages.keys())[:10]
                    }
                except json.JSONDecodeError:
                    return {"status": "no_outdated_packages"}
            else:
                return {"status": "no_outdated_packages"}
                
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def check_docker_base_images(self) -> Dict[str, Any]:
        """Check for Docker base image updates"""
        dockerfile = self.repo_path / "Dockerfile"
        if not dockerfile.exists():
            return {"status": "no_dockerfile"}
        
        try:
            with open(dockerfile, 'r') as f:
                content = f.read()
            
            # Extract base images
            import re
            base_images = re.findall(r'FROM\s+([^\s]+)', content)
            
            return {
                "status": "success",
                "base_images": base_images,
                "recommendation": "Check Docker Hub for latest tags"
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def run_security_scan(self) -> Dict[str, Any]:
        """Run comprehensive security scan"""
        logger.info("Running security scan...")
        
        results = {
            "bandit_scan": self.run_bandit_scan(),
            "safety_check": self.run_safety_check(),
            "docker_scan": self.run_docker_security_scan(),
            "secrets_check": self.check_for_secrets()
        }
        
        return results
    
    def run_bandit_scan(self) -> Dict[str, Any]:
        """Run Bandit security scan"""
        try:
            result = subprocess.run(
                ["bandit", "-r", ".", "-f", "json", "-o", "bandit-report.json"],
                capture_output=True,
                text=True,
                cwd=self.repo_path
            )
            
            # Bandit returns non-zero when issues are found
            report_file = self.repo_path / "bandit-report.json"
            if report_file.exists():
                with open(report_file, 'r') as f:
                    report_data = json.load(f)
                
                return {
                    "status": "completed",
                    "issues_found": len(report_data.get("results", [])),
                    "high_severity": len([r for r in report_data.get("results", []) if r.get("issue_severity") == "HIGH"]),
                    "medium_severity": len([r for r in report_data.get("results", []) if r.get("issue_severity") == "MEDIUM"])
                }
            else:
                return {"status": "no_report_generated"}
                
        except FileNotFoundError:
            return {"status": "bandit_not_installed"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def run_safety_check(self) -> Dict[str, Any]:
        """Run Safety dependency vulnerability check"""
        try:
            result = subprocess.run(
                ["safety", "check", "--json", "--output", "safety-report.json"],
                capture_output=True,
                text=True,
                cwd=self.repo_path
            )
            
            report_file = self.repo_path / "safety-report.json"
            if report_file.exists():
                with open(report_file, 'r') as f:
                    report_data = json.load(f)
                
                vulnerabilities = report_data.get("vulnerabilities", [])
                return {
                    "status": "completed",
                    "vulnerabilities_found": len(vulnerabilities),
                    "critical": len([v for v in vulnerabilities if v.get("severity", "").lower() == "critical"]),
                    "high": len([v for v in vulnerabilities if v.get("severity", "").lower() == "high"])
                }
            else:
                return {"status": "no_vulnerabilities_found"}
                
        except FileNotFoundError:
            return {"status": "safety_not_installed"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def run_docker_security_scan(self) -> Dict[str, Any]:
        """Run Docker security scan using Trivy or similar"""
        dockerfile = self.repo_path / "Dockerfile"
        if not dockerfile.exists():
            return {"status": "no_dockerfile"}
        
        try:
            # Try to scan with trivy if available
            result = subprocess.run(
                ["trivy", "fs", "--format", "json", "--output", "trivy-report.json", "."],
                capture_output=True,
                text=True,
                cwd=self.repo_path
            )
            
            report_file = self.repo_path / "trivy-report.json"
            if report_file.exists():
                with open(report_file, 'r') as f:
                    report_data = json.load(f)
                
                return {
                    "status": "completed",
                    "scan_tool": "trivy",
                    "results": "check trivy-report.json for details"
                }
            else:
                return {"status": "scan_completed_no_report"}
                
        except FileNotFoundError:
            return {"status": "trivy_not_installed", "recommendation": "Install Trivy for container scanning"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def check_for_secrets(self) -> Dict[str, Any]:
        """Check for accidentally committed secrets"""
        try:
            # Use detect-secrets if available
            result = subprocess.run(
                ["detect-secrets", "scan", "--all-files", "--force-use-all-plugins"],
                capture_output=True,
                text=True,
                cwd=self.repo_path
            )
            
            if result.returncode == 0:
                try:
                    secrets_data = json.loads(result.stdout)
                    secrets_count = sum(len(files) for files in secrets_data.get("results", {}).values())
                    
                    return {
                        "status": "completed",
                        "potential_secrets_found": secrets_count,
                        "files_with_secrets": len(secrets_data.get("results", {}))
                    }
                except json.JSONDecodeError:
                    return {"status": "completed", "potential_secrets_found": 0}
            else:
                return {"status": "error", "error": result.stderr}
                
        except FileNotFoundError:
            return {"status": "detect_secrets_not_installed"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def cleanup_repository(self) -> Dict[str, Any]:
        """Clean up repository artifacts and temporary files"""
        logger.info("Cleaning up repository...")
        
        cleanup_tasks = {
            "python_cache": self.cleanup_python_cache(),
            "node_modules": self.cleanup_node_cache(),
            "docker_cache": self.cleanup_docker_cache(),
            "temp_files": self.cleanup_temp_files(),
            "old_reports": self.cleanup_old_reports()
        }
        
        return cleanup_tasks
    
    def cleanup_python_cache(self) -> Dict[str, Any]:
        """Clean Python cache files"""
        cleaned_files = 0
        
        # Remove __pycache__ directories
        for pycache_dir in self.repo_path.rglob("__pycache__"):
            try:
                import shutil
                shutil.rmtree(pycache_dir)
                cleaned_files += 1
            except Exception:
                continue
        
        # Remove .pyc files
        for pyc_file in self.repo_path.rglob("*.pyc"):
            try:
                pyc_file.unlink()
                cleaned_files += 1
            except Exception:
                continue
        
        return {"status": "completed", "files_cleaned": cleaned_files}
    
    def cleanup_node_cache(self) -> Dict[str, Any]:
        """Clean Node.js cache and build artifacts"""
        cleaned_items = 0
        
        # Remove node_modules/.cache if it exists
        cache_dir = self.repo_path / "node_modules" / ".cache"
        if cache_dir.exists():
            try:
                import shutil
                shutil.rmtree(cache_dir)
                cleaned_items += 1
            except Exception:
                pass
        
        # Remove build directories
        for build_dir in ["build", "dist", ".next"]:
            build_path = self.repo_path / build_dir
            if build_path.exists() and build_path.is_dir():
                try:
                    import shutil
                    shutil.rmtree(build_path)
                    cleaned_items += 1
                except Exception:
                    continue
        
        return {"status": "completed", "items_cleaned": cleaned_items}
    
    def cleanup_docker_cache(self) -> Dict[str, Any]:
        """Clean Docker cache (if Docker is available)"""
        try:
            # Clean unused Docker images and containers
            result = subprocess.run(
                ["docker", "system", "prune", "-f"],
                capture_output=True,
                text=True
            )
            
            if result.returncode == 0:
                return {"status": "completed", "docker_cleanup": "success"}
            else:
                return {"status": "error", "error": result.stderr}
                
        except FileNotFoundError:
            return {"status": "docker_not_available"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def cleanup_temp_files(self) -> Dict[str, Any]:
        """Clean temporary files"""
        temp_patterns = ["*.tmp", "*.temp", "*.log", "*.pid"]
        cleaned_files = 0
        
        for pattern in temp_patterns:
            for temp_file in self.repo_path.rglob(pattern):
                try:
                    # Only remove files older than 1 day
                    if temp_file.stat().st_mtime < (datetime.now().timestamp() - 86400):
                        temp_file.unlink()
                        cleaned_files += 1
                except Exception:
                    continue
        
        return {"status": "completed", "files_cleaned": cleaned_files}
    
    def cleanup_old_reports(self) -> Dict[str, Any]:
        """Clean up old security and test reports"""
        report_patterns = ["*-report.json", "*.xml", "coverage.xml"]
        cleaned_files = 0
        
        for pattern in report_patterns:
            for report_file in self.repo_path.rglob(pattern):
                try:
                    # Keep reports newer than 7 days
                    if report_file.stat().st_mtime < (datetime.now().timestamp() - 7 * 86400):
                        report_file.unlink()
                        cleaned_files += 1
                except Exception:
                    continue
        
        return {"status": "completed", "files_cleaned": cleaned_files}
    
    def repository_health_check(self) -> Dict[str, Any]:
        """Perform repository health check"""
        logger.info("Performing repository health check...")
        
        health_checks = {
            "git_status": self.check_git_status(),
            "required_files": self.check_required_files(),
            "configuration_files": self.check_configuration_health(),
            "dependencies": self.check_dependency_health(),
            "tests": self.check_test_health()
        }
        
        return health_checks
    
    def check_git_status(self) -> Dict[str, Any]:
        """Check Git repository status"""
        try:
            result = subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                cwd=self.repo_path
            )
            
            if result.returncode == 0:
                modified_files = result.stdout.strip().split('\n') if result.stdout.strip() else []
                return {
                    "status": "healthy",
                    "uncommitted_changes": len(modified_files),
                    "is_clean": len(modified_files) == 0
                }
            else:
                return {"status": "error", "error": result.stderr}
                
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def check_required_files(self) -> Dict[str, Any]:
        """Check for required project files"""
        required_files = [
            "README.md", "LICENSE", "requirements.txt", "package.json",
            "Dockerfile", "docker-compose.yml", ".gitignore", "CONTRIBUTING.md"
        ]
        
        file_status = {}
        for file_name in required_files:
            file_path = self.repo_path / file_name
            file_status[file_name] = file_path.exists()
        
        missing_files = [f for f, exists in file_status.items() if not exists]
        
        return {
            "status": "completed",
            "required_files_present": len(file_status) - len(missing_files),
            "total_required_files": len(required_files),
            "missing_files": missing_files
        }
    
    def check_configuration_health(self) -> Dict[str, Any]:
        """Check configuration file health"""
        config_files = {
            "package.json": self.validate_package_json,
            "docker-compose.yml": self.validate_docker_compose,
            "pyproject.toml": self.validate_pyproject_toml
        }
        
        results = {}
        for config_file, validator in config_files.items():
            config_path = self.repo_path / config_file
            if config_path.exists():
                results[config_file] = validator(config_path)
            else:
                results[config_file] = {"status": "file_not_found"}
        
        return results
    
    def validate_package_json(self, file_path: Path) -> Dict[str, Any]:
        """Validate package.json structure"""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            required_fields = ["name", "version", "description", "scripts"]
            missing_fields = [field for field in required_fields if field not in data]
            
            return {
                "status": "valid" if not missing_fields else "invalid",
                "missing_fields": missing_fields,
                "has_scripts": "scripts" in data,
                "script_count": len(data.get("scripts", {}))
            }
            
        except Exception as e:
            return {"status": "parse_error", "error": str(e)}
    
    def validate_docker_compose(self, file_path: Path) -> Dict[str, Any]:
        """Validate docker-compose.yml structure"""
        try:
            with open(file_path, 'r') as f:
                data = yaml.safe_load(f)
            
            return {
                "status": "valid",
                "version": data.get("version", "not_specified"),
                "services_count": len(data.get("services", {})),
                "networks_defined": "networks" in data,
                "volumes_defined": "volumes" in data
            }
            
        except Exception as e:
            return {"status": "parse_error", "error": str(e)}
    
    def validate_pyproject_toml(self, file_path: Path) -> Dict[str, Any]:
        """Validate pyproject.toml structure"""
        try:
            import toml
            with open(file_path, 'r') as f:
                data = toml.load(f)
            
            return {
                "status": "valid",
                "has_build_system": "build-system" in data,
                "has_project_info": "project" in data,
                "has_tool_config": "tool" in data
            }
            
        except ImportError:
            return {"status": "toml_library_not_available"}
        except Exception as e:
            return {"status": "parse_error", "error": str(e)}
    
    def check_dependency_health(self) -> Dict[str, Any]:
        """Check dependency health and compatibility"""
        return {
            "python_deps": self.check_python_dep_health(),
            "nodejs_deps": self.check_nodejs_dep_health()
        }
    
    def check_python_dep_health(self) -> Dict[str, Any]:
        """Check Python dependency health"""
        requirements_file = self.repo_path / "requirements.txt"
        if not requirements_file.exists():
            return {"status": "no_requirements_file"}
        
        try:
            with open(requirements_file, 'r') as f:
                requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            
            return {
                "status": "healthy",
                "total_dependencies": len(requirements),
                "pinned_versions": len([req for req in requirements if "==" in req])
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def check_nodejs_dep_health(self) -> Dict[str, Any]:
        """Check Node.js dependency health"""
        package_json = self.repo_path / "package.json"
        if not package_json.exists():
            return {"status": "no_package_json"}
        
        try:
            with open(package_json, 'r') as f:
                data = json.load(f)
            
            deps = data.get("dependencies", {})
            dev_deps = data.get("devDependencies", {})
            
            return {
                "status": "healthy",
                "production_dependencies": len(deps),
                "dev_dependencies": len(dev_deps),
                "total_dependencies": len(deps) + len(dev_deps)
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def check_test_health(self) -> Dict[str, Any]:
        """Check test suite health"""
        test_dir = self.repo_path / "tests"
        if not test_dir.exists():
            return {"status": "no_test_directory"}
        
        test_files = list(test_dir.rglob("test_*.py")) + list(test_dir.rglob("*_test.py"))
        
        return {
            "status": "healthy",
            "test_files_count": len(test_files),
            "test_directories": len([d for d in test_dir.iterdir() if d.is_dir()])
        }
    
    def check_documentation_health(self) -> Dict[str, Any]:
        """Check documentation health and completeness"""
        logger.info("Checking documentation health...")
        
        docs_checks = {
            "readme_health": self.check_readme_health,
            "docs_directory": self.check_docs_directory,
            "api_documentation": self.check_api_documentation,
            "code_comments": self.check_code_comments
        }
        
        results = {}
        for check_name, check_func in docs_checks.items():
            try:
                results[check_name] = check_func()
            except Exception as e:
                results[check_name] = {"status": "error", "error": str(e)}
        
        return results
    
    def check_readme_health(self) -> Dict[str, Any]:
        """Check README.md health"""
        readme_file = self.repo_path / "README.md"
        if not readme_file.exists():
            return {"status": "missing"}
        
        try:
            with open(readme_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for essential sections
            essential_sections = [
                ("installation", ["install", "setup", "getting started"]),
                ("usage", ["usage", "example", "how to"]),
                ("contributing", ["contribut", "development"]),
                ("license", ["license"])
            ]
            
            sections_found = {}
            for section, keywords in essential_sections:
                sections_found[section] = any(keyword.lower() in content.lower() for keyword in keywords)
            
            return {
                "status": "healthy",
                "length": len(content),
                "sections_found": sections_found,
                "has_badges": "[![" in content,
                "has_code_blocks": "```" in content
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def check_docs_directory(self) -> Dict[str, Any]:
        """Check docs directory structure"""
        docs_dir = self.repo_path / "docs"
        if not docs_dir.exists():
            return {"status": "no_docs_directory"}
        
        try:
            doc_files = list(docs_dir.rglob("*.md"))
            return {
                "status": "healthy",
                "total_doc_files": len(doc_files),
                "has_guides": (docs_dir / "guides").exists(),
                "has_api_docs": any("api" in f.name.lower() for f in doc_files)
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def check_api_documentation(self) -> Dict[str, Any]:
        """Check API documentation completeness"""
        # This is a basic check - could be enhanced with OpenAPI validation
        openapi_files = list(self.repo_path.rglob("openapi.json")) + list(self.repo_path.rglob("swagger.json"))
        
        return {
            "status": "checked",
            "openapi_files_found": len(openapi_files),
            "has_openapi_spec": len(openapi_files) > 0
        }
    
    def check_code_comments(self) -> Dict[str, Any]:
        """Check code comment coverage (basic)"""
        python_files = list(self.repo_path.rglob("*.py"))
        if not python_files:
            return {"status": "no_python_files"}
        
        try:
            total_lines = 0
            comment_lines = 0
            
            for py_file in python_files[:10]:  # Sample first 10 files
                if any(part.startswith('.') for part in py_file.parts):
                    continue
                    
                try:
                    with open(py_file, 'r', encoding='utf-8') as f:
                        lines = f.readlines()
                        total_lines += len(lines)
                        comment_lines += len([line for line in lines if line.strip().startswith('#')])
                except Exception:
                    continue
            
            comment_ratio = (comment_lines / total_lines * 100) if total_lines > 0 else 0
            
            return {
                "status": "healthy",
                "comment_ratio_percent": round(comment_ratio, 2),
                "files_sampled": min(10, len(python_files))
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def verify_backups(self) -> Dict[str, Any]:
        """Verify backup procedures and data integrity"""
        logger.info("Verifying backup procedures...")
        
        backup_checks = {
            "git_remote": self.check_git_remote_backup(),
            "documentation_backup": self.check_documentation_backup(),
            "configuration_backup": self.check_configuration_backup()
        }
        
        return backup_checks
    
    def check_git_remote_backup(self) -> Dict[str, Any]:
        """Check Git remote backup status"""
        try:
            result = subprocess.run(
                ["git", "remote", "-v"],
                capture_output=True,
                text=True,
                cwd=self.repo_path
            )
            
            if result.returncode == 0:
                remotes = result.stdout.strip().split('\n')
                return {
                    "status": "healthy",
                    "remotes_count": len([r for r in remotes if r.strip()]),
                    "has_origin": any("origin" in remote for remote in remotes)
                }
            else:
                return {"status": "error", "error": result.stderr}
                
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def check_documentation_backup(self) -> Dict[str, Any]:
        """Check if documentation is properly versioned"""
        docs_dir = self.repo_path / "docs"
        if not docs_dir.exists():
            return {"status": "no_docs_directory"}
        
        try:
            # Check if docs are tracked in git
            result = subprocess.run(
                ["git", "ls-files", "docs/"],
                capture_output=True,
                text=True,
                cwd=self.repo_path
            )
            
            tracked_docs = result.stdout.strip().split('\n') if result.stdout.strip() else []
            
            return {
                "status": "healthy",
                "tracked_doc_files": len(tracked_docs),
                "docs_in_git": len(tracked_docs) > 0
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def check_configuration_backup(self) -> Dict[str, Any]:
        """Check if configuration files are properly backed up"""
        config_files = [
            "package.json", "requirements.txt", "docker-compose.yml",
            ".env.example", "Dockerfile", "pyproject.toml"
        ]
        
        backed_up_configs = 0
        total_configs = 0
        
        for config_file in config_files:
            config_path = self.repo_path / config_file
            if config_path.exists():
                total_configs += 1
                try:
                    result = subprocess.run(
                        ["git", "ls-files", config_file],
                        capture_output=True,
                        text=True,
                        cwd=self.repo_path
                    )
                    if result.stdout.strip():
                        backed_up_configs += 1
                except Exception:
                    continue
        
        return {
            "status": "healthy",
            "backed_up_configs": backed_up_configs,
            "total_configs": total_configs,
            "backup_ratio": (backed_up_configs / total_configs * 100) if total_configs > 0 else 0
        }
    
    def check_performance_metrics(self) -> Dict[str, Any]:
        """Check basic performance metrics"""
        logger.info("Checking performance metrics...")
        
        return {
            "repository_size": self.check_repository_size(),
            "build_performance": self.check_build_performance(),
            "test_performance": self.estimate_test_performance()
        }
    
    def check_repository_size(self) -> Dict[str, Any]:
        """Check repository size and growth"""
        try:
            result = subprocess.run(
                ["du", "-sh", ".git"],
                capture_output=True,
                text=True,
                cwd=self.repo_path
            )
            
            if result.returncode == 0:
                git_size = result.stdout.split()[0]
                return {
                    "status": "measured",
                    "git_size": git_size,
                    "recommendation": "Monitor for excessive growth"
                }
            else:
                return {"status": "measurement_failed"}
                
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def check_build_performance(self) -> Dict[str, Any]:
        """Check build performance indicators"""
        dockerfile = self.repo_path / "Dockerfile"
        if not dockerfile.exists():
            return {"status": "no_dockerfile"}
        
        try:
            with open(dockerfile, 'r') as f:
                content = f.read()
            
            # Basic heuristics for build performance
            has_multi_stage = content.count("FROM ") > 1
            has_cache_mounts = "RUN --mount=type=cache" in content
            layers_count = content.count("RUN ")
            
            return {
                "status": "analyzed",
                "multi_stage_build": has_multi_stage,
                "cache_optimization": has_cache_mounts,
                "dockerfile_layers": layers_count,
                "recommendation": "Use multi-stage builds and cache mounts for better performance"
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def estimate_test_performance(self) -> Dict[str, Any]:
        """Estimate test performance based on test count"""
        test_files = list(self.repo_path.rglob("test_*.py")) + list(self.repo_path.rglob("*_test.py"))
        
        if not test_files:
            return {"status": "no_tests_found"}
        
        # Rough estimate based on file count
        estimated_time_minutes = len(test_files) * 0.5  # 30 seconds per test file average
        
        return {
            "status": "estimated",
            "test_files_count": len(test_files),
            "estimated_runtime_minutes": round(estimated_time_minutes, 1),
            "recommendation": "Use parallel testing to improve performance"
        }
    
    def generate_maintenance_report(self, results: Dict[str, Any]) -> None:
        """Generate maintenance report"""
        report_file = self.repo_path / "maintenance-report.md"
        
        report_lines = [
            "# Repository Maintenance Report",
            f"Generated: {results['timestamp']}",
            "",
            "## Summary",
            ""
        ]
        
        # Count successful vs failed tasks
        tasks = results.get("tasks", {})
        successful_tasks = sum(1 for task in tasks.values() if isinstance(task, dict) and task.get("status") != "failed")
        total_tasks = len(tasks)
        
        report_lines.extend([
            f"- Total maintenance tasks: {total_tasks}",
            f"- Successful tasks: {successful_tasks}",
            f"- Failed tasks: {total_tasks - successful_tasks}",
            ""
        ])
        
        # Add task details
        for task_name, task_result in tasks.items():
            report_lines.extend([
                f"## {task_name.replace('_', ' ').title()}",
                ""
            ])
            
            if isinstance(task_result, dict):
                if task_result.get("status") == "failed":
                    report_lines.append(f"❌ **Status**: Failed")
                    report_lines.append(f"**Error**: {task_result.get('error', 'Unknown error')}")
                else:
                    report_lines.append(f"✅ **Status**: Completed")
                    
                    # Add specific details based on task type
                    if task_name == "dependency_updates":
                        self.add_dependency_details(report_lines, task_result)
                    elif task_name == "security_scan":
                        self.add_security_details(report_lines, task_result)
                    elif task_name == "cleanup":
                        self.add_cleanup_details(report_lines, task_result)
            
            report_lines.append("")
        
        # Write report
        with open(report_file, 'w') as f:
            f.write('\n'.join(report_lines))
        
        logger.info(f"Maintenance report generated: {report_file}")
    
    def add_dependency_details(self, report_lines: List[str], result: Dict[str, Any]) -> None:
        """Add dependency update details to report"""
        python_updates = result.get("python_updates", {})
        nodejs_updates = result.get("nodejs_updates", {})
        
        if python_updates.get("outdated_count", 0) > 0:
            report_lines.append(f"- Python packages to update: {python_updates['outdated_count']}")
        
        if nodejs_updates.get("outdated_count", 0) > 0:
            report_lines.append(f"- Node.js packages to update: {nodejs_updates['outdated_count']}")
    
    def add_security_details(self, report_lines: List[str], result: Dict[str, Any]) -> None:
        """Add security scan details to report"""
        bandit_result = result.get("bandit_scan", {})
        safety_result = result.get("safety_check", {})
        
        if bandit_result.get("issues_found", 0) > 0:
            report_lines.append(f"- Bandit security issues: {bandit_result['issues_found']}")
        
        if safety_result.get("vulnerabilities_found", 0) > 0:
            report_lines.append(f"- Safety vulnerabilities: {safety_result['vulnerabilities_found']}")
    
    def add_cleanup_details(self, report_lines: List[str], result: Dict[str, Any]) -> None:
        """Add cleanup details to report"""
        for cleanup_type, cleanup_result in result.items():
            if isinstance(cleanup_result, dict) and cleanup_result.get("files_cleaned", 0) > 0:
                report_lines.append(f"- {cleanup_type}: {cleanup_result['files_cleaned']} items cleaned")

def main():
    """Main function to run repository maintenance"""
    maintenance = RepositoryMaintenance()
    
    try:
        results = maintenance.run_all_maintenance_tasks()
        
        # Print summary
        tasks = results.get("tasks", {})
        successful = sum(1 for task in tasks.values() if isinstance(task, dict) and task.get("status") != "failed")
        total = len(tasks)
        
        print(f"Repository maintenance completed: {successful}/{total} tasks successful")
        
        if successful < total:
            print("Some tasks failed. Check maintenance-report.md for details.")
            exit(1)
        
    except Exception as e:
        logger.error(f"Repository maintenance failed: {e}")
        raise

if __name__ == "__main__":
    main()