#!/usr/bin/env python3
"""
Workflow Validation Script

This script validates GitHub Actions workflow files for:
- Syntax correctness
- Security best practices
- Required job dependencies
- Proper secret handling
- SLSA compliance requirements
"""

import os
import yaml
import json
import sys
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ValidationResult:
    file_path: str
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    security_issues: List[str]

class WorkflowValidator:
    def __init__(self, workflows_dir: str = ".github/workflows"):
        self.workflows_dir = Path(workflows_dir)
        self.results: List[ValidationResult] = []
        
        # Security best practices
        self.required_permissions = {
            'contents': ['read', 'write'],
            'actions': ['read'],
            'security-events': ['write'],
            'id-token': ['write']
        }
        
        # Allowed action versions (pinned to specific SHAs)
        self.allowed_actions = {
            'actions/checkout': ['v4', 'b4ffde65f46336ab88eb53be808477a3936bae11'],
            'actions/setup-node': ['v4', 'b39b52d1213e96004bfcb1c61a8a6fa8ab84f3e8'],
            'actions/setup-python': ['v5', '0a5c61591373683505ea898e09a3ea4f39ef2b9c'],
            'docker/setup-buildx-action': ['v3', 'f95db51fddba0c2d1ec667646a06c2ce06100226'],
            'docker/login-action': ['v3', '343f7c4344506bcbf9b4de18042ae17996df046d'],
            'docker/build-push-action': ['v5', '2cdde995de11925a030ce8070c3d77a52ffcf1c0']
        }
        
        # Required jobs for different workflow types
        self.required_jobs = {
            'ci': ['lint', 'test', 'security-scan'],
            'cd': ['build', 'deploy'],
            'security': ['scan', 'report'],
            'release': ['build', 'publish', 'release']
        }

    def validate_all_workflows(self) -> bool:
        """Validate all workflow files in the workflows directory."""
        if not self.workflows_dir.exists():
            print(f"❌ Workflows directory not found: {self.workflows_dir}")
            return False
        
        workflow_files = list(self.workflows_dir.glob("*.yml")) + list(self.workflows_dir.glob("*.yaml"))
        
        if not workflow_files:
            print(f"⚠️  No workflow files found in {self.workflows_dir}")
            return True
        
        print(f"🔍 Validating {len(workflow_files)} workflow files...")
        
        all_valid = True
        for workflow_file in workflow_files:
            result = self.validate_workflow(workflow_file)
            self.results.append(result)
            if not result.is_valid:
                all_valid = False
        
        self.print_summary()
        return all_valid

    def validate_workflow(self, file_path: Path) -> ValidationResult:
        """Validate a single workflow file."""
        errors = []
        warnings = []
        security_issues = []
        
        try:
            # Load and parse YAML
            with open(file_path, 'r') as f:
                content = f.read()
                workflow = yaml.safe_load(content)
            
            if not workflow:
                errors.append("Empty or invalid YAML file")
                return ValidationResult(str(file_path), False, errors, warnings, security_issues)
            
            # Basic structure validation
            self._validate_basic_structure(workflow, errors)
            
            # Security validation
            self._validate_security(workflow, security_issues, warnings)
            
            # Job dependencies validation
            self._validate_job_dependencies(workflow, errors, warnings)
            
            # SLSA compliance validation
            self._validate_slsa_compliance(workflow, warnings, file_path.stem)
            
            # Best practices validation
            self._validate_best_practices(workflow, warnings)
            
        except yaml.YAMLError as e:
            errors.append(f"YAML parsing error: {e}")
        except Exception as e:
            errors.append(f"Unexpected error: {e}")
        
        is_valid = len(errors) == 0 and len(security_issues) == 0
        return ValidationResult(str(file_path), is_valid, errors, warnings, security_issues)

    def _validate_basic_structure(self, workflow: Dict, errors: List[str]) -> None:
        """Validate basic workflow structure."""
        required_fields = ['name', 'on', 'jobs']
        
        for field in required_fields:
            if field not in workflow:
                errors.append(f"Missing required field: {field}")
        
        # Validate 'on' triggers
        if 'on' in workflow:
            on_config = workflow['on']
            if isinstance(on_config, str):
                if on_config not in ['push', 'pull_request', 'schedule', 'workflow_dispatch']:
                    errors.append(f"Invalid trigger: {on_config}")
            elif isinstance(on_config, dict):
                valid_triggers = ['push', 'pull_request', 'schedule', 'workflow_dispatch', 'release']
                for trigger in on_config.keys():
                    if trigger not in valid_triggers:
                        errors.append(f"Invalid trigger: {trigger}")
        
        # Validate jobs
        if 'jobs' in workflow:
            jobs = workflow['jobs']
            if not isinstance(jobs, dict) or len(jobs) == 0:
                errors.append("Jobs section must be a non-empty dictionary")
            else:
                for job_name, job_config in jobs.items():
                    if not isinstance(job_config, dict):
                        errors.append(f"Job '{job_name}' must be a dictionary")
                    elif 'runs-on' not in job_config:
                        errors.append(f"Job '{job_name}' missing 'runs-on' field")

    def _validate_security(self, workflow: Dict, security_issues: List[str], warnings: List[str]) -> None:
        """Validate security aspects of the workflow."""
        
        # Check for hardcoded secrets
        workflow_str = str(workflow)
        secret_patterns = [
            r'password["\s]*[:=]["\s]*[^\s"]+',
            r'token["\s]*[:=]["\s]*[^\s"]+',
            r'key["\s]*[:=]["\s]*[^\s"]+',
            r'secret["\s]*[:=]["\s]*[^\s"]+',
        ]
        
        for pattern in secret_patterns:
            if re.search(pattern, workflow_str, re.IGNORECASE):
                security_issues.append(f"Potential hardcoded secret detected: {pattern}")
        
        # Validate action versions
        self._validate_action_versions(workflow, security_issues)
        
        # Check permissions
        self._validate_permissions(workflow, warnings)
        
        # Check for shell injection vulnerabilities
        self._validate_shell_injection(workflow, security_issues)

    def _validate_action_versions(self, workflow: Dict, security_issues: List[str]) -> None:
        """Validate that actions use pinned versions."""
        if 'jobs' not in workflow:
            return
        
        for job_name, job_config in workflow['jobs'].items():
            if 'steps' not in job_config:
                continue
            
            for step in job_config['steps']:
                if 'uses' not in step:
                    continue
                
                action = step['uses']
                
                # Skip local actions (start with ./)
                if action.startswith('./'):
                    continue
                
                # Extract action name and version
                if '@' not in action:
                    security_issues.append(f"Action '{action}' in job '{job_name}' missing version")
                    continue
                
                action_name, version = action.rsplit('@', 1)
                
                # Check if action is in allowed list
                if action_name in self.allowed_actions:
                    allowed_versions = self.allowed_actions[action_name]
                    if version not in allowed_versions:
                        security_issues.append(
                            f"Action '{action_name}' version '{version}' not in allowed list: {allowed_versions}"
                        )
                else:
                    # For unknown actions, check if version looks like a SHA (security best practice)
                    if not re.match(r'^[a-f0-9]{40}$', version) and not re.match(r'^v\d+(\.\d+)*$', version):
                        security_issues.append(
                            f"Action '{action_name}' should use pinned SHA or semantic version, got: {version}"
                        )

    def _validate_permissions(self, workflow: Dict, warnings: List[str]) -> None:
        """Validate workflow permissions."""
        if 'permissions' in workflow:
            permissions = workflow['permissions']
            if isinstance(permissions, str) and permissions == 'read-all':
                warnings.append("Using 'read-all' permissions - consider using minimal permissions")
            elif isinstance(permissions, dict):
                for perm, level in permissions.items():
                    if perm in self.required_permissions:
                        if level not in self.required_permissions[perm]:
                            warnings.append(f"Permission '{perm}' has level '{level}', consider: {self.required_permissions[perm]}")
        
        # Check job-level permissions
        if 'jobs' in workflow:
            for job_name, job_config in workflow['jobs'].items():
                if 'permissions' in job_config:
                    if job_config['permissions'] == 'read-all':
                        warnings.append(f"Job '{job_name}' uses 'read-all' permissions")

    def _validate_shell_injection(self, workflow: Dict, security_issues: List[str]) -> None:
        """Check for potential shell injection vulnerabilities."""
        dangerous_patterns = [
            r'\$\{\{.*github\.event\..*\}\}',  # Using github.event data directly
            r'\$\{\{.*github\.head_ref.*\}\}',  # Using head_ref directly
            r'\$\{\{.*steps\..*\.outputs\..*\}\}.*\|',  # Piping step outputs
        ]
        
        workflow_str = str(workflow)
        for pattern in dangerous_patterns:
            if re.search(pattern, workflow_str):
                security_issues.append(f"Potential shell injection vulnerability: {pattern}")

    def _validate_job_dependencies(self, workflow: Dict, errors: List[str], warnings: List[str]) -> None:
        """Validate job dependencies and sequencing."""
        if 'jobs' not in workflow:
            return
        
        jobs = workflow['jobs']
        job_names = set(jobs.keys())
        
        # Check for circular dependencies
        for job_name, job_config in jobs.items():
            if 'needs' in job_config:
                needs = job_config['needs']
                if isinstance(needs, str):
                    needs = [needs]
                
                for needed_job in needs:
                    if needed_job not in job_names:
                        errors.append(f"Job '{job_name}' depends on non-existent job '{needed_job}'")
                    elif needed_job == job_name:
                        errors.append(f"Job '{job_name}' has circular dependency on itself")

    def _validate_slsa_compliance(self, workflow: Dict, warnings: List[str], workflow_name: str) -> None:
        """Validate SLSA compliance requirements."""
        
        # Check for provenance generation in release workflows
        if 'release' in workflow_name or 'cd' in workflow_name:
            has_provenance = False
            has_sbom = False
            has_signing = False
            
            if 'jobs' in workflow:
                for job_config in workflow['jobs'].values():
                    if 'steps' in job_config:
                        for step in job_config['steps']:
                            if 'uses' in step:
                                action = step['uses']
                                if 'slsa-framework' in action:
                                    has_provenance = True
                                elif 'anchore/sbom-action' in action:
                                    has_sbom = True
                                elif 'cosign' in str(step):
                                    has_signing = True
            
            if not has_provenance:
                warnings.append("Release workflow should include SLSA provenance generation")
            if not has_sbom:
                warnings.append("Release workflow should include SBOM generation")
            if not has_signing:
                warnings.append("Release workflow should include artifact signing")

    def _validate_best_practices(self, workflow: Dict, warnings: List[str]) -> None:
        """Validate workflow best practices."""
        
        # Check for timeout settings
        if 'jobs' in workflow:
            for job_name, job_config in workflow['jobs'].items():
                if 'timeout-minutes' not in job_config:
                    warnings.append(f"Job '{job_name}' missing timeout-minutes setting")
                
                # Check for proper error handling in steps
                if 'steps' in job_config:
                    for step in job_config['steps']:
                        if 'run' in step and 'continue-on-error' not in step:
                            # Check if it's a critical step that should handle errors
                            if any(keyword in str(step).lower() for keyword in ['deploy', 'publish', 'release']):
                                warnings.append(f"Critical step in job '{job_name}' should consider error handling")
        
        # Check for environment specification in deployment jobs
        if 'jobs' in workflow:
            for job_name, job_config in workflow['jobs'].items():
                if any(keyword in job_name.lower() for keyword in ['deploy', 'prod', 'staging']):
                    if 'environment' not in job_config:
                        warnings.append(f"Deployment job '{job_name}' should specify environment")

    def print_summary(self) -> None:
        """Print validation summary."""
        total_files = len(self.results)
        valid_files = sum(1 for r in self.results if r.is_valid)
        
        print(f"\n📊 Validation Summary:")
        print(f"   Total files: {total_files}")
        print(f"   Valid files: {valid_files}")
        print(f"   Invalid files: {total_files - valid_files}")
        
        if total_files == valid_files:
            print(f"✅ All workflow files are valid!")
        else:
            print(f"❌ {total_files - valid_files} workflow files have issues")
        
        # Print detailed results
        for result in self.results:
            print(f"\n📄 {result.file_path}")
            
            if result.is_valid:
                print("   ✅ Valid")
            else:
                print("   ❌ Invalid")
            
            if result.errors:
                print("   🚨 Errors:")
                for error in result.errors:
                    print(f"      - {error}")
            
            if result.security_issues:
                print("   🔒 Security Issues:")
                for issue in result.security_issues:
                    print(f"      - {issue}")
            
            if result.warnings:
                print("   ⚠️  Warnings:")
                for warning in result.warnings:
                    print(f"      - {warning}")

    def generate_report(self, output_file: str = "workflow-validation-report.json") -> None:
        """Generate a JSON report of validation results."""
        report = {
            "timestamp": datetime.utcnow().isoformat(),
            "summary": {
                "total_files": len(self.results),
                "valid_files": sum(1 for r in self.results if r.is_valid),
                "invalid_files": sum(1 for r in self.results if not r.is_valid),
                "total_errors": sum(len(r.errors) for r in self.results),
                "total_warnings": sum(len(r.warnings) for r in self.results),
                "total_security_issues": sum(len(r.security_issues) for r in self.results)
            },
            "results": [
                {
                    "file_path": r.file_path,
                    "is_valid": r.is_valid,
                    "errors": r.errors,
                    "warnings": r.warnings,
                    "security_issues": r.security_issues
                }
                for r in self.results
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"📋 Validation report saved to: {output_file}")

def main():
    """Main function to run workflow validation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate GitHub Actions workflow files")
    parser.add_argument(
        "--workflows-dir",
        default=".github/workflows",
        help="Directory containing workflow files (default: .github/workflows)"
    )
    parser.add_argument(
        "--report",
        help="Generate JSON report file"
    )
    parser.add_argument(
        "--fail-on-warnings",
        action="store_true",
        help="Exit with error code if warnings are found"
    )
    
    args = parser.parse_args()
    
    validator = WorkflowValidator(args.workflows_dir)
    is_valid = validator.validate_all_workflows()
    
    if args.report:
        validator.generate_report(args.report)
    
    # Determine exit code
    has_warnings = any(r.warnings for r in validator.results)
    
    if not is_valid:
        sys.exit(1)
    elif args.fail_on_warnings and has_warnings:
        sys.exit(1)
    else:
        sys.exit(0)

if __name__ == "__main__":
    main()