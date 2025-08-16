#!/usr/bin/env python3
"""
Security Audit for Quantum Task Planner Quality Gates
"""

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import List, Dict, Any
import json

class SecurityAuditor:
    """Comprehensive security auditor for the quantum task planner"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.security_issues = []
        self.security_score = 0
        self.max_score = 0
    
    def run_full_audit(self) -> Dict[str, Any]:
        """Run comprehensive security audit"""
        print("🔒 Starting Security Audit for Quantum Task Planner")
        print("=" * 60)
        
        # 1. Input validation audit
        self._audit_input_validation()
        
        # 2. Code injection prevention
        self._audit_code_injection()
        
        # 3. Sensitive data handling
        self._audit_sensitive_data()
        
        # 4. Authentication and authorization
        self._audit_auth_mechanisms()
        
        # 5. Error handling security
        self._audit_error_handling()
        
        # 6. Dependency security
        self._audit_dependencies()
        
        # 7. File system security
        self._audit_file_operations()
        
        # Generate final report
        return self._generate_security_report()
    
    def _audit_input_validation(self):
        """Audit input validation mechanisms"""
        print("🔍 Auditing Input Validation...")
        self.max_score += 20
        
        # Check for validation functions
        validation_files = list(self.project_root.glob("**/validation.py")) + \
                          list(self.project_root.glob("**/simple_validation.py"))
        
        if validation_files:
            self.security_score += 10
            print("  ✅ Input validation module found")
            
            # Check for specific validation patterns
            for file_path in validation_files:
                content = file_path.read_text()
                
                if "sanitize" in content.lower():
                    self.security_score += 3
                    print("  ✅ Input sanitization implemented")
                
                if "validate" in content.lower():
                    self.security_score += 3
                    print("  ✅ Input validation functions found")
                
                if re.search(r'<script.*?>', content, re.IGNORECASE):
                    self.security_issues.append("Found script tag in validation code - potential XSS risk")
                
                if "eval(" in content or "exec(" in content:
                    self.security_issues.append("Found eval/exec in validation code - code injection risk")
                else:
                    self.security_score += 4
                    print("  ✅ No dangerous eval/exec found in validation")
        else:
            self.security_issues.append("No input validation module found")
    
    def _audit_code_injection(self):
        """Audit for code injection vulnerabilities"""
        print("🔍 Auditing Code Injection Prevention...")
        self.max_score += 15
        
        dangerous_patterns = [
            (r'eval\s*\(', "eval() usage found"),
            (r'exec\s*\(', "exec() usage found"),
            (r'__import__\s*\(', "dynamic import found"),
            (r'subprocess\.call\s*\(', "subprocess.call usage found"),
            (r'os\.system\s*\(', "os.system usage found"),
        ]
        
        python_files = list(self.project_root.glob("**/*.py"))
        injection_risks = 0
        
        for file_path in python_files:
            try:
                content = file_path.read_text()
                for pattern, description in dangerous_patterns:
                    if re.search(pattern, content):
                        self.security_issues.append(f"{description} in {file_path}")
                        injection_risks += 1
            except Exception:
                continue
        
        if injection_risks == 0:
            self.security_score += 15
            print("  ✅ No code injection vulnerabilities found")
        else:
            print(f"  ⚠️ Found {injection_risks} potential code injection risks")
    
    def _audit_sensitive_data(self):
        """Audit sensitive data handling"""
        print("🔍 Auditing Sensitive Data Handling...")
        self.max_score += 15
        
        sensitive_patterns = [
            (r'password\s*=\s*["\'].*["\']', "Hardcoded password found"),
            (r'api_key\s*=\s*["\'].*["\']', "Hardcoded API key found"),
            (r'secret\s*=\s*["\'].*["\']', "Hardcoded secret found"),
            (r'token\s*=\s*["\'].*["\']', "Hardcoded token found"),
        ]
        
        python_files = list(self.project_root.glob("**/*.py"))
        sensitive_data_issues = 0
        
        for file_path in python_files:
            try:
                content = file_path.read_text()
                for pattern, description in sensitive_patterns:
                    if re.search(pattern, content, re.IGNORECASE):
                        self.security_issues.append(f"{description} in {file_path}")
                        sensitive_data_issues += 1
            except Exception:
                continue
        
        # Check for proper environment variable usage
        env_usage = 0
        for file_path in python_files:
            try:
                content = file_path.read_text()
                if "os.environ" in content or "getenv" in content:
                    env_usage += 1
            except Exception:
                continue
        
        if sensitive_data_issues == 0:
            self.security_score += 10
            print("  ✅ No hardcoded sensitive data found")
        
        if env_usage > 0:
            self.security_score += 5
            print("  ✅ Environment variable usage found")
    
    def _audit_auth_mechanisms(self):
        """Audit authentication and authorization"""
        print("🔍 Auditing Authentication Mechanisms...")
        self.max_score += 10
        
        # Check for authentication-related files
        auth_patterns = ["auth", "login", "security", "session"]
        auth_files = []
        
        for pattern in auth_patterns:
            auth_files.extend(list(self.project_root.glob(f"**/*{pattern}*.py")))
        
        if auth_files:
            self.security_score += 5
            print("  ✅ Authentication modules found")
            
            # Check for secure practices
            for file_path in auth_files:
                try:
                    content = file_path.read_text()
                    if "bcrypt" in content or "hashlib" in content:
                        self.security_score += 3
                        print("  ✅ Password hashing mechanisms found")
                        break
                except Exception:
                    continue
        
        # Check for session management
        session_patterns = ["session", "cookie", "jwt", "token"]
        for pattern in session_patterns:
            files = list(self.project_root.glob(f"**/*{pattern}*.py"))
            if files:
                self.security_score += 2
                print(f"  ✅ {pattern.upper()} related code found")
                break
    
    def _audit_error_handling(self):
        """Audit error handling security"""
        print("🔍 Auditing Error Handling Security...")
        self.max_score += 15
        
        python_files = list(self.project_root.glob("**/*.py"))
        error_handling_score = 0
        
        for file_path in python_files:
            try:
                content = file_path.read_text()
                
                # Check for proper exception handling
                if "try:" in content and "except" in content:
                    error_handling_score += 1
                
                # Check for logging instead of printing sensitive info
                if "logger." in content:
                    error_handling_score += 1
                
                # Check for generic exception messages
                if "Exception as e" in content and "str(e)" not in content:
                    error_handling_score += 1
                
            except Exception:
                continue
        
        if error_handling_score > 10:
            self.security_score += 15
            print("  ✅ Good error handling practices found")
        elif error_handling_score > 5:
            self.security_score += 10
            print("  ⚠️ Moderate error handling found")
        else:
            self.security_score += 5
            print("  ⚠️ Limited error handling found")
    
    def _audit_dependencies(self):
        """Audit dependency security"""
        print("🔍 Auditing Dependencies...")
        self.max_score += 15
        
        # Check for requirements files
        req_files = list(self.project_root.glob("**/requirements*.txt"))
        
        if req_files:
            self.security_score += 5
            print("  ✅ Requirements files found")
            
            # Check for pinned versions
            pinned_versions = 0
            total_deps = 0
            
            for req_file in req_files:
                try:
                    content = req_file.read_text()
                    lines = [line.strip() for line in content.split('\n') if line.strip() and not line.startswith('#')]
                    
                    for line in lines:
                        if '==' in line:
                            pinned_versions += 1
                        total_deps += 1
                
                except Exception:
                    continue
            
            if total_deps > 0:
                pin_ratio = pinned_versions / total_deps
                if pin_ratio > 0.8:
                    self.security_score += 10
                    print("  ✅ Most dependencies are pinned")
                elif pin_ratio > 0.5:
                    self.security_score += 7
                    print("  ⚠️ Some dependencies are pinned")
                else:
                    self.security_score += 3
                    print("  ⚠️ Few dependencies are pinned")
        else:
            self.security_issues.append("No requirements.txt found")
    
    def _audit_file_operations(self):
        """Audit file operation security"""
        print("🔍 Auditing File Operations...")
        self.max_score += 10
        
        python_files = list(self.project_root.glob("**/*.py"))
        file_op_issues = 0
        
        dangerous_file_ops = [
            (r'open\s*\(\s*["\'].*\.\./.*["\']', "Path traversal vulnerability"),
            (r'open\s*\(\s*.*input\(.*\)', "User input in file operations"),
            (r'eval\s*\(\s*open\(', "eval() with file content"),
        ]
        
        for file_path in python_files:
            try:
                content = file_path.read_text()
                for pattern, description in dangerous_file_ops:
                    if re.search(pattern, content):
                        self.security_issues.append(f"{description} in {file_path}")
                        file_op_issues += 1
            except Exception:
                continue
        
        if file_op_issues == 0:
            self.security_score += 10
            print("  ✅ No dangerous file operations found")
        else:
            print(f"  ⚠️ Found {file_op_issues} file operation security issues")
    
    def _generate_security_report(self) -> Dict[str, Any]:
        """Generate comprehensive security report"""
        print("\\n" + "=" * 60)
        print("🔒 SECURITY AUDIT REPORT")
        print("=" * 60)
        
        # Calculate security score percentage
        score_percentage = (self.security_score / self.max_score * 100) if self.max_score > 0 else 0
        
        print(f"Security Score: {self.security_score}/{self.max_score} ({score_percentage:.1f}%)")
        
        # Determine security level
        if score_percentage >= 90:
            security_level = "EXCELLENT"
            status_emoji = "🛡️"
        elif score_percentage >= 75:
            security_level = "GOOD"
            status_emoji = "✅"
        elif score_percentage >= 60:
            security_level = "MODERATE"
            status_emoji = "⚠️"
        else:
            security_level = "NEEDS IMPROVEMENT"
            status_emoji = "❌"
        
        print(f"Security Level: {status_emoji} {security_level}")
        
        # List security issues
        if self.security_issues:
            print(f"\\n🚨 Security Issues Found ({len(self.security_issues)}):")
            for i, issue in enumerate(self.security_issues, 1):
                print(f"  {i}. {issue}")
        else:
            print("\\n✅ No security issues found!")
        
        # Security recommendations
        print("\\n📋 Security Recommendations:")
        if score_percentage < 90:
            recommendations = [
                "Implement comprehensive input validation",
                "Add rate limiting to API endpoints", 
                "Enable security headers",
                "Implement proper logging and monitoring",
                "Regular security dependency updates",
                "Add security testing to CI/CD pipeline"
            ]
            
            for rec in recommendations:
                print(f"  • {rec}")
        else:
            print("  ✅ Security implementation is excellent!")
        
        # Quality gate assessment
        print("\\n" + "=" * 60)
        if score_percentage >= 75:
            print("✅ SECURITY QUALITY GATE: PASSED")
            print("System meets security requirements for production.")
        else:
            print("❌ SECURITY QUALITY GATE: FAILED")
            print("System requires security improvements before production.")
        
        return {
            "security_score": self.security_score,
            "max_score": self.max_score,
            "score_percentage": score_percentage,
            "security_level": security_level,
            "issues": self.security_issues,
            "passed_quality_gate": score_percentage >= 75
        }


def main():
    """Run security audit"""
    auditor = SecurityAuditor()
    report = auditor.run_full_audit()
    
    # Save report to file
    with open("security_audit_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\\n📄 Security report saved to: security_audit_report.json")
    
    # Exit with appropriate code
    sys.exit(0 if report["passed_quality_gate"] else 1)


if __name__ == "__main__":
    main()