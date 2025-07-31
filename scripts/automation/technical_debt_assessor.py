#!/usr/bin/env python3
"""
Advanced Technical Debt Assessment and Management System

This module provides automated technical debt detection, measurement, and management
for enterprise-grade software projects with advanced analytics and reporting.

Features:
- Automated code quality analysis
- Dependency vulnerability assessment
- Architecture debt detection
- Technical debt scoring and trending
- Cost impact analysis
- Automated remediation suggestions
- Integration with CI/CD pipelines
"""

import ast
import json
import logging
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
import yaml
import pandas as pd
import numpy as np
from radon.complexity import cc_visit
from radon.metrics import mi_visit, h_visit
from radon.raw import analyze


@dataclass
class DebtItem:
    """Technical debt item structure."""
    id: str
    category: str
    subcategory: str
    severity: str  # 'critical', 'high', 'medium', 'low'
    file_path: str
    line_number: Optional[int]
    description: str
    impact: str
    effort_hours: float
    business_cost_annual: float
    remediation_suggestion: str
    detection_method: str
    first_detected: datetime
    last_updated: datetime
    status: str  # 'new', 'acknowledged', 'in_progress', 'resolved', 'accepted'


@dataclass
class DebtTrend:
    """Technical debt trend data."""
    date: datetime
    total_debt_hours: float
    total_debt_cost: float
    debt_score: float
    new_debt: int
    resolved_debt: int
    debt_by_category: Dict[str, float]


@dataclass
class CodeQualityMetrics:
    """Code quality metrics structure."""
    file_path: str
    lines_of_code: int
    cyclomatic_complexity: float
    maintainability_index: float
    halstead_volume: float
    duplicated_lines: int
    test_coverage: float
    type_annotation_coverage: float
    documentation_coverage: float


class TechnicalDebtAssessor:
    """Advanced technical debt assessment and management system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the technical debt assessor."""
        self.config = self._load_config(config_path)
        self.debt_items: List[DebtItem] = []
        self.quality_metrics: List[CodeQualityMetrics] = []
        self.trends: List[DebtTrend] = []
        self.logger = self._setup_logging()
        
        # Load historical data if available
        self._load_historical_data()
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            "project_root": ".",
            "include_patterns": ["**/*.py", "**/*.js", "**/*.ts", "**/*.jsx", "**/*.tsx"],
            "exclude_patterns": [
                "**/node_modules/**", "**/venv/**", "**/.git/**",
                "**/build/**", "**/dist/**", "**/__pycache__/**",
                "**/test/**", "**/tests/**"
            ],
            "complexity_thresholds": {
                "cyclomatic_complexity": {"warning": 8, "critical": 15},
                "maintainability_index": {"warning": 70, "critical": 50},
                "lines_of_code": {"warning": 200, "critical": 500}
            },
            "cost_factors": {
                "developer_hourly_rate": 100,
                "maintenance_multiplier": 1.5,
                "bug_fix_multiplier": 2.0,
                "refactoring_efficiency": 0.7
            },
            "debt_scoring": {
                "complexity_weight": 0.3,
                "maintainability_weight": 0.25,
                "coverage_weight": 0.2,
                "duplication_weight": 0.15,
                "dependency_weight": 0.1
            },
            "report_output": {
                "json_path": "reports/technical_debt_assessment.json",
                "html_path": "reports/technical_debt_report.html",
                "trend_path": "reports/debt_trends.json"
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config)
                
        return default_config
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger("technical_debt_assessor")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _load_historical_data(self):
        """Load historical debt data and trends."""
        try:
            trend_path = Path(self.config["report_output"]["trend_path"])
            if trend_path.exists():
                with open(trend_path) as f:
                    trend_data = json.load(f)
                    self.trends = [
                        DebtTrend(
                            date=datetime.fromisoformat(item["date"]),
                            total_debt_hours=item["total_debt_hours"],
                            total_debt_cost=item["total_debt_cost"],
                            debt_score=item["debt_score"],
                            new_debt=item["new_debt"],
                            resolved_debt=item["resolved_debt"],
                            debt_by_category=item["debt_by_category"]
                        )
                        for item in trend_data
                    ]
        except Exception as e:
            self.logger.warning(f"Could not load historical data: {e}")
    
    def analyze_project(self) -> Dict[str, Any]:
        """Perform comprehensive technical debt analysis."""
        self.logger.info("Starting comprehensive technical debt analysis...")
        
        # Clear previous analysis
        self.debt_items.clear()
        self.quality_metrics.clear()
        
        # Get all relevant files
        files_to_analyze = self._get_files_to_analyze()
        self.logger.info(f"Analyzing {len(files_to_analyze)} files")
        
        # Analyze each file
        for file_path in files_to_analyze:
            try:
                self._analyze_file(file_path)
            except Exception as e:
                self.logger.error(f"Error analyzing {file_path}: {e}")
        
        # Perform cross-file analysis
        self._analyze_architecture_debt()
        self._analyze_dependency_debt()
        self._analyze_test_debt()
        self._analyze_documentation_debt()
        
        # Calculate overall metrics
        analysis_results = self._calculate_debt_metrics()
        
        # Update trends
        self._update_trends(analysis_results)
        
        self.logger.info("Technical debt analysis completed")
        
        return analysis_results
    
    def _get_files_to_analyze(self) -> List[Path]:
        """Get list of files to analyze based on include/exclude patterns."""
        project_root = Path(self.config["project_root"])
        files_to_analyze = []
        
        for pattern in self.config["include_patterns"]:
            files = list(project_root.glob(pattern))
            for file_path in files:
                if file_path.is_file() and not self._should_exclude(file_path):
                    files_to_analyze.append(file_path)
        
        return list(set(files_to_analyze))  # Remove duplicates
    
    def _should_exclude(self, file_path: Path) -> bool:
        """Check if file should be excluded from analysis."""
        file_str = str(file_path)
        for pattern in self.config["exclude_patterns"]:
            if file_path.match(pattern) or pattern in file_str:
                return True
        return False
    
    def _analyze_file(self, file_path: Path):
        """Analyze a single file for technical debt."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except (UnicodeDecodeError, PermissionError):
            return
        
        file_extension = file_path.suffix.lower()
        
        if file_extension == '.py':
            self._analyze_python_file(file_path, content)
        elif file_extension in ['.js', '.ts', '.jsx', '.tsx']:
            self._analyze_javascript_file(file_path, content)
        else:
            self._analyze_generic_file(file_path, content)
    
    def _analyze_python_file(self, file_path: Path, content: str):
        """Analyze Python file for technical debt."""
        try:
            tree = ast.parse(content)
        except SyntaxError as e:
            self._add_debt_item(
                category="syntax",
                subcategory="syntax_error",
                severity="critical",
                file_path=str(file_path),
                line_number=e.lineno,
                description=f"Syntax error: {e.msg}",
                impact="Code cannot be executed",
                effort_hours=1.0,
                remediation_suggestion="Fix syntax error"
            )
            return
        
        # Analyze complexity
        complexity_results = cc_visit(content)
        for result in complexity_results:
            if result.complexity > self.config["complexity_thresholds"]["cyclomatic_complexity"]["critical"]:
                severity = "critical"
            elif result.complexity > self.config["complexity_thresholds"]["cyclomatic_complexity"]["warning"]:
                severity = "high"
            else:
                continue  # Skip if within acceptable range
            
            self._add_debt_item(
                category="complexity",
                subcategory="cyclomatic_complexity",
                severity=severity,
                file_path=str(file_path),
                line_number=result.lineno,
                description=f"High cyclomatic complexity: {result.complexity}",
                impact="Difficult to test and maintain",
                effort_hours=result.complexity * 0.5,
                remediation_suggestion="Break down into smaller functions"
            )
        
        # Analyze maintainability
        mi_results = mi_visit(content, multi=True)
        maintainability_index = mi_results.mi
        
        if maintainability_index < self.config["complexity_thresholds"]["maintainability_index"]["critical"]:
            severity = "critical"
        elif maintainability_index < self.config["complexity_thresholds"]["maintainability_index"]["warning"]:
            severity = "medium"
        else:
            severity = None
        
        if severity:
            self._add_debt_item(
                category="maintainability",
                subcategory="maintainability_index",
                severity=severity,
                file_path=str(file_path),
                line_number=None,
                description=f"Low maintainability index: {maintainability_index:.1f}",
                impact="Difficult to maintain and modify",
                effort_hours=8.0,
                remediation_suggestion="Refactor to improve code structure and reduce complexity"
            )
        
        # Analyze type annotations
        self._analyze_type_annotations(file_path, tree)
        
        # Analyze documentation
        self._analyze_function_documentation(file_path, tree)
        
        # Create quality metrics
        raw_metrics = analyze(content)
        self.quality_metrics.append(CodeQualityMetrics(
            file_path=str(file_path),
            lines_of_code=raw_metrics.loc,
            cyclomatic_complexity=sum(r.complexity for r in complexity_results) / max(len(complexity_results), 1),
            maintainability_index=maintainability_index,
            halstead_volume=h_visit(content).volume,
            duplicated_lines=0,  # Would need additional analysis
            test_coverage=0.0,   # Would need coverage report
            type_annotation_coverage=self._calculate_type_annotation_coverage(tree),
            documentation_coverage=self._calculate_documentation_coverage(tree)
        ))
    
    def _analyze_javascript_file(self, file_path: Path, content: str):
        """Analyze JavaScript/TypeScript file for technical debt."""
        # Basic analysis for JS/TS files
        lines = content.split('\n')
        loc = len([line for line in lines if line.strip() and not line.strip().startswith('//')])
        
        # Check for TODO/FIXME comments
        for i, line in enumerate(lines, 1):
            if re.search(r'(TODO|FIXME|HACK)', line, re.IGNORECASE):
                self._add_debt_item(
                    category="code_quality",
                    subcategory="todo_comment",
                    severity="low",
                    file_path=str(file_path),
                    line_number=i,
                    description=f"TODO/FIXME comment: {line.strip()}",
                    impact="Indicates incomplete or temporary code",
                    effort_hours=2.0,
                    remediation_suggestion="Complete the implementation or remove the comment"
                )
        
        # Check file size
        if loc > self.config["complexity_thresholds"]["lines_of_code"]["critical"]:
            severity = "critical"
        elif loc > self.config["complexity_thresholds"]["lines_of_code"]["warning"]:
            severity = "medium"
        else:
            severity = None
        
        if severity:
            self._add_debt_item(
                category="code_quality",
                subcategory="file_size",
                severity=severity,
                file_path=str(file_path),
                line_number=None,
                description=f"Large file: {loc} lines of code",
                impact="Difficult to navigate and maintain",
                effort_hours=loc * 0.02,
                remediation_suggestion="Split into smaller, more focused modules"
            )
    
    def _analyze_generic_file(self, file_path: Path, content: str):
        """Analyze generic file for basic issues."""
        lines = content.split('\n')
        
        # Check for very large files
        if len(lines) > 1000:
            self._add_debt_item(
                category="code_quality",
                subcategory="file_size",
                severity="medium",
                file_path=str(file_path),
                line_number=None,
                description=f"Very large file: {len(lines)} lines",
                impact="Difficult to navigate and review",
                effort_hours=2.0,
                remediation_suggestion="Consider splitting into smaller files"
            )
    
    def _analyze_type_annotations(self, file_path: Path, tree: ast.AST):
        """Analyze type annotation coverage."""
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        for func in functions:
            # Skip private functions and test functions
            if func.name.startswith('_') or func.name.startswith('test_'):
                continue
            
            # Check for missing return type annotation
            if func.returns is None and func.name not in ['__init__', '__str__', '__repr__']:
                self._add_debt_item(
                    category="type_safety",
                    subcategory="missing_return_type",
                    severity="low",
                    file_path=str(file_path),
                    line_number=func.lineno,
                    description=f"Function '{func.name}' missing return type annotation",
                    impact="Reduced IDE support and type safety",
                    effort_hours=0.25,
                    remediation_suggestion="Add return type annotation"
                )
            
            # Check for missing parameter type annotations
            for arg in func.args.args:
                if arg.annotation is None and arg.arg != 'self':
                    self._add_debt_item(
                        category="type_safety",
                        subcategory="missing_parameter_type",
                        severity="low",
                        file_path=str(file_path),
                        line_number=func.lineno,
                        description=f"Parameter '{arg.arg}' in function '{func.name}' missing type annotation",
                        impact="Reduced IDE support and type safety",
                        effort_hours=0.1,
                        remediation_suggestion="Add parameter type annotation"
                    )
    
    def _analyze_function_documentation(self, file_path: Path, tree: ast.AST):
        """Analyze function documentation coverage."""
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        for func in functions:
            # Skip private functions and very simple functions
            if func.name.startswith('_') or len(func.body) <= 2:
                continue
            
            # Check for missing docstring
            docstring = ast.get_docstring(func)
            if not docstring:
                self._add_debt_item(
                    category="documentation",
                    subcategory="missing_docstring",
                    severity="low",
                    file_path=str(file_path),
                    line_number=func.lineno,
                    description=f"Function '{func.name}' missing docstring",
                    impact="Poor developer experience and maintainability",
                    effort_hours=0.5,
                    remediation_suggestion="Add comprehensive docstring with description, parameters, and return value"
                )
    
    def _analyze_architecture_debt(self):
        """Analyze architecture-level technical debt."""
        # Analyze import dependencies for circular dependencies
        self._detect_circular_dependencies()
        
        # Analyze module coupling
        self._analyze_module_coupling()
        
        # Check for God classes/modules
        self._detect_god_classes()
    
    def _analyze_dependency_debt(self):
        """Analyze dependency-related technical debt."""
        try:
            # Check for outdated dependencies
            self._check_outdated_dependencies()
            
            # Check for security vulnerabilities
            self._check_security_vulnerabilities()
            
            # Check for unused dependencies
            self._check_unused_dependencies()
            
        except Exception as e:
            self.logger.error(f"Error analyzing dependencies: {e}")
    
    def _analyze_test_debt(self):
        """Analyze test-related technical debt."""
        # This would typically require integration with coverage tools
        # For now, we'll do basic analysis
        
        test_files = []
        source_files = []
        
        for metric in self.quality_metrics:
            file_path = Path(metric.file_path)
            if 'test' in str(file_path).lower():
                test_files.append(file_path)
            else:
                source_files.append(file_path)
        
        # Check test coverage ratio
        test_ratio = len(test_files) / max(len(source_files), 1)
        
        if test_ratio < 0.3:  # Less than 30% test files
            self._add_debt_item(
                category="testing",
                subcategory="low_test_coverage",
                severity="high",
                file_path="project_root",
                line_number=None,
                description=f"Low test coverage: {test_ratio:.1%} test files",
                impact="Higher risk of undetected bugs",
                effort_hours=20.0,
                remediation_suggestion="Add comprehensive test suite"
            )
    
    def _analyze_documentation_debt(self):
        """Analyze documentation-related technical debt."""
        # Check for missing README sections
        readme_path = Path(self.config["project_root"]) / "README.md"
        
        if not readme_path.exists():
            self._add_debt_item(
                category="documentation",
                subcategory="missing_readme",
                severity="medium",
                file_path="README.md",
                line_number=None,
                description="Missing README.md file",
                impact="Poor project onboarding experience",
                effort_hours=4.0,
                remediation_suggestion="Create comprehensive README with setup and usage instructions"
            )
    
    def _detect_circular_dependencies(self):
        """Detect circular import dependencies."""
        # This would require more sophisticated dependency analysis
        # For now, we'll add a placeholder
        pass
    
    def _analyze_module_coupling(self):
        """Analyze module coupling and cohesion."""
        # This would require import graph analysis
        # For now, we'll add a placeholder
        pass
    
    def _detect_god_classes(self):
        """Detect God classes (classes with too many responsibilities)."""
        for metric in self.quality_metrics:
            if metric.lines_of_code > 500:  # Arbitrary threshold
                self._add_debt_item(
                    category="design",
                    subcategory="god_class",
                    severity="medium",
                    file_path=metric.file_path,
                    line_number=None,
                    description=f"Potentially oversized class/module: {metric.lines_of_code} lines",
                    impact="Violates single responsibility principle",
                    effort_hours=metric.lines_of_code * 0.05,
                    remediation_suggestion="Split into smaller, more focused classes/modules"
                )
    
    def _check_outdated_dependencies(self):
        """Check for outdated dependencies."""
        try:
            # Check Python dependencies
            requirements_files = [
                Path(self.config["project_root"]) / "requirements.txt",
                Path(self.config["project_root"]) / "pyproject.toml"
            ]
            
            for req_file in requirements_files:
                if req_file.exists():
                    result = subprocess.run(
                        ["pip", "list", "--outdated", "--format=json"],
                        capture_output=True, text=True
                    )
                    
                    if result.returncode == 0:
                        outdated = json.loads(result.stdout)
                        for package in outdated:
                            self._add_debt_item(
                                category="dependencies",
                                subcategory="outdated_dependency",
                                severity="low",
                                file_path=str(req_file),
                                line_number=None,
                                description=f"Outdated dependency: {package['name']} {package['version']} -> {package['latest_version']}",
                                impact="Missing security updates and new features",
                                effort_hours=1.0,
                                remediation_suggestion=f"Update {package['name']} to version {package['latest_version']}"
                            )
                    break
                    
        except Exception as e:
            self.logger.warning(f"Could not check outdated dependencies: {e}")
    
    def _check_security_vulnerabilities(self):
        """Check for security vulnerabilities in dependencies."""
        try:
            result = subprocess.run(
                ["safety", "check", "--json"],
                capture_output=True, text=True
            )
            
            if result.returncode != 0 and result.stdout:
                vulnerabilities = json.loads(result.stdout)
                for vuln in vulnerabilities:
                    self._add_debt_item(
                        category="security",
                        subcategory="vulnerability",
                        severity="critical",
                        file_path="dependencies",
                        line_number=None,
                        description=f"Security vulnerability in {vuln.get('package', 'unknown')}: {vuln.get('advisory', '')}",
                        impact="Security risk to application",
                        effort_hours=4.0,
                        remediation_suggestion=f"Update package to secure version"
                    )
                    
        except Exception as e:
            self.logger.warning(f"Could not check security vulnerabilities: {e}")
    
    def _check_unused_dependencies(self):
        """Check for unused dependencies."""
        # This would require sophisticated import analysis
        # For now, we'll add a placeholder
        pass
    
    def _add_debt_item(self, category: str, subcategory: str, severity: str, 
                      file_path: str, line_number: Optional[int], description: str,
                      impact: str, effort_hours: float, remediation_suggestion: str):
        """Add a technical debt item."""
        debt_id = f"{category}_{subcategory}_{file_path}_{line_number or 0}"
        
        # Check if this debt item already exists
        existing_item = next((item for item in self.debt_items if item.id == debt_id), None)
        
        if existing_item:
            existing_item.last_updated = datetime.now()
            return
        
        business_cost = effort_hours * self.config["cost_factors"]["developer_hourly_rate"]
        business_cost *= self.config["cost_factors"]["maintenance_multiplier"]
        
        debt_item = DebtItem(
            id=debt_id,
            category=category,
            subcategory=subcategory,
            severity=severity,
            file_path=file_path,
            line_number=line_number,
            description=description,
            impact=impact,
            effort_hours=effort_hours,
            business_cost_annual=business_cost,
            remediation_suggestion=remediation_suggestion,
            detection_method="automated_analysis",
            first_detected=datetime.now(),
            last_updated=datetime.now(),
            status="new"
        )
        
        self.debt_items.append(debt_item)
    
    def _calculate_type_annotation_coverage(self, tree: ast.AST) -> float:
        """Calculate type annotation coverage for Python code."""
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        
        if not functions:
            return 100.0
        
        annotated_count = 0
        total_count = 0
        
        for func in functions:
            if func.name.startswith('_'):  # Skip private functions
                continue
                
            total_count += 1
            
            # Check if function has return annotation
            if func.returns is not None:
                annotated_count += 1
            
            # Check parameter annotations
            for arg in func.args.args:
                if arg.arg != 'self':
                    total_count += 1
                    if arg.annotation is not None:
                        annotated_count += 1
        
        return (annotated_count / max(total_count, 1)) * 100
    
    def _calculate_documentation_coverage(self, tree: ast.AST) -> float:
        """Calculate documentation coverage for Python code."""
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        
        total_items = len(functions) + len(classes)
        
        if total_items == 0:
            return 100.0
        
        documented_items = 0
        
        for func in functions:
            if func.name.startswith('_'):  # Skip private functions
                continue
            if ast.get_docstring(func):
                documented_items += 1
        
        for cls in classes:
            if ast.get_docstring(cls):
                documented_items += 1
        
        return (documented_items / total_items) * 100
    
    def _calculate_debt_metrics(self) -> Dict[str, Any]:
        """Calculate comprehensive technical debt metrics."""
        total_debt_hours = sum(item.effort_hours for item in self.debt_items)
        total_debt_cost = sum(item.business_cost_annual for item in self.debt_items)
        
        # Calculate debt by category
        debt_by_category = {}
        for item in self.debt_items:
            if item.category not in debt_by_category:
                debt_by_category[item.category] = {"count": 0, "hours": 0, "cost": 0}
            debt_by_category[item.category]["count"] += 1
            debt_by_category[item.category]["hours"] += item.effort_hours
            debt_by_category[item.category]["cost"] += item.business_cost_annual
        
        # Calculate debt by severity
        debt_by_severity = {}
        for item in self.debt_items:
            if item.severity not in debt_by_severity:
                debt_by_severity[item.severity] = {"count": 0, "hours": 0, "cost": 0}
            debt_by_severity[item.severity]["count"] += 1
            debt_by_severity[item.severity]["hours"] += item.effort_hours
            debt_by_severity[item.severity]["cost"] += item.business_cost_annual
        
        # Calculate overall debt score (0-100, higher is better)
        debt_score = self._calculate_debt_score()
        
        # Calculate quality metrics averages
        if self.quality_metrics:
            avg_complexity = np.mean([m.cyclomatic_complexity for m in self.quality_metrics])
            avg_maintainability = np.mean([m.maintainability_index for m in self.quality_metrics])
            avg_type_coverage = np.mean([m.type_annotation_coverage for m in self.quality_metrics])
            avg_doc_coverage = np.mean([m.documentation_coverage for m in self.quality_metrics])
        else:
            avg_complexity = avg_maintainability = avg_type_coverage = avg_doc_coverage = 0
        
        return {
            "assessment_date": datetime.now().isoformat(),
            "total_files_analyzed": len(self.quality_metrics),
            "total_debt_items": len(self.debt_items),
            "total_debt_hours": total_debt_hours,
            "total_debt_cost_annual": total_debt_cost,
            "debt_score": debt_score,
            "debt_by_category": debt_by_category,
            "debt_by_severity": debt_by_severity,
            "quality_metrics": {
                "average_cyclomatic_complexity": avg_complexity,
                "average_maintainability_index": avg_maintainability,
                "average_type_annotation_coverage": avg_type_coverage,
                "average_documentation_coverage": avg_doc_coverage
            },
            "debt_items": [asdict(item) for item in self.debt_items],
            "recommendations": self._generate_prioritized_recommendations()
        }
    
    def _calculate_debt_score(self) -> float:
        """Calculate overall technical debt score (0-100, higher is better)."""
        if not self.quality_metrics:
            return 50.0  # Neutral score if no data
        
        # Calculate component scores
        complexity_score = 100 - min(np.mean([m.cyclomatic_complexity for m in self.quality_metrics]) * 10, 100)
        maintainability_score = np.mean([m.maintainability_index for m in self.quality_metrics])
        type_coverage_score = np.mean([m.type_annotation_coverage for m in self.quality_metrics])
        doc_coverage_score = np.mean([m.documentation_coverage for m in self.quality_metrics])
        
        # Calculate penalty for critical issues
        critical_penalty = len([item for item in self.debt_items if item.severity == "critical"]) * 10
        high_penalty = len([item for item in self.debt_items if item.severity == "high"]) * 5
        
        # Weighted score calculation
        weights = self.config["debt_scoring"]
        weighted_score = (
            complexity_score * weights["complexity_weight"] +
            maintainability_score * weights["maintainability_weight"] +
            type_coverage_score * weights["coverage_weight"] +
            doc_coverage_score * weights["coverage_weight"]
        )
        
        # Apply penalties
        final_score = max(0, weighted_score - critical_penalty - high_penalty)
        
        return min(100, final_score)
    
    def _generate_prioritized_recommendations(self) -> List[Dict[str, Any]]:
        """Generate prioritized recommendations for debt reduction."""
        recommendations = []
        
        # Group debt items by category and severity
        critical_items = [item for item in self.debt_items if item.severity == "critical"]
        high_items = [item for item in self.debt_items if item.severity == "high"]
        
        # Critical items first
        if critical_items:
            total_critical_hours = sum(item.effort_hours for item in critical_items)
            recommendations.append({
                "priority": "P0",
                "title": "Address Critical Technical Debt",
                "description": f"Resolve {len(critical_items)} critical debt items",
                "estimated_effort_hours": total_critical_hours,
                "estimated_cost": total_critical_hours * self.config["cost_factors"]["developer_hourly_rate"],
                "impact": "High - Prevents production issues and security vulnerabilities",
                "items": [item.id for item in critical_items[:5]]  # Show top 5
            })
        
        # High-impact items
        if high_items:
            total_high_hours = sum(item.effort_hours for item in high_items)
            recommendations.append({
                "priority": "P1",
                "title": "Reduce High-Impact Technical Debt",
                "description": f"Address {len(high_items)} high-impact debt items",
                "estimated_effort_hours": total_high_hours,
                "estimated_cost": total_high_hours * self.config["cost_factors"]["developer_hourly_rate"],
                "impact": "Medium - Improves maintainability and developer productivity",
                "items": [item.id for item in high_items[:5]]  # Show top 5
            })
        
        # Category-specific recommendations
        category_recommendations = {
            "complexity": "Refactor complex functions to improve maintainability",
            "type_safety": "Add type annotations to improve code quality",
            "documentation": "Improve code documentation for better maintainability",
            "testing": "Increase test coverage to reduce bug risk",
            "security": "Address security vulnerabilities immediately"
        }
        
        for category, description in category_recommendations.items():
            category_items = [item for item in self.debt_items if item.category == category]
            if category_items:
                total_hours = sum(item.effort_hours for item in category_items)
                recommendations.append({
                    "priority": "P2",
                    "title": f"Improve {category.title()}",
                    "description": description,
                    "estimated_effort_hours": total_hours,
                    "estimated_cost": total_hours * self.config["cost_factors"]["developer_hourly_rate"],
                    "impact": "Medium - Gradual improvement in code quality",
                    "items": [item.id for item in category_items[:3]]
                })
        
        return recommendations
    
    def _update_trends(self, analysis_results: Dict[str, Any]):
        """Update technical debt trends."""
        current_trend = DebtTrend(
            date=datetime.now(),
            total_debt_hours=analysis_results["total_debt_hours"],
            total_debt_cost=analysis_results["total_debt_cost_annual"],
            debt_score=analysis_results["debt_score"],
            new_debt=len([item for item in self.debt_items if item.status == "new"]),
            resolved_debt=0,  # Would need to track resolved items
            debt_by_category={k: v["hours"] for k, v in analysis_results["debt_by_category"].items()}
        )
        
        self.trends.append(current_trend)
        
        # Keep only last 90 days of trends
        cutoff_date = datetime.now() - timedelta(days=90)
        self.trends = [trend for trend in self.trends if trend.date > cutoff_date]
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive technical debt report."""
        # Perform analysis
        analysis_results = self.analyze_project()
        
        # Create comprehensive report
        report = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "assessor_version": "1.0.0",
                "project_root": self.config["project_root"]
            },
            "executive_summary": self._generate_executive_summary(analysis_results),
            "detailed_analysis": analysis_results,
            "trends": [asdict(trend) for trend in self.trends],
            "action_plan": self._generate_action_plan(analysis_results)
        }
        
        # Save reports
        self._save_reports(report)
        
        return report
    
    def _generate_executive_summary(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary of technical debt assessment."""
        debt_score = analysis["debt_score"]
        total_cost = analysis["total_debt_cost_annual"]
        critical_items = len([item for item in self.debt_items if item.severity == "critical"])
        
        health_status = (
            "excellent" if debt_score >= 90 else
            "good" if debt_score >= 75 else
            "fair" if debt_score >= 60 else
            "poor"
        )
        
        return {
            "debt_score": debt_score,
            "health_status": health_status,
            "total_annual_cost": total_cost,
            "critical_issues": critical_items,
            "total_debt_items": len(self.debt_items),
            "files_analyzed": analysis["total_files_analyzed"],
            "key_findings": [
                f"Technical debt score: {debt_score:.1f}/100 ({health_status})",
                f"Annual cost of technical debt: ${total_cost:,.0f}",
                f"{critical_items} critical issues requiring immediate attention",
                f"{len(self.debt_items)} total debt items identified"
            ],
            "immediate_actions_needed": critical_items > 0
        }
    
    def _generate_action_plan(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate actionable plan for debt reduction."""
        recommendations = analysis["recommendations"]
        
        # Create sprint-based action plan
        sprint_1 = [rec for rec in recommendations if rec["priority"] == "P0"]
        sprint_2 = [rec for rec in recommendations if rec["priority"] == "P1"]
        sprint_3 = [rec for rec in recommendations if rec["priority"] == "P2"]
        
        return {
            "sprint_1_critical": {
                "focus": "Critical Issues",
                "duration_weeks": 2,
                "recommendations": sprint_1,
                "total_effort_hours": sum(rec["estimated_effort_hours"] for rec in sprint_1),
                "expected_score_improvement": "5-10 points"
            },
            "sprint_2_high_impact": {
                "focus": "High Impact Items",
                "duration_weeks": 4,
                "recommendations": sprint_2,
                "total_effort_hours": sum(rec["estimated_effort_hours"] for rec in sprint_2),
                "expected_score_improvement": "10-15 points"
            },
            "sprint_3_improvements": {
                "focus": "Gradual Improvements",
                "duration_weeks": 6,
                "recommendations": sprint_3,
                "total_effort_hours": sum(rec["estimated_effort_hours"] for rec in sprint_3),
                "expected_score_improvement": "5-10 points"
            },
            "total_investment": {
                "hours": sum(rec["estimated_effort_hours"] for rec in recommendations),
                "cost": sum(rec["estimated_cost"] for rec in recommendations),
                "expected_final_score": min(100, analysis["debt_score"] + 30)
            }
        }
    
    def _save_reports(self, report: Dict[str, Any]):
        """Save technical debt reports to files."""
        # Save JSON report
        json_path = Path(self.config["report_output"]["json_path"])
        json_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        # Save trends
        trend_path = Path(self.config["report_output"]["trend_path"])
        with open(trend_path, 'w') as f:
            json.dump([asdict(trend) for trend in self.trends], f, indent=2)
        
        self.logger.info(f"Technical debt assessment report saved to {json_path}")


def main():
    """Main function for standalone execution."""
    assessor = TechnicalDebtAssessor()
    
    # Generate and display report
    report = assessor.generate_report()
    
    print("Technical Debt Assessment Report")
    print("=" * 50)
    
    summary = report["executive_summary"]
    print(f"Debt Score: {summary['debt_score']:.1f}/100 ({summary['health_status']})")
    print(f"Annual Cost: ${summary['total_annual_cost']:,.0f}")
    print(f"Critical Issues: {summary['critical_issues']}")
    print(f"Total Debt Items: {summary['total_debt_items']}")
    print(f"Files Analyzed: {summary['files_analyzed']}")
    
    if summary["immediate_actions_needed"]:
        print("\n⚠️  IMMEDIATE ACTION REQUIRED")
        action_plan = report["action_plan"]
        sprint_1 = action_plan["sprint_1_critical"]
        print(f"  Sprint 1: {sprint_1['total_effort_hours']} hours to address critical issues")


if __name__ == "__main__":
    main()