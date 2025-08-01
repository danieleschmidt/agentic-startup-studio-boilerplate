#!/usr/bin/env python3
"""
Terragon Autonomous SDLC - Continuous Value Discovery Engine

This engine continuously discovers, scores, and prioritizes work items
based on WSJF, ICE, and Technical Debt scoring models.
"""

import json
import subprocess
import yaml
import os
import re
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class WorkItem:
    """Represents a discovered work item with scoring and metadata."""
    id: str
    title: str
    description: str
    category: str
    source: str
    priority: str
    estimated_effort: float
    confidence: float
    impact_score: float
    ease_score: float
    technical_debt_score: float
    security_score: float
    wsjf_score: float
    ice_score: float
    composite_score: float
    files_affected: List[str]
    created_at: str
    status: str = "discovered"
    execution_history: List[Dict] = None

    def __post_init__(self):
        if self.execution_history is None:
            self.execution_history = []


class ValueDiscoveryEngine:
    """Main engine for discovering and scoring value opportunities."""
    
    def __init__(self, config_path: str = ".terragon/value-config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.work_items: List[WorkItem] = []
        self.learning_data: Dict = {}
        self._setup_paths()
    
    def _load_config(self) -> Dict:
        """Load the value configuration file."""
        try:
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            logger.error(f"Config file not found: {self.config_path}")
            return {}
    
    def _setup_paths(self):
        """Setup required directories and files."""
        terragon_dir = Path(".terragon")
        terragon_dir.mkdir(exist_ok=True)
        
        self.backlog_file = terragon_dir / "backlog.json"
        self.metrics_file = terragon_dir / "value-metrics.json"
        self.learning_file = terragon_dir / "learning-data.json"
    
    def discover_work_items(self) -> List[WorkItem]:
        """Main discovery method that aggregates from all sources."""
        logger.info("Starting comprehensive work item discovery...")
        
        discovered_items = []
        
        # Discovery from multiple sources
        discovered_items.extend(self._discover_from_git_history())
        discovered_items.extend(self._discover_from_static_analysis())
        discovered_items.extend(self._discover_from_security_scan())
        discovered_items.extend(self._discover_from_dependencies())
        discovered_items.extend(self._discover_from_code_comments())
        discovered_items.extend(self._discover_from_test_coverage())
        discovered_items.extend(self._discover_from_documentation())
        discovered_items.extend(self._discover_from_performance())
        
        # Score all discovered items
        for item in discovered_items:
            self._calculate_composite_score(item)
        
        # Deduplicate and sort by score
        unique_items = self._deduplicate_items(discovered_items)
        self.work_items = sorted(unique_items, key=lambda x: x.composite_score, reverse=True)
        
        logger.info(f"Discovered {len(self.work_items)} unique work items")
        return self.work_items
    
    def _discover_from_git_history(self) -> List[WorkItem]:
        """Discover work items from Git history analysis."""
        items = []
        
        try:
            # Look for TODO, FIXME, HACK markers in recent commits
            result = subprocess.run([
                'git', 'log', '--oneline', '--grep=TODO\\|FIXME\\|HACK\\|TEMPORARY', 
                '--since=30.days.ago'
            ], capture_output=True, text=True)
            
            for line in result.stdout.strip().split('\n'):
                if line:
                    commit_hash, message = line.split(' ', 1)
                    items.append(WorkItem(
                        id=f"git-{commit_hash[:8]}",
                        title=f"Address commit marker: {message[:50]}...",
                        description=f"Git history indicates technical debt: {message}",
                        category="technical_debt",
                        source="git_history",
                        priority="medium",
                        estimated_effort=2.0,
                        confidence=0.7,
                        impact_score=5.0,
                        ease_score=7.0,
                        technical_debt_score=8.0,
                        security_score=0.0,
                        wsjf_score=0.0,
                        ice_score=0.0,
                        composite_score=0.0,
                        files_affected=[],
                        created_at=datetime.now().isoformat()
                    ))
        except Exception as e:
            logger.warning(f"Git history analysis failed: {e}")
        
        return items
    
    def _discover_from_static_analysis(self) -> List[WorkItem]:
        """Discover issues from static analysis tools."""
        items = []
        
        # Run ruff for Python files
        try:
            result = subprocess.run([
                'ruff', 'check', '.', '--format=json'
            ], capture_output=True, text=True)
            
            if result.stdout:
                ruff_issues = json.loads(result.stdout)
                for issue in ruff_issues[:10]:  # Limit to top 10
                    items.append(WorkItem(
                        id=f"ruff-{hashlib.md5(str(issue).encode()).hexdigest()[:8]}",
                        title=f"Fix {issue.get('code', 'unknown')}: {issue.get('message', '')[:50]}",
                        description=f"Static analysis issue: {issue.get('message', '')}",
                        category="code_quality",
                        source="static_analysis",
                        priority="low",
                        estimated_effort=0.5,
                        confidence=0.9,
                        impact_score=3.0,
                        ease_score=8.0,
                        technical_debt_score=4.0,
                        security_score=0.0,
                        wsjf_score=0.0,
                        ice_score=0.0,
                        composite_score=0.0,
                        files_affected=[issue.get('filename', '')],
                        created_at=datetime.now().isoformat()
                    ))
        except Exception as e:
            logger.warning(f"Ruff analysis failed: {e}")
        
        return items
    
    def _discover_from_security_scan(self) -> List[WorkItem]:
        """Discover security-related work items."""
        items = []
        
        # Check for dependency vulnerabilities
        try:
            result = subprocess.run([
                'safety', 'check', '--json'
            ], capture_output=True, text=True)
            
            if result.stdout:
                safety_data = json.loads(result.stdout)
                for vuln in safety_data.get('vulnerabilities', [])[:5]:
                    items.append(WorkItem(
                        id=f"security-{vuln.get('id', 'unknown')}",
                        title=f"Fix vulnerability in {vuln.get('package_name', 'unknown')}",
                        description=f"Security vulnerability: {vuln.get('advisory', '')}",
                        category="security",
                        source="security_scan",
                        priority="high",
                        estimated_effort=4.0,
                        confidence=0.95,
                        impact_score=9.0,
                        ease_score=6.0,
                        technical_debt_score=2.0,
                        security_score=10.0,
                        wsjf_score=0.0,
                        ice_score=0.0,
                        composite_score=0.0,
                        files_affected=["requirements.txt"],
                        created_at=datetime.now().isoformat()
                    ))
        except Exception as e:
            logger.info(f"Safety scan failed (may be normal): {e}")
        
        return items
    
    def _discover_from_dependencies(self) -> List[WorkItem]:
        """Discover dependency update opportunities."""
        items = []
        
        # Check for outdated Python packages
        try:
            result = subprocess.run([
                'pip', 'list', '--outdated', '--format=json'
            ], capture_output=True, text=True)
            
            if result.stdout:
                outdated = json.loads(result.stdout)
                for pkg in outdated[:5]:  # Top 5 outdated packages
                    items.append(WorkItem(
                        id=f"dep-{pkg['name'].lower()}",
                        title=f"Update {pkg['name']} from {pkg['version']} to {pkg['latest_version']}",
                        description=f"Dependency update available for {pkg['name']}",
                        category="dependency_update",
                        source="dependency_scan",
                        priority="medium",
                        estimated_effort=1.0,
                        confidence=0.8,
                        impact_score=4.0,
                        ease_score=9.0,
                        technical_debt_score=3.0,
                        security_score=2.0,
                        wsjf_score=0.0,
                        ice_score=0.0,
                        composite_score=0.0,
                        files_affected=["requirements.txt", "pyproject.toml"],
                        created_at=datetime.now().isoformat()
                    ))
        except Exception as e:
            logger.warning(f"Dependency check failed: {e}")
        
        return items
    
    def _discover_from_code_comments(self) -> List[WorkItem]:
        """Discover work items from code comments (TODO, FIXME, etc.)."""
        items = []
        
        try:
            # Search for TODO, FIXME, HACK comments
            patterns = ["TODO", "FIXME", "HACK", "XXX", "DEPRECATED"]
            for pattern in patterns:
                result = subprocess.run([
                    'rg', '--type', 'py', '--type', 'js', '--type', 'ts', 
                    '-n', pattern
                ], capture_output=True, text=True)
                
                for line in result.stdout.strip().split('\n')[:5]:  # Top 5 per pattern
                    if line and ':' in line:
                        file_path, line_num, comment = line.split(':', 2)
                        items.append(WorkItem(
                            id=f"comment-{hashlib.md5(line.encode()).hexdigest()[:8]}",
                            title=f"Address {pattern} in {os.path.basename(file_path)}",
                            description=f"Code comment indicates work needed: {comment.strip()}",
                            category="technical_debt",
                            source="code_comments",
                            priority="medium",
                            estimated_effort=3.0,
                            confidence=0.6,
                            impact_score=6.0,
                            ease_score=5.0,
                            technical_debt_score=7.0,
                            security_score=0.0,
                            wsjf_score=0.0,
                            ice_score=0.0,
                            composite_score=0.0,
                            files_affected=[file_path],
                            created_at=datetime.now().isoformat()
                        ))
        except Exception as e:
            logger.warning(f"Code comment analysis failed: {e}")
        
        return items
    
    def _discover_from_test_coverage(self) -> List[WorkItem]:
        """Discover areas needing test coverage improvement."""
        items = []
        
        try:
            # Run coverage analysis
            subprocess.run(['python', '-m', 'pytest', '--cov=.', '--cov-report=json'], 
                         capture_output=True)
            
            if os.path.exists('coverage.json'):
                with open('coverage.json', 'r') as f:
                    coverage_data = json.load(f)
                
                files = coverage_data.get('files', {})
                for file_path, file_data in files.items():
                    coverage_percent = file_data.get('summary', {}).get('percent_covered', 100)
                    if coverage_percent < 80:  # Below threshold
                        items.append(WorkItem(
                            id=f"coverage-{hashlib.md5(file_path.encode()).hexdigest()[:8]}",
                            title=f"Improve test coverage for {os.path.basename(file_path)}",
                            description=f"Test coverage is {coverage_percent:.1f}%, below 80% threshold",
                            category="testing",
                            source="test_coverage",
                            priority="medium",
                            estimated_effort=6.0,
                            confidence=0.7,
                            impact_score=7.0,
                            ease_score=4.0,
                            technical_debt_score=6.0,
                            security_score=1.0,
                            wsjf_score=0.0,
                            ice_score=0.0,
                            composite_score=0.0,
                            files_affected=[file_path],
                            created_at=datetime.now().isoformat()
                        ))
        except Exception as e:
            logger.info(f"Coverage analysis failed: {e}")
        
        return items
    
    def _discover_from_documentation(self) -> List[WorkItem]:
        """Discover documentation improvement opportunities."""
        items = []
        
        # Check for missing or outdated documentation
        docs_to_check = [
            ("README.md", "Update README with latest features"),
            ("CONTRIBUTING.md", "Review contribution guidelines"),
            ("docs/ARCHITECTURE.md", "Update architecture documentation"),
            ("CHANGELOG.md", "Update changelog with recent changes")
        ]
        
        for doc_path, suggestion in docs_to_check:
            if os.path.exists(doc_path):
                # Check if file is old (>30 days since last update)
                mtime = os.path.getmtime(doc_path)
                if datetime.now().timestamp() - mtime > 30 * 24 * 3600:
                    items.append(WorkItem(
                        id=f"docs-{hashlib.md5(doc_path.encode()).hexdigest()[:8]}",
                        title=suggestion,
                        description=f"Documentation file {doc_path} may need updates",
                        category="documentation",
                        source="documentation_scan",
                        priority="low",
                        estimated_effort=2.0,
                        confidence=0.5,
                        impact_score=4.0,
                        ease_score=8.0,
                        technical_debt_score=2.0,
                        security_score=0.0,
                        wsjf_score=0.0,
                        ice_score=0.0,
                        composite_score=0.0,
                        files_affected=[doc_path],
                        created_at=datetime.now().isoformat()
                    ))
        
        return items
    
    def _discover_from_performance(self) -> List[WorkItem]:
        """Discover performance optimization opportunities."""
        items = []
        
        # Placeholder for performance analysis
        # In a real implementation, this would analyze:
        # - Large file sizes
        # - Complex functions (cyclomatic complexity)
        # - Slow database queries
        # - Bundle size analysis
        
        # Example: Check for large Python files that might need refactoring
        try:
            result = subprocess.run([
                'find', '.', '-name', '*.py', '-size', '+500c', '-not', '-path', './.*'
            ], capture_output=True, text=True)
            
            for file_path in result.stdout.strip().split('\n')[:3]:
                if file_path and os.path.exists(file_path):
                    file_size = os.path.getsize(file_path)
                    if file_size > 10000:  # > 10KB
                        items.append(WorkItem(
                            id=f"perf-{hashlib.md5(file_path.encode()).hexdigest()[:8]}",
                            title=f"Consider refactoring large file {os.path.basename(file_path)}",
                            description=f"File {file_path} is {file_size} bytes, may benefit from refactoring",
                            category="performance",
                            source="performance_scan",
                            priority="low",
                            estimated_effort=8.0,
                            confidence=0.4,
                            impact_score=5.0,
                            ease_score=3.0,
                            technical_debt_score=5.0,
                            security_score=0.0,
                            wsjf_score=0.0,
                            ice_score=0.0,
                            composite_score=0.0,
                            files_affected=[file_path],
                            created_at=datetime.now().isoformat()
                        ))
        except Exception as e:
            logger.warning(f"Performance analysis failed: {e}")
        
        return items
    
    def _calculate_composite_score(self, item: WorkItem):
        """Calculate the composite score using WSJF, ICE, and Technical Debt models."""
        maturity = self.config.get('metadata', {}).get('maturity_level', 'developing')
        weights = self.config.get('scoring', {}).get('weights', {}).get(maturity, {})
        thresholds = self.config.get('scoring', {}).get('thresholds', {})
        
        # WSJF Calculation (Weighted Shortest Job First)
        user_business_value = item.impact_score * 0.4
        time_criticality = self._calculate_time_criticality(item) * 0.3
        risk_reduction = self._calculate_risk_reduction(item) * 0.2
        opportunity_enablement = self._calculate_opportunity_enablement(item) * 0.1
        
        cost_of_delay = user_business_value + time_criticality + risk_reduction + opportunity_enablement
        job_size = max(item.estimated_effort, 0.5)  # Avoid division by zero
        
        item.wsjf_score = cost_of_delay / job_size
        
        # ICE Calculation (Impact, Confidence, Ease)
        item.ice_score = item.impact_score * item.confidence * item.ease_score
        
        # Apply security and compliance boosts
        security_boost = thresholds.get('securityBoost', 2.0) if item.security_score > 7 else 1.0
        compliance_boost = thresholds.get('complianceBoost', 1.8) if item.category == 'compliance' else 1.0
        
        # Composite Score Calculation
        wsjf_weight = weights.get('wsjf', 0.5)
        ice_weight = weights.get('ice', 0.2)
        debt_weight = weights.get('technicalDebt', 0.2)
        security_weight = weights.get('security', 0.1)
        
        normalized_wsjf = min(item.wsjf_score / 10, 10)  # Normalize to 0-10 scale
        normalized_ice = min(item.ice_score / 100, 10)   # Normalize to 0-10 scale
        normalized_debt = min(item.technical_debt_score, 10)
        normalized_security = min(item.security_score, 10)
        
        item.composite_score = (
            wsjf_weight * normalized_wsjf +
            ice_weight * normalized_ice +
            debt_weight * normalized_debt +
            security_weight * normalized_security
        ) * security_boost * compliance_boost
        
        # Apply category-specific adjustments
        category_multipliers = {
            'security': 1.5,
            'compliance': 1.3,
            'performance': 1.2,
            'technical_debt': 1.1,
            'documentation': 0.8,
            'testing': 1.0
        }
        
        multiplier = category_multipliers.get(item.category, 1.0)
        item.composite_score *= multiplier
    
    def _calculate_time_criticality(self, item: WorkItem) -> float:
        """Calculate time criticality based on item characteristics."""
        if item.category == 'security':
            return 10.0
        elif item.category == 'compliance':
            return 8.0
        elif item.source == 'dependency_scan':
            return 6.0
        else:
            return 4.0
    
    def _calculate_risk_reduction(self, item: WorkItem) -> float:
        """Calculate risk reduction value."""
        if item.security_score > 7:
            return 9.0
        elif item.category == 'technical_debt':
            return 6.0
        elif item.category == 'testing':
            return 5.0
        else:
            return 3.0
    
    def _calculate_opportunity_enablement(self, item: WorkItem) -> float:
        """Calculate opportunity enablement value."""
        if item.category in ['performance', 'automation']:
            return 8.0
        elif item.category == 'documentation':
            return 6.0
        elif item.category == 'testing':
            return 5.0
        else:
            return 3.0
    
    def _deduplicate_items(self, items: List[WorkItem]) -> List[WorkItem]:
        """Remove duplicate work items based on similarity."""
        unique_items = []
        seen_titles = set()
        
        for item in items:
            # Simple deduplication based on title similarity
            title_key = item.title.lower().replace(' ', '')[:30]
            if title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_items.append(item)
        
        return unique_items
    
    def save_backlog(self):
        """Save the current backlog to disk."""
        backlog_data = {
            'last_updated': datetime.now().isoformat(),
            'total_items': len(self.work_items),
            'items': [asdict(item) for item in self.work_items]
        }
        
        with open(self.backlog_file, 'w') as f:
            json.dump(backlog_data, f, indent=2)
        
        logger.info(f"Saved {len(self.work_items)} items to backlog")
    
    def get_next_best_value_item(self) -> Optional[WorkItem]:
        """Get the next highest-value work item for execution."""
        for item in self.work_items:
            if item.status == 'discovered':
                # Apply selection filters
                if item.composite_score < self.config.get('scoring', {}).get('thresholds', {}).get('minScore', 10):
                    continue
                
                return item
        
        return None
    
    def generate_backlog_markdown(self) -> str:
        """Generate a markdown representation of the backlog."""
        if not self.work_items:
            return "# 📊 Autonomous Value Backlog\n\nNo items discovered yet."
        
        next_item = self.get_next_best_value_item()
        
        md = f"""# 📊 Autonomous Value Backlog

Last Updated: {datetime.now().isoformat()}
Total Items: {len(self.work_items)}

## 🎯 Next Best Value Item
"""
        
        if next_item:
            md += f"""**[{next_item.id.upper()}] {next_item.title}**
- **Composite Score**: {next_item.composite_score:.1f}
- **WSJF**: {next_item.wsjf_score:.1f} | **ICE**: {next_item.ice_score:.1f} | **Tech Debt**: {next_item.technical_debt_score:.1f}
- **Category**: {next_item.category.replace('_', ' ').title()}
- **Estimated Effort**: {next_item.estimated_effort} hours
- **Expected Impact**: {next_item.description}

"""
        else:
            md += "No qualifying items found.\n\n"
        
        md += """## 📋 Top 10 Backlog Items

| Rank | ID | Title | Score | Category | Est. Hours |
|------|-----|--------|---------|----------|------------|
"""
        
        for i, item in enumerate(self.work_items[:10], 1):
            title_short = item.title[:40] + "..." if len(item.title) > 40 else item.title
            category = item.category.replace('_', ' ').title()
            md += f"| {i} | {item.id.upper()} | {title_short} | {item.composite_score:.1f} | {category} | {item.estimated_effort} |\n"
        
        md += f"""

## 📈 Discovery Stats
- **Items by Category**:
"""
        
        # Category breakdown
        categories = {}
        for item in self.work_items:
            cat = item.category.replace('_', ' ').title()
            categories[cat] = categories.get(cat, 0) + 1
        
        for cat, count in sorted(categories.items()):
            md += f"  - {cat}: {count}\n"
        
        md += f"""
- **Items by Source**:
"""
        
        # Source breakdown
        sources = {}
        for item in self.work_items:
            source = item.source.replace('_', ' ').title()
            sources[source] = sources.get(source, 0) + 1
        
        for source, count in sorted(sources.items()):
            md += f"  - {source}: {count}\n"
        
        return md


if __name__ == "__main__":
    engine = ValueDiscoveryEngine()
    engine.discover_work_items()
    engine.save_backlog()
    
    # Generate backlog markdown
    backlog_md = engine.generate_backlog_markdown()
    with open(".terragon/BACKLOG.md", "w") as f:
        f.write(backlog_md)
    
    print(f"Discovered {len(engine.work_items)} work items")
    print(f"Next best value item: {engine.get_next_best_value_item()}")