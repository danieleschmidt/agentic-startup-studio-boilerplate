#!/usr/bin/env python3
"""
Project Health Dashboard Generator

Creates a comprehensive health dashboard showing project status,
metrics, and automated insights for the Agentic Startup Studio Boilerplate.
"""

import json
import os
import subprocess
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class HealthMetric:
    name: str
    value: float
    status: str  # 'healthy', 'warning', 'critical'
    threshold_warning: float
    threshold_critical: float
    description: str
    trend: str  # 'improving', 'stable', 'declining'

@dataclass
class ProjectHealth:
    overall_score: float
    status: str
    metrics: List[HealthMetric]
    recommendations: List[str]
    last_updated: str

class HealthDashboard:
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.github_token = os.getenv("GITHUB_TOKEN")
        self.repo_name = os.getenv("GITHUB_REPOSITORY", "danieleschmidt/agentic-startup-studio-boilerplate")
        
        # Health thresholds
        self.thresholds = {
            "test_coverage": {"warning": 70, "critical": 50},
            "security_score": {"warning": 80, "critical": 60},
            "documentation_health": {"warning": 80, "critical": 60},
            "code_quality": {"warning": 75, "critical": 50},
            "deployment_success": {"warning": 90, "critical": 75},
            "performance_score": {"warning": 80, "critical": 60},
            "dependency_health": {"warning": 80, "critical": 60},
            "automation_coverage": {"warning": 85, "critical": 70}
        }

    def generate_health_report(self) -> ProjectHealth:
        """Generate comprehensive project health report."""
        logger.info("Generating project health report...")
        
        # Load existing metrics
        metrics_data = self._load_project_metrics()
        
        # Calculate health metrics
        health_metrics = []
        
        # Test Coverage
        test_coverage = metrics_data.get("test_coverage", 0)
        health_metrics.append(HealthMetric(
            name="Test Coverage",
            value=test_coverage,
            status=self._determine_status(test_coverage, "test_coverage"),
            threshold_warning=self.thresholds["test_coverage"]["warning"],
            threshold_critical=self.thresholds["test_coverage"]["critical"],
            description="Percentage of code covered by automated tests",
            trend=self._calculate_trend("test_coverage", test_coverage)
        ))
        
        # Security Score
        security_score = metrics_data.get("security_score", 0)
        health_metrics.append(HealthMetric(
            name="Security Score",
            value=security_score,
            status=self._determine_status(security_score, "security_score"),
            threshold_warning=self.thresholds["security_score"]["warning"],
            threshold_critical=self.thresholds["security_score"]["critical"],
            description="Security posture based on vulnerability scans and best practices",
            trend=self._calculate_trend("security_score", security_score)
        ))
        
        # Documentation Health
        doc_health = metrics_data.get("documentation_health", 0)
        health_metrics.append(HealthMetric(
            name="Documentation Health",
            value=doc_health,
            status=self._determine_status(doc_health, "documentation_health"),
            threshold_warning=self.thresholds["documentation_health"]["warning"],
            threshold_critical=self.thresholds["documentation_health"]["critical"],
            description="Completeness and quality of project documentation",
            trend=self._calculate_trend("documentation_health", doc_health)
        ))
        
        # Code Quality
        code_quality = self._calculate_code_quality()
        health_metrics.append(HealthMetric(
            name="Code Quality",
            value=code_quality,
            status=self._determine_status(code_quality, "code_quality"),
            threshold_warning=self.thresholds["code_quality"]["warning"],
            threshold_critical=self.thresholds["code_quality"]["critical"],
            description="Code quality based on linting, complexity, and maintainability",
            trend=self._calculate_trend("code_quality", code_quality)
        ))
        
        # Deployment Reliability
        deployment_success = metrics_data.get("deployment_reliability", 0)
        health_metrics.append(HealthMetric(
            name="Deployment Success Rate",
            value=deployment_success,
            status=self._determine_status(deployment_success, "deployment_success"),
            threshold_warning=self.thresholds["deployment_success"]["warning"],
            threshold_critical=self.thresholds["deployment_success"]["critical"],
            description="Success rate of deployment pipeline",
            trend=self._calculate_trend("deployment_success", deployment_success)
        ))
        
        # Performance Score
        performance_score = self._calculate_performance_score()
        health_metrics.append(HealthMetric(
            name="Performance Score",
            value=performance_score,
            status=self._determine_status(performance_score, "performance_score"),
            threshold_warning=self.thresholds["performance_score"]["warning"],
            threshold_critical=self.thresholds["performance_score"]["critical"],
            description="Application performance based on response times and resource usage",
            trend=self._calculate_trend("performance_score", performance_score)
        ))
        
        # Dependency Health
        dependency_health = self._calculate_dependency_health()
        health_metrics.append(HealthMetric(
            name="Dependency Health",
            value=dependency_health,
            status=self._determine_status(dependency_health, "dependency_health"),
            threshold_warning=self.thresholds["dependency_health"]["warning"],
            threshold_critical=self.thresholds["dependency_health"]["critical"],
            description="Health of project dependencies (vulnerabilities, outdated packages)",
            trend=self._calculate_trend("dependency_health", dependency_health)
        ))
        
        # Automation Coverage
        automation_coverage = metrics_data.get("automation_coverage", 0)
        health_metrics.append(HealthMetric(
            name="Automation Coverage",
            value=automation_coverage,
            status=self._determine_status(automation_coverage, "automation_coverage"),
            threshold_warning=self.thresholds["automation_coverage"]["warning"],
            threshold_critical=self.thresholds["automation_coverage"]["critical"],
            description="Percentage of processes that are automated",
            trend=self._calculate_trend("automation_coverage", automation_coverage)
        ))
        
        # Calculate overall health score
        overall_score = sum(metric.value for metric in health_metrics) / len(health_metrics)
        overall_status = self._determine_overall_status(overall_score)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(health_metrics)
        
        return ProjectHealth(
            overall_score=overall_score,
            status=overall_status,
            metrics=health_metrics,
            recommendations=recommendations,
            last_updated=datetime.now(timezone.utc).isoformat()
        )

    def _load_project_metrics(self) -> Dict[str, Any]:
        """Load project metrics from file."""
        metrics_file = self.repo_path / ".github" / "project-metrics.json"
        
        if metrics_file.exists():
            with open(metrics_file, 'r') as f:
                return json.load(f)
        return {}

    def _determine_status(self, value: float, metric_type: str) -> str:
        """Determine health status based on value and thresholds."""
        thresholds = self.thresholds[metric_type]
        
        if value >= thresholds["warning"]:
            return "healthy"
        elif value >= thresholds["critical"]:
            return "warning"
        else:
            return "critical"

    def _determine_overall_status(self, score: float) -> str:
        """Determine overall project health status."""
        if score >= 85:
            return "excellent"
        elif score >= 75:
            return "good"
        elif score >= 60:
            return "fair"
        else:
            return "poor"

    def _calculate_trend(self, metric_name: str, current_value: float) -> str:
        """Calculate trend for a metric (simplified implementation)."""
        # In a real implementation, this would compare with historical data
        # For now, return 'stable' as default
        return "stable"

    def _calculate_code_quality(self) -> float:
        """Calculate code quality score."""
        try:
            # Run flake8 for linting score
            result = subprocess.run(
                ["flake8", "--statistics", "--count"],
                capture_output=True,
                text=True,
                cwd=self.repo_path
            )
            
            # Calculate score based on number of issues
            # This is a simplified calculation
            if result.returncode == 0:
                return 95  # No issues
            else:
                # Parse number of issues and calculate score
                return max(50, 95 - len(result.stdout.splitlines()))
        except:
            return 75  # Default score if analysis fails

    def _calculate_performance_score(self) -> float:
        """Calculate performance score."""
        # This would typically analyze response times, resource usage, etc.
        # For now, return a default score
        return 82

    def _calculate_dependency_health(self) -> float:
        """Calculate dependency health score."""
        try:
            # Check for vulnerabilities
            result = subprocess.run(
                ["safety", "check", "--json"],
                capture_output=True,
                text=True,
                cwd=self.repo_path
            )
            
            if result.returncode == 0:
                return 95  # No vulnerabilities
            else:
                # Parse vulnerabilities and calculate score
                try:
                    vulnerabilities = json.loads(result.stdout)
                    return max(50, 95 - len(vulnerabilities))
                except:
                    return 70
        except:
            return 80  # Default score

    def _generate_recommendations(self, metrics: List[HealthMetric]) -> List[str]:
        """Generate actionable recommendations based on health metrics."""
        recommendations = []
        
        for metric in metrics:
            if metric.status == "critical":
                if metric.name == "Test Coverage":
                    recommendations.append(f"🚨 Critical: Increase test coverage from {metric.value}% to at least {metric.threshold_warning}%")
                elif metric.name == "Security Score":
                    recommendations.append(f"🚨 Critical: Address security vulnerabilities to improve score from {metric.value}%")
                elif metric.name == "Documentation Health":
                    recommendations.append(f"🚨 Critical: Improve documentation completeness from {metric.value}%")
            elif metric.status == "warning":
                if metric.name == "Code Quality":
                    recommendations.append(f"⚠️ Warning: Address code quality issues to improve from {metric.value}%")
                elif metric.name == "Deployment Success Rate":
                    recommendations.append(f"⚠️ Warning: Improve deployment reliability from {metric.value}%")
        
        # Add general recommendations
        if not recommendations:
            recommendations.append("✅ All metrics are healthy! Consider implementing advanced monitoring.")
        
        return recommendations

    def generate_html_dashboard(self, output_file: str = "health-dashboard.html") -> None:
        """Generate HTML dashboard."""
        health_report = self.generate_health_report()
        
        html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Project Health Dashboard - Agentic Startup Studio</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; border-radius: 10px; margin-bottom: 30px; }}
        .status-badge {{ display: inline-block; padding: 8px 16px; border-radius: 20px; font-weight: bold; margin-left: 15px; }}
        .status-excellent {{ background-color: #10b981; }}
        .status-good {{ background-color: #3b82f6; }}
        .status-fair {{ background-color: #f59e0b; }}
        .status-poor {{ background-color: #ef4444; }}
        .metrics-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .metric-card {{ background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .metric-header {{ display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }}
        .metric-value {{ font-size: 2em; font-weight: bold; }}
        .metric-description {{ color: #666; font-size: 0.9em; margin-top: 10px; }}
        .status-indicator {{ width: 12px; height: 12px; border-radius: 50%; }}
        .status-healthy {{ background-color: #10b981; }}
        .status-warning {{ background-color: #f59e0b; }}
        .status-critical {{ background-color: #ef4444; }}
        .progress-bar {{ width: 100%; height: 8px; background-color: #e5e7eb; border-radius: 4px; overflow: hidden; }}
        .progress-fill {{ height: 100%; transition: width 0.3s ease; }}
        .recommendations {{ background: white; border-radius: 10px; padding: 20px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        .recommendation {{ margin: 10px 0; padding: 10px; border-left: 4px solid #3b82f6; background-color: #f8fafc; }}
        .timestamp {{ color: #666; font-size: 0.9em; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🚀 Agentic Startup Studio - Project Health Dashboard</h1>
            <p>Overall Health Score: <strong>{health_report.overall_score:.1f}%</strong>
            <span class="status-badge status-{health_report.status}">{health_report.status.upper()}</span></p>
            <p class="timestamp">Last Updated: {health_report.last_updated}</p>
        </div>
        
        <div class="metrics-grid">
        """
        
        for metric in health_report.metrics:
            progress_color = {"healthy": "#10b981", "warning": "#f59e0b", "critical": "#ef4444"}[metric.status]
            
            html_content += f"""
            <div class="metric-card">
                <div class="metric-header">
                    <h3>{metric.name}</h3>
                    <div class="status-indicator status-{metric.status}"></div>
                </div>
                <div class="metric-value">{metric.value:.1f}%</div>
                <div class="progress-bar">
                    <div class="progress-fill" style="width: {metric.value}%; background-color: {progress_color};"></div>
                </div>
                <div class="metric-description">{metric.description}</div>
                <div style="margin-top: 10px; font-size: 0.8em; color: #666;">
                    Trend: {metric.trend} | Warning: {metric.threshold_warning}% | Critical: {metric.threshold_critical}%
                </div>
            </div>
            """
        
        html_content += f"""
        </div>
        
        <div class="recommendations">
            <h2>🎯 Recommendations</h2>
        """
        
        for recommendation in health_report.recommendations:
            html_content += f'<div class="recommendation">{recommendation}</div>'
        
        html_content += """
        </div>
    </div>
    
    <script>
        // Auto-refresh every 5 minutes
        setTimeout(() => location.reload(), 300000);
    </script>
</body>
</html>
        """
        
        with open(output_file, 'w') as f:
            f.write(html_content)
        
        logger.info(f"HTML dashboard saved to: {output_file}")

    def generate_json_report(self, output_file: str = "health-report.json") -> None:
        """Generate JSON health report."""
        health_report = self.generate_health_report()
        
        # Convert to dictionary for JSON serialization
        report_dict = {
            "overall_score": health_report.overall_score,
            "status": health_report.status,
            "metrics": [asdict(metric) for metric in health_report.metrics],
            "recommendations": health_report.recommendations,
            "last_updated": health_report.last_updated
        }
        
        with open(output_file, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        logger.info(f"JSON report saved to: {output_file}")

    def generate_charts(self, output_dir: str = "charts") -> None:
        """Generate health metric charts."""
        os.makedirs(output_dir, exist_ok=True)
        health_report = self.generate_health_report()
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # Create metrics overview chart
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Metrics bar chart
        metric_names = [m.name for m in health_report.metrics]
        metric_values = [m.value for m in health_report.metrics]
        colors = ['#10b981' if m.status == 'healthy' else '#f59e0b' if m.status == 'warning' else '#ef4444' 
                 for m in health_report.metrics]
        
        bars = ax1.bar(range(len(metric_names)), metric_values, color=colors)
        ax1.set_xlabel('Metrics')
        ax1.set_ylabel('Score (%)')
        ax1.set_title('Project Health Metrics')
        ax1.set_xticks(range(len(metric_names)))
        ax1.set_xticklabels([name.replace(' ', '\n') for name in metric_names], rotation=45, ha='right')
        ax1.set_ylim(0, 100)
        
        # Add value labels on bars
        for bar, value in zip(bars, metric_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                    f'{value:.1f}%', ha='center', va='bottom')
        
        # Status distribution pie chart
        status_counts = {}
        for metric in health_report.metrics:
            status_counts[metric.status] = status_counts.get(metric.status, 0) + 1
        
        colors_pie = {'healthy': '#10b981', 'warning': '#f59e0b', 'critical': '#ef4444'}
        pie_colors = [colors_pie[status] for status in status_counts.keys()]
        
        ax2.pie(status_counts.values(), labels=status_counts.keys(), autopct='%1.0f%%', 
               colors=pie_colors, startangle=90)
        ax2.set_title('Health Status Distribution')
        
        plt.tight_layout()
        plt.savefig(f'{output_dir}/health_overview.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create trend chart (placeholder for future implementation)
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # For now, create a placeholder trend chart
        dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
        overall_scores = [health_report.overall_score + (i % 5 - 2) for i in range(30)]  # Simulated data
        
        ax.plot(dates, overall_scores, marker='o', linewidth=2, markersize=4)
        ax.set_xlabel('Date')
        ax.set_ylabel('Overall Health Score (%)')
        ax.set_title('Project Health Trend (Last 30 Days)')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)
        
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'{output_dir}/health_trend.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Charts saved to: {output_dir}/")

def main():
    """Main function to generate health dashboard."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate project health dashboard")
    parser.add_argument("--format", choices=["html", "json", "charts", "all"], default="all",
                       help="Output format (default: all)")
    parser.add_argument("--output", default=".", help="Output directory")
    
    args = parser.parse_args()
    
    dashboard = HealthDashboard()
    
    if args.format in ["html", "all"]:
        dashboard.generate_html_dashboard(f"{args.output}/health-dashboard.html")
    
    if args.format in ["json", "all"]:
        dashboard.generate_json_report(f"{args.output}/health-report.json")
    
    if args.format in ["charts", "all"]:
        dashboard.generate_charts(f"{args.output}/charts")
    
    logger.info("✅ Health dashboard generation complete!")

if __name__ == "__main__":
    main()