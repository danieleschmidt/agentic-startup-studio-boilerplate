#!/usr/bin/env python3
"""
Advanced Performance Analysis and Optimization System

This module provides comprehensive performance analysis, automated bottleneck detection,
and optimization recommendations for the Agentic Startup Studio Boilerplate.

Features:
- Real-time performance monitoring
- Automated bottleneck detection
- Performance regression analysis
- Resource optimization recommendations
- AI/ML workload performance tracking
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import psutil
import aiohttp
import pandas as pd
import numpy as np
from prometheus_client.parser import text_string_to_metric_families


@dataclass
class PerformanceMetric:
    """Performance metric data structure."""
    timestamp: datetime
    metric_name: str
    value: float
    unit: str
    labels: Dict[str, str]
    source: str


@dataclass
class PerformanceThreshold:
    """Performance threshold configuration."""
    metric_name: str
    warning_threshold: float
    critical_threshold: float
    direction: str  # 'above' or 'below'
    window_minutes: int = 5


@dataclass 
class OptimizationRecommendation:
    """Optimization recommendation structure."""
    component: str
    issue: str
    recommendation: str
    impact: str  # 'high', 'medium', 'low'
    effort: str  # 'high', 'medium', 'low'
    estimated_improvement: str
    implementation_guide: str


class PerformanceAnalyzer:
    """Advanced performance analysis and optimization system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the performance analyzer."""
        self.config = self._load_config(config_path)
        self.metrics_history: List[PerformanceMetric] = []
        self.thresholds = self._initialize_thresholds()
        self.logger = self._setup_logging()
        
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            "prometheus_url": "http://localhost:9090",
            "analysis_window_hours": 24,
            "collection_interval_seconds": 30,
            "alert_webhook_url": None,
            "optimization_report_path": "reports/performance_analysis.json",
            "ml_model_monitoring": {
                "enabled": True,
                "token_usage_threshold": 10000,
                "response_time_threshold": 30,
                "memory_threshold_mb": 512
            },
            "database_monitoring": {
                "enabled": True,
                "slow_query_threshold_ms": 1000,
                "connection_pool_utilization_threshold": 0.8
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                user_config = json.load(f)
                default_config.update(user_config)
                
        return default_config
    
    def _initialize_thresholds(self) -> List[PerformanceThreshold]:
        """Initialize performance thresholds."""
        return [
            # API Performance
            PerformanceThreshold("http_request_duration_seconds", 0.2, 0.5, "above", 5),
            PerformanceThreshold("http_requests_per_second", 100, 50, "below", 5),
            
            # System Resources
            PerformanceThreshold("cpu_usage_percent", 70, 90, "above", 5),
            PerformanceThreshold("memory_usage_percent", 80, 95, "above", 5),
            PerformanceThreshold("disk_usage_percent", 85, 95, "above", 60),
            
            # Database Performance
            PerformanceThreshold("db_connection_pool_utilization", 0.8, 0.95, "above", 5),
            PerformanceThreshold("db_query_duration_ms", 1000, 5000, "above", 5),
            
            # AI/ML Performance
            PerformanceThreshold("agent_response_time_seconds", 30, 60, "above", 5),
            PerformanceThreshold("token_usage_per_minute", 1000, 5000, "above", 15),
            PerformanceThreshold("model_memory_usage_mb", 512, 1024, "above", 5),
            
            # Frontend Performance
            PerformanceThreshold("first_contentful_paint_ms", 1500, 3000, "above", 5),
            PerformanceThreshold("largest_contentful_paint_ms", 2500, 4000, "above", 5),
            PerformanceThreshold("cumulative_layout_shift", 0.1, 0.25, "above", 5),
        ]
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger("performance_analyzer")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    async def collect_metrics(self) -> List[PerformanceMetric]:
        """Collect performance metrics from various sources."""
        metrics = []
        
        # Collect system metrics
        system_metrics = await self._collect_system_metrics()
        metrics.extend(system_metrics)
        
        # Collect Prometheus metrics
        prometheus_metrics = await self._collect_prometheus_metrics()
        metrics.extend(prometheus_metrics)
        
        # Collect application-specific metrics
        app_metrics = await self._collect_application_metrics()
        metrics.extend(app_metrics)
        
        # Store metrics in history
        self.metrics_history.extend(metrics)
        
        # Keep only recent metrics (configurable window)
        cutoff_time = datetime.now() - timedelta(
            hours=self.config["analysis_window_hours"]
        )
        self.metrics_history = [
            m for m in self.metrics_history if m.timestamp > cutoff_time
        ]
        
        return metrics
    
    async def _collect_system_metrics(self) -> List[PerformanceMetric]:
        """Collect system-level performance metrics."""
        timestamp = datetime.now()
        metrics = []
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            metrics.append(PerformanceMetric(
                timestamp=timestamp,
                metric_name="cpu_usage_percent",
                value=cpu_percent,
                unit="percent",
                labels={"source": "system"},
                source="psutil"
            ))
            
            # Memory metrics
            memory = psutil.virtual_memory()
            metrics.append(PerformanceMetric(
                timestamp=timestamp,
                metric_name="memory_usage_percent",
                value=memory.percent,
                unit="percent",
                labels={"source": "system"},
                source="psutil"
            ))
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            disk_percent = (disk.used / disk.total) * 100
            metrics.append(PerformanceMetric(
                timestamp=timestamp,
                metric_name="disk_usage_percent",
                value=disk_percent,
                unit="percent",
                labels={"source": "system"},
                source="psutil"
            ))
            
            # Network I/O
            net_io = psutil.net_io_counters()
            metrics.extend([
                PerformanceMetric(
                    timestamp=timestamp,
                    metric_name="network_bytes_sent",
                    value=net_io.bytes_sent,
                    unit="bytes",
                    labels={"source": "system"},
                    source="psutil"
                ),
                PerformanceMetric(
                    timestamp=timestamp,
                    metric_name="network_bytes_recv",
                    value=net_io.bytes_recv,
                    unit="bytes",
                    labels={"source": "system"},
                    source="psutil"
                )
            ])
            
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            
        return metrics
    
    async def _collect_prometheus_metrics(self) -> List[PerformanceMetric]:
        """Collect metrics from Prometheus."""
        metrics = []
        
        try:
            async with aiohttp.ClientSession() as session:
                # Query specific metrics
                queries = [
                    'http_request_duration_seconds{quantile="0.95"}',
                    'http_requests_total',
                    'database_connections_active',
                    'database_query_duration_seconds',
                    'agent_execution_duration_seconds',
                    'token_usage_total',
                    'model_memory_usage_bytes'
                ]
                
                for query in queries:
                    url = f"{self.config['prometheus_url']}/api/v1/query"
                    params = {"query": query}
                    
                    async with session.get(url, params=params) as response:
                        if response.status == 200:
                            data = await response.json()
                            metrics.extend(self._parse_prometheus_response(data))
                        
        except Exception as e:
            self.logger.error(f"Error collecting Prometheus metrics: {e}")
            
        return metrics
    
    def _parse_prometheus_response(self, data: Dict) -> List[PerformanceMetric]:
        """Parse Prometheus API response into PerformanceMetric objects."""
        metrics = []
        timestamp = datetime.now()
        
        if data.get("status") == "success":
            result = data.get("data", {}).get("result", [])
            
            for item in result:
                metric_name = item.get("metric", {}).get("__name__", "unknown")
                labels = {k: v for k, v in item.get("metric", {}).items() 
                         if k != "__name__"}
                
                value_data = item.get("value", [])
                if len(value_data) >= 2:
                    try:
                        value = float(value_data[1])
                        metrics.append(PerformanceMetric(
                            timestamp=timestamp,
                            metric_name=metric_name,
                            value=value,
                            unit="",
                            labels=labels,
                            source="prometheus"
                        ))
                    except (ValueError, IndexError):
                        continue
                        
        return metrics
    
    async def _collect_application_metrics(self) -> List[PerformanceMetric]:
        """Collect application-specific performance metrics."""
        metrics = []
        timestamp = datetime.now()
        
        try:
            # Check if application endpoints are responsive
            async with aiohttp.ClientSession() as session:
                # Health check endpoint
                try:
                    start_time = time.time()
                    async with session.get("http://localhost:8000/health") as response:
                        response_time = time.time() - start_time
                        
                        metrics.append(PerformanceMetric(
                            timestamp=timestamp,
                            metric_name="health_check_response_time",
                            value=response_time,
                            unit="seconds",
                            labels={"endpoint": "health"},
                            source="application"
                        ))
                        
                        metrics.append(PerformanceMetric(
                            timestamp=timestamp,
                            metric_name="health_check_status",
                            value=1 if response.status == 200 else 0,
                            unit="boolean",
                            labels={"endpoint": "health"},
                            source="application"
                        ))
                except Exception:
                    metrics.append(PerformanceMetric(
                        timestamp=timestamp,
                        metric_name="health_check_status",
                        value=0,
                        unit="boolean",
                        labels={"endpoint": "health"},
                        source="application"
                    ))
                
        except Exception as e:
            self.logger.error(f"Error collecting application metrics: {e}")
            
        return metrics
    
    def analyze_performance(self, window_minutes: int = 60) -> Dict[str, Any]:
        """Analyze performance metrics and identify issues."""
        cutoff_time = datetime.now() - timedelta(minutes=window_minutes)
        recent_metrics = [
            m for m in self.metrics_history if m.timestamp > cutoff_time
        ]
        
        analysis = {
            "timestamp": datetime.now().isoformat(),
            "analysis_window_minutes": window_minutes,
            "total_metrics": len(recent_metrics),
            "threshold_violations": [],
            "performance_trends": {},
            "bottlenecks": [],
            "recommendations": []
        }
        
        # Check threshold violations
        violations = self._check_threshold_violations(recent_metrics)
        analysis["threshold_violations"] = violations
        
        # Analyze trends
        trends = self._analyze_trends(recent_metrics)
        analysis["performance_trends"] = trends
        
        # Detect bottlenecks
        bottlenecks = self._detect_bottlenecks(recent_metrics)
        analysis["bottlenecks"] = bottlenecks
        
        # Generate recommendations
        recommendations = self._generate_recommendations(violations, bottlenecks, trends)
        analysis["recommendations"] = [asdict(r) for r in recommendations]
        
        return analysis
    
    def _check_threshold_violations(self, metrics: List[PerformanceMetric]) -> List[Dict]:
        """Check for threshold violations."""
        violations = []
        
        # Group metrics by name
        metrics_by_name = {}
        for metric in metrics:
            if metric.metric_name not in metrics_by_name:
                metrics_by_name[metric.metric_name] = []
            metrics_by_name[metric.metric_name].append(metric)
        
        # Check each threshold
        for threshold in self.thresholds:
            if threshold.metric_name in metrics_by_name:
                metric_values = metrics_by_name[threshold.metric_name]
                
                # Get recent values within the threshold window
                window_cutoff = datetime.now() - timedelta(minutes=threshold.window_minutes)
                recent_values = [
                    m.value for m in metric_values if m.timestamp > window_cutoff
                ]
                
                if recent_values:
                    avg_value = np.mean(recent_values)
                    max_value = np.max(recent_values)
                    
                    violation_level = None
                    if threshold.direction == "above":
                        if avg_value > threshold.critical_threshold:
                            violation_level = "critical"
                        elif avg_value > threshold.warning_threshold:
                            violation_level = "warning"
                    else:  # below
                        if avg_value < threshold.critical_threshold:
                            violation_level = "critical"
                        elif avg_value < threshold.warning_threshold:
                            violation_level = "warning"
                    
                    if violation_level:
                        violations.append({
                            "metric_name": threshold.metric_name,
                            "level": violation_level,
                            "current_value": avg_value,
                            "threshold": (threshold.critical_threshold 
                                        if violation_level == "critical" 
                                        else threshold.warning_threshold),
                            "max_value": max_value,
                            "window_minutes": threshold.window_minutes,
                            "sample_count": len(recent_values)
                        })
        
        return violations
    
    def _analyze_trends(self, metrics: List[PerformanceMetric]) -> Dict[str, Any]:
        """Analyze performance trends over time."""
        trends = {}
        
        # Group metrics by name
        metrics_by_name = {}
        for metric in metrics:
            if metric.metric_name not in metrics_by_name:
                metrics_by_name[metric.metric_name] = []
            metrics_by_name[metric.metric_name].append(metric)
        
        for metric_name, metric_list in metrics_by_name.items():
            if len(metric_list) < 3:  # Need at least 3 points for trend analysis
                continue
                
            # Sort by timestamp
            sorted_metrics = sorted(metric_list, key=lambda x: x.timestamp)
            
            # Extract values and timestamps
            values = [m.value for m in sorted_metrics]
            timestamps = [m.timestamp.timestamp() for m in sorted_metrics]
            
            # Calculate trend
            if len(values) >= 2:
                # Simple linear regression
                correlation = np.corrcoef(timestamps, values)[0, 1]
                slope = np.polyfit(timestamps, values, 1)[0]
                
                trend_direction = "stable"
                if abs(correlation) > 0.7:  # Strong correlation
                    if slope > 0:
                        trend_direction = "increasing"
                    else:
                        trend_direction = "decreasing"
                
                trends[metric_name] = {
                    "direction": trend_direction,
                    "correlation": correlation,
                    "slope": slope,
                    "current_value": values[-1],
                    "min_value": min(values),
                    "max_value": max(values),
                    "sample_count": len(values)
                }
        
        return trends
    
    def _detect_bottlenecks(self, metrics: List[PerformanceMetric]) -> List[Dict]:
        """Detect performance bottlenecks."""
        bottlenecks = []
        
        # Define bottleneck detection rules
        bottleneck_rules = [
            {
                "name": "High CPU with Low Throughput",
                "condition": lambda m: (
                    self._get_metric_avg(m, "cpu_usage_percent") > 80 and
                    self._get_metric_avg(m, "http_requests_per_second", default=0) < 50
                ),
                "description": "High CPU usage combined with low request throughput indicates CPU bottleneck"
            },
            {
                "name": "High Memory Usage",
                "condition": lambda m: self._get_metric_avg(m, "memory_usage_percent") > 90,
                "description": "Memory usage is critically high, may cause OOM issues"
            },
            {
                "name": "Slow Database Queries",
                "condition": lambda m: self._get_metric_avg(m, "db_query_duration_ms", default=0) > 2000,
                "description": "Database queries are taking too long, impacting overall performance"
            },
            {
                "name": "AI Agent Performance Degradation",
                "condition": lambda m: self._get_metric_avg(m, "agent_response_time_seconds", default=0) > 45,
                "description": "AI agents are responding slowly, may need optimization or scaling"
            },
            {
                "name": "High Connection Pool Utilization",
                "condition": lambda m: self._get_metric_avg(m, "db_connection_pool_utilization", default=0) > 0.9,
                "description": "Database connection pool is nearly exhausted"
            }
        ]
        
        for rule in bottleneck_rules:
            try:
                if rule["condition"](metrics):
                    bottlenecks.append({
                        "name": rule["name"],
                        "description": rule["description"],
                        "severity": "high",
                        "detected_at": datetime.now().isoformat()
                    })
            except Exception as e:
                self.logger.error(f"Error evaluating bottleneck rule {rule['name']}: {e}")
        
        return bottlenecks
    
    def _get_metric_avg(self, metrics: List[PerformanceMetric], metric_name: str, default: float = 0) -> float:
        """Get average value for a specific metric."""
        values = [m.value for m in metrics if m.metric_name == metric_name]
        return np.mean(values) if values else default
    
    def _generate_recommendations(self, violations: List[Dict], bottlenecks: List[Dict], trends: Dict) -> List[OptimizationRecommendation]:
        """Generate optimization recommendations based on analysis."""
        recommendations = []
        
        # CPU-related recommendations
        if any(v["metric_name"] == "cpu_usage_percent" for v in violations):
            recommendations.append(OptimizationRecommendation(
                component="System",
                issue="High CPU usage detected",
                recommendation="Consider horizontal scaling or CPU optimization",
                impact="high",
                effort="medium",
                estimated_improvement="30-50% performance improvement",
                implementation_guide="Add more worker processes, optimize CPU-intensive operations, consider async processing"
            ))
        
        # Memory-related recommendations
        if any(v["metric_name"] == "memory_usage_percent" for v in violations):
            recommendations.append(OptimizationRecommendation(
                component="System",
                issue="High memory usage detected",
                recommendation="Implement memory optimization and monitoring",
                impact="high",
                effort="medium",
                estimated_improvement="20-40% memory reduction",
                implementation_guide="Add memory profiling, implement caching strategies, optimize data structures"
            ))
        
        # Database-related recommendations
        if any("db_" in v["metric_name"] for v in violations):
            recommendations.append(OptimizationRecommendation(
                component="Database",
                issue="Database performance issues detected",
                recommendation="Optimize database queries and connection management",
                impact="high",
                effort="high",
                estimated_improvement="50-80% query performance improvement",
                implementation_guide="Add query optimization, implement connection pooling, add database indices"
            ))
        
        # AI/ML-related recommendations
        if any("agent_" in v["metric_name"] or "token_" in v["metric_name"] for v in violations):
            recommendations.append(OptimizationRecommendation(
                component="AI/ML",
                issue="AI agent performance degradation",
                recommendation="Optimize AI model usage and prompt engineering",
                impact="medium",
                effort="medium",
                estimated_improvement="25-50% response time improvement",
                implementation_guide="Implement prompt caching, optimize model selection, add response streaming"
            ))
        
        # Frontend performance recommendations
        if any(v["metric_name"].endswith("_ms") for v in violations):
            recommendations.append(OptimizationRecommendation(
                component="Frontend",
                issue="Frontend performance issues detected",
                recommendation="Implement frontend optimizations",
                impact="medium",
                effort="low",
                estimated_improvement="20-40% load time improvement",
                implementation_guide="Add code splitting, optimize images, implement service worker caching"
            ))
        
        return recommendations
    
    async def generate_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance analysis report."""
        # Collect fresh metrics
        await self.collect_metrics()
        
        # Perform analysis
        analysis = self.analyze_performance()
        
        # Add additional context
        report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "analyzer_version": "1.0.0",
                "config": self.config
            },
            "executive_summary": self._generate_executive_summary(analysis),
            "detailed_analysis": analysis,
            "historical_trends": self._generate_historical_trends(),
            "action_items": self._prioritize_action_items(analysis)
        }
        
        # Save report
        report_path = Path(self.config["optimization_report_path"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Performance analysis report saved to {report_path}")
        
        return report
    
    def _generate_executive_summary(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary of performance analysis."""
        violations = analysis.get("threshold_violations", [])
        bottlenecks = analysis.get("bottlenecks", [])
        recommendations = analysis.get("recommendations", [])
        
        critical_issues = len([v for v in violations if v.get("level") == "critical"])
        warning_issues = len([v for v in violations if v.get("level") == "warning"])
        
        # Calculate overall health score (0-100)
        health_score = max(0, 100 - (critical_issues * 20) - (warning_issues * 10) - (len(bottlenecks) * 15))
        
        return {
            "overall_health_score": health_score,
            "health_status": (
                "excellent" if health_score >= 90 else
                "good" if health_score >= 70 else
                "fair" if health_score >= 50 else
                "poor"
            ),
            "critical_issues": critical_issues,
            "warning_issues": warning_issues,
            "bottlenecks_detected": len(bottlenecks),
            "recommendations_count": len(recommendations),
            "immediate_actions_needed": critical_issues > 0 or len(bottlenecks) > 0,
            "summary": f"System health score: {health_score}/100. " +
                      f"{critical_issues} critical issues, {warning_issues} warnings detected."
        }
    
    def _generate_historical_trends(self) -> Dict[str, Any]:
        """Generate historical performance trends."""
        # This would typically connect to a time-series database
        # For now, return basic trend information
        return {
            "data_availability": f"Last {self.config['analysis_window_hours']} hours",
            "metrics_collected": len(self.metrics_history),
            "trend_note": "Historical trend analysis requires longer data collection period"
        }
    
    def _prioritize_action_items(self, analysis: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Prioritize action items based on impact and urgency."""
        action_items = []
        
        violations = analysis.get("threshold_violations", [])
        bottlenecks = analysis.get("bottlenecks", [])
        recommendations = analysis.get("recommendations", [])
        
        # Critical violations first
        for violation in violations:
            if violation.get("level") == "critical":
                action_items.append({
                    "priority": "P0",
                    "type": "threshold_violation",
                    "title": f"Critical: {violation['metric_name']} threshold exceeded",
                    "description": f"Current value: {violation['current_value']}, Threshold: {violation['threshold']}",
                    "urgency": "immediate"
                })
        
        # High-impact bottlenecks
        for bottleneck in bottlenecks:
            action_items.append({
                "priority": "P1",
                "type": "bottleneck",
                "title": f"Bottleneck: {bottleneck['name']}",
                "description": bottleneck['description'],
                "urgency": "high"
            })
        
        # High-impact recommendations
        for rec in recommendations:
            if rec.get("impact") == "high":
                action_items.append({
                    "priority": "P2",
                    "type": "optimization",
                    "title": f"Optimize {rec['component']}: {rec['issue']}",
                    "description": rec['recommendation'],
                    "urgency": "medium"
                })
        
        return action_items


async def main():
    """Main function for standalone execution."""
    analyzer = PerformanceAnalyzer()
    
    # Generate and display report
    report = await analyzer.generate_report()
    
    print("Performance Analysis Report Generated")
    print("=" * 50)
    
    summary = report["executive_summary"]
    print(f"Overall Health Score: {summary['overall_health_score']}/100 ({summary['health_status']})")
    print(f"Critical Issues: {summary['critical_issues']}")
    print(f"Warning Issues: {summary['warning_issues']}")
    print(f"Bottlenecks: {summary['bottlenecks_detected']}")
    print(f"Recommendations: {summary['recommendations_count']}")
    
    if summary["immediate_actions_needed"]:
        print("\n⚠️  IMMEDIATE ACTION REQUIRED")
        action_items = report["action_items"]
        for item in action_items[:3]:  # Show top 3 priority items
            print(f"  {item['priority']}: {item['title']}")


if __name__ == "__main__":
    asyncio.run(main())