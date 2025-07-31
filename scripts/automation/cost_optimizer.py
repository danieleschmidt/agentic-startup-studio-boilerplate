#!/usr/bin/env python3
"""
Advanced Cost Optimization and Financial Analytics System

This module provides comprehensive cost optimization capabilities for cloud infrastructure,
including automated cost analysis, resource right-sizing recommendations, and budget
management for the Agentic Startup Studio Boilerplate.

Features:
- Multi-cloud cost analysis and optimization
- Resource utilization monitoring and right-sizing
- Automated cost alerting and budget management
- Waste detection and elimination recommendations
- Cost forecasting and trend analysis
- ROI analysis for infrastructure investments
"""

import json
import logging
import asyncio
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
import numpy as np
import pandas as pd
import boto3
from azure.identity import DefaultAzureCredential
from azure.mgmt.costmanagement import CostManagementClient
from google.cloud import billing_v1
from kubernetes import client, config


@dataclass
class CostMetric:
    """Cost metric data structure."""
    timestamp: datetime
    service: str
    resource_id: str
    cost: float
    currency: str
    region: str
    environment: str
    tags: Dict[str, str]
    utilization: Optional[float] = None


@dataclass
class OptimizationRecommendation:
    """Cost optimization recommendation structure."""
    id: str
    resource_type: str
    resource_id: str
    current_cost_monthly: float
    optimized_cost_monthly: float
    potential_savings_monthly: float
    potential_savings_annual: float
    recommendation_type: str  # 'right_size', 'terminate', 'reserved_instance', 'spot_instance'
    confidence_level: str     # 'high', 'medium', 'low'
    implementation_effort: str # 'low', 'medium', 'high'
    risk_level: str           # 'low', 'medium', 'high'
    description: str
    implementation_steps: List[str]
    business_impact: str


@dataclass
class BudgetAlert:
    """Budget alert configuration."""
    name: str
    budget_amount: float
    currency: str
    threshold_percentage: float
    current_spend: float
    forecasted_spend: float
    alert_level: str  # 'info', 'warning', 'critical'
    recipients: List[str]


class CostOptimizer:
    """Advanced cost optimization and financial analytics system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the cost optimizer."""
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.cost_metrics: List[CostMetric] = []
        self.recommendations: List[OptimizationRecommendation] = []
        self.budget_alerts: List[BudgetAlert] = []
        
        # Initialize cloud clients
        self._initialize_cloud_clients()
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            "cloud_providers": {
                "aws": {
                    "enabled": True,
                    "regions": ["us-east-1", "us-west-2", "eu-west-1"],
                    "cost_allocation_tags": ["Environment", "Project", "Owner", "CostCenter"]
                },
                "azure": {
                    "enabled": False,
                    "subscription_id": None,
                    "resource_groups": []
                },
                "gcp": {
                    "enabled": False,
                    "project_id": None,
                    "billing_account": None
                }
            },
            "optimization_settings": {
                "utilization_threshold": 0.3,  # 30% utilization threshold
                "cost_threshold": 10.0,        # $10 minimum monthly cost for recommendations
                "forecast_days": 30,           # 30-day forecast period
                "reserved_instance_term": 12,  # 12-month RI term
                "spot_instance_savings_threshold": 0.5  # 50% savings threshold for spot recommendations
            },
            "budget_management": {
                "default_currency": "USD",
                "alert_thresholds": [50, 80, 95, 100],  # Percentage thresholds
                "notification_email": "finance@company.com"
            },
            "report_settings": {
                "output_path": "reports/cost_optimization.json",
                "dashboard_path": "reports/cost_dashboard.html",
                "retention_days": 90
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                user_config = json.load(f)
                default_config.update(user_config)
                
        return default_config
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger("cost_optimizer")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _initialize_cloud_clients(self):
        """Initialize cloud provider clients."""
        self.cloud_clients = {}
        
        # AWS Client
        if self.config["cloud_providers"]["aws"]["enabled"]:
            try:
                self.cloud_clients["aws"] = {
                    "ce": boto3.client('ce'),  # Cost Explorer
                    "ec2": boto3.client('ec2'),
                    "rds": boto3.client('rds'),
                    "cloudwatch": boto3.client('cloudwatch'),
                    "pricing": boto3.client('pricing', region_name='us-east-1')
                }
                self.logger.info("AWS clients initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize AWS clients: {e}")
        
        # Azure Client
        if self.config["cloud_providers"]["azure"]["enabled"]:
            try:
                credential = DefaultAzureCredential()
                subscription_id = self.config["cloud_providers"]["azure"]["subscription_id"]
                self.cloud_clients["azure"] = {
                    "cost_management": CostManagementClient(credential, subscription_id)
                }
                self.logger.info("Azure clients initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize Azure clients: {e}")
        
        # GCP Client
        if self.config["cloud_providers"]["gcp"]["enabled"]:
            try:
                self.cloud_clients["gcp"] = {
                    "billing": billing_v1.CloudBillingClient()
                }
                self.logger.info("GCP clients initialized successfully")
            except Exception as e:
                self.logger.error(f"Failed to initialize GCP clients: {e}")
    
    async def analyze_costs(self, days_back: int = 30) -> Dict[str, Any]:
        """Perform comprehensive cost analysis across all cloud providers."""
        self.logger.info(f"Starting cost analysis for the last {days_back} days")
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days_back)
        
        # Clear previous metrics
        self.cost_metrics.clear()
        
        # Collect cost data from all enabled providers
        tasks = []
        
        if "aws" in self.cloud_clients:
            tasks.append(self._analyze_aws_costs(start_date, end_date))
        
        if "azure" in self.cloud_clients:
            tasks.append(self._analyze_azure_costs(start_date, end_date))
        
        if "gcp" in self.cloud_clients:
            tasks.append(self._analyze_gcp_costs(start_date, end_date))
        
        # Collect Kubernetes costs if available
        tasks.append(self._analyze_kubernetes_costs(start_date, end_date))
        
        # Execute all analysis tasks
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Analyze collected data
        analysis_results = self._analyze_cost_data()
        
        # Generate recommendations
        await self._generate_optimization_recommendations()
        
        # Update budget alerts
        self._update_budget_alerts()
        
        return analysis_results
    
    async def _analyze_aws_costs(self, start_date: datetime, end_date: datetime):
        """Analyze AWS costs and resource utilization."""
        try:
            ce_client = self.cloud_clients["aws"]["ce"]
            
            # Get cost and usage data
            response = ce_client.get_cost_and_usage(
                TimePeriod={
                    'Start': start_date.strftime('%Y-%m-%d'),
                    'End': end_date.strftime('%Y-%m-%d')
                },
                Granularity='DAILY',
                Metrics=['BlendedCost', 'UsageQuantity'],
                GroupBy=[
                    {'Type': 'DIMENSION', 'Key': 'SERVICE'},
                    {'Type': 'DIMENSION', 'Key': 'REGION'}
                ]
            )
            
            # Process cost data
            for result in response['ResultsByTime']:
                date = datetime.strptime(result['TimePeriod']['Start'], '%Y-%m-%d')
                
                for group in result['Groups']:
                    service = group['Keys'][0]
                    region = group['Keys'][1]
                    
                    cost = float(group['Metrics']['BlendedCost']['Amount'])
                    currency = group['Metrics']['BlendedCost']['Unit']
                    
                    self.cost_metrics.append(CostMetric(
                        timestamp=date,
                        service=service,
                        resource_id=f"aws-{service}-{region}",
                        cost=cost,
                        currency=currency,
                        region=region,
                        environment="production",  # Would need tag-based detection
                        tags={"provider": "aws", "service": service}
                    ))
            
            # Get resource utilization data
            await self._collect_aws_utilization_metrics()
            
        except Exception as e:
            self.logger.error(f"Error analyzing AWS costs: {e}")
    
    async def _collect_aws_utilization_metrics(self):
        """Collect AWS resource utilization metrics."""
        try:
            cloudwatch = self.cloud_clients["aws"]["cloudwatch"]
            ec2 = self.cloud_clients["aws"]["ec2"]
            
            # Get EC2 instances
            instances_response = ec2.describe_instances()
            
            for reservation in instances_response['Reservations']:
                for instance in reservation['Instances']:
                    if instance['State']['Name'] != 'running':
                        continue
                    
                    instance_id = instance['InstanceId']
                    instance_type = instance['InstanceType']
                    
                    # Get CPU utilization
                    cpu_response = cloudwatch.get_metric_statistics(
                        Namespace='AWS/EC2',
                        MetricName='CPUUtilization',
                        Dimensions=[
                            {'Name': 'InstanceId', 'Value': instance_id}
                        ],
                        StartTime=datetime.now() - timedelta(days=7),
                        EndTime=datetime.now(),
                        Period=3600,  # 1 hour
                        Statistics=['Average']
                    )
                    
                    if cpu_response['Datapoints']:
                        avg_cpu = np.mean([point['Average'] for point in cpu_response['Datapoints']])
                        
                        # Update existing cost metrics with utilization data
                        for metric in self.cost_metrics:
                            if metric.resource_id == f"aws-EC2-Instance-{instance_id}":
                                metric.utilization = avg_cpu / 100.0
                                break
        
        except Exception as e:
            self.logger.error(f"Error collecting AWS utilization metrics: {e}")
    
    async def _analyze_azure_costs(self, start_date: datetime, end_date: datetime):
        """Analyze Azure costs and resource utilization."""
        # Implementation would go here for Azure cost analysis
        self.logger.info("Azure cost analysis not implemented yet")
        pass
    
    async def _analyze_gcp_costs(self, start_date: datetime, end_date: datetime):
        """Analyze GCP costs and resource utilization."""
        # Implementation would go here for GCP cost analysis
        self.logger.info("GCP cost analysis not implemented yet")
        pass
    
    async def _analyze_kubernetes_costs(self, start_date: datetime, end_date: datetime):
        """Analyze Kubernetes resource costs and utilization."""
        try:
            # Try to load Kubernetes config
            try:
                config.load_incluster_config()
            except:
                try:
                    config.load_kube_config()
                except:
                    self.logger.warning("Kubernetes config not available, skipping K8s cost analysis")
                    return
            
            v1 = client.CoreV1Api()
            apps_v1 = client.AppsV1Api()
            
            # Get node information
            nodes = v1.list_node()
            
            for node in nodes.items:
                node_name = node.metadata.name
                
                # Get node resource capacity
                capacity = node.status.allocatable
                cpu_capacity = float(capacity.get('cpu', '0').rstrip('m')) / 1000
                memory_capacity = self._parse_memory(capacity.get('memory', '0'))
                
                # Get pods on this node
                pods = v1.list_pod_for_all_namespaces(field_selector=f"spec.nodeName={node_name}")
                
                total_cpu_requests = 0
                total_memory_requests = 0
                
                for pod in pods.items:
                    if pod.spec.containers:
                        for container in pod.spec.containers:
                            if container.resources and container.resources.requests:
                                requests = container.resources.requests
                                cpu_request = requests.get('cpu', '0')
                                memory_request = requests.get('memory', '0')
                                
                                total_cpu_requests += self._parse_cpu(cpu_request)
                                total_memory_requests += self._parse_memory(memory_request)
                
                # Calculate utilization
                cpu_utilization = total_cpu_requests / max(cpu_capacity, 0.001)
                memory_utilization = total_memory_requests / max(memory_capacity, 1)
                
                # Estimate cost (this would need integration with cloud provider pricing)
                estimated_hourly_cost = self._estimate_node_cost(node)
                estimated_daily_cost = estimated_hourly_cost * 24
                
                self.cost_metrics.append(CostMetric(
                    timestamp=datetime.now(),
                    service="Kubernetes",
                    resource_id=f"k8s-node-{node_name}",
                    cost=estimated_daily_cost,
                    currency="USD",
                    region="unknown",
                    environment="production",
                    tags={
                        "provider": "kubernetes",
                        "node_name": node_name,
                        "instance_type": node.metadata.labels.get("node.kubernetes.io/instance-type", "unknown")
                    },
                    utilization=max(cpu_utilization, memory_utilization)
                ))
        
        except Exception as e:
            self.logger.error(f"Error analyzing Kubernetes costs: {e}")
    
    def _parse_cpu(self, cpu_str: str) -> float:
        """Parse Kubernetes CPU resource string."""
        if cpu_str.endswith('m'):
            return float(cpu_str[:-1]) / 1000
        return float(cpu_str)
    
    def _parse_memory(self, memory_str: str) -> float:
        """Parse Kubernetes memory resource string to bytes."""
        units = {'Ki': 1024, 'Mi': 1024**2, 'Gi': 1024**3, 'Ti': 1024**4}
        
        for unit, multiplier in units.items():
            if memory_str.endswith(unit):
                return float(memory_str[:-len(unit)]) * multiplier
        
        return float(memory_str)
    
    def _estimate_node_cost(self, node) -> float:
        """Estimate hourly cost of a Kubernetes node."""
        # This is a simplified estimation - in practice, you'd integrate with cloud provider pricing APIs
        instance_type = node.metadata.labels.get("node.kubernetes.io/instance-type", "unknown")
        
        # Basic pricing estimation (would need real pricing data)
        pricing_map = {
            "t3.micro": 0.0104,
            "t3.small": 0.0208,
            "t3.medium": 0.0416,
            "t3.large": 0.0832,
            "t3.xlarge": 0.1664,
            "t3.2xlarge": 0.3328,
            "m5.large": 0.096,
            "m5.xlarge": 0.192,
            "m5.2xlarge": 0.384
        }
        
        return pricing_map.get(instance_type, 0.1)  # Default to $0.1/hour
    
    def _analyze_cost_data(self) -> Dict[str, Any]:
        """Analyze collected cost data and generate insights."""
        if not self.cost_metrics:
            return {"error": "No cost data available for analysis"}
        
        # Convert to pandas DataFrame for easier analysis
        df = pd.DataFrame([asdict(metric) for metric in self.cost_metrics])
        
        # Calculate key metrics
        total_cost = df['cost'].sum()
        daily_average = df.groupby('timestamp')['cost'].sum().mean()
        monthly_projection = daily_average * 30
        
        # Cost by service
        cost_by_service = df.groupby('service')['cost'].sum().sort_values(ascending=False)
        
        # Cost by region
        cost_by_region = df.groupby('region')['cost'].sum().sort_values(ascending=False)
        
        # Utilization analysis
        utilization_df = df[df['utilization'].notna()]
        low_utilization_resources = utilization_df[
            utilization_df['utilization'] < self.config["optimization_settings"]["utilization_threshold"]
        ]
        
        # Cost trends
        daily_costs = df.groupby('timestamp')['cost'].sum()
        cost_trend = "increasing" if daily_costs.tail(7).mean() > daily_costs.head(7).mean() else "decreasing"
        
        return {
            "analysis_date": datetime.now().isoformat(),
            "total_cost_analyzed": total_cost,
            "daily_average_cost": daily_average,
            "monthly_projection": monthly_projection,
            "cost_trend": cost_trend,
            "cost_by_service": cost_by_service.to_dict(),
            "cost_by_region": cost_by_region.to_dict(),
            "low_utilization_resources": len(low_utilization_resources),
            "average_utilization": utilization_df['utilization'].mean() if len(utilization_df) > 0 else None,
            "total_metrics_collected": len(self.cost_metrics)
        }
    
    async def _generate_optimization_recommendations(self):
        """Generate cost optimization recommendations based on analysis."""
        self.recommendations.clear()
        
        # Right-sizing recommendations
        await self._generate_rightsizing_recommendations()
        
        # Reserved instance recommendations
        await self._generate_reserved_instance_recommendations()
        
        # Spot instance recommendations
        await self._generate_spot_instance_recommendations()
        
        # Resource termination recommendations
        await self._generate_termination_recommendations()
        
        # Storage optimization recommendations
        await self._generate_storage_optimization_recommendations()
    
    async def _generate_rightsizing_recommendations(self):
        """Generate right-sizing recommendations for underutilized resources."""
        utilization_threshold = self.config["optimization_settings"]["utilization_threshold"]
        
        for metric in self.cost_metrics:
            if metric.utilization is None or metric.utilization >= utilization_threshold:
                continue
            
            if metric.cost < self.config["optimization_settings"]["cost_threshold"]:
                continue
            
            # Calculate potential savings (simplified calculation)
            current_monthly_cost = metric.cost * 30
            
            # Estimate smaller instance cost (assume 50% cost reduction for right-sizing)
            optimized_monthly_cost = current_monthly_cost * 0.5
            monthly_savings = current_monthly_cost - optimized_monthly_cost
            annual_savings = monthly_savings * 12
            
            recommendation = OptimizationRecommendation(
                id=f"rightsize_{metric.resource_id}",
                resource_type=metric.service,
                resource_id=metric.resource_id,
                current_cost_monthly=current_monthly_cost,
                optimized_cost_monthly=optimized_monthly_cost,
                potential_savings_monthly=monthly_savings,
                potential_savings_annual=annual_savings,
                recommendation_type="right_size",
                confidence_level="high" if metric.utilization < 0.1 else "medium",
                implementation_effort="low",
                risk_level="low",
                description=f"Resource is underutilized ({metric.utilization:.1%} utilization). Consider downsizing to reduce costs.",
                implementation_steps=[
                    "Create snapshot/backup of the resource",
                    "Stop the resource during maintenance window",
                    "Resize to smaller instance type",
                    "Start the resource and monitor performance",
                    "Validate application functionality"
                ],
                business_impact=f"Potential annual savings of ${annual_savings:.2f} with minimal performance impact"
            )
            
            self.recommendations.append(recommendation)
    
    async def _generate_reserved_instance_recommendations(self):
        """Generate reserved instance recommendations for stable workloads."""
        # Group by service and instance type
        service_costs = {}
        
        for metric in self.cost_metrics:
            service_key = f"{metric.service}_{metric.tags.get('instance_type', 'unknown')}"
            if service_key not in service_costs:
                service_costs[service_key] = []
            service_costs[service_key].append(metric.cost)
        
        for service_key, costs in service_costs.items():
            monthly_cost = sum(costs) * 30 / len(costs)  # Extrapolate to monthly
            
            if monthly_cost < 50:  # Skip low-cost resources
                continue
            
            # Assume 30% savings with reserved instances
            ri_monthly_cost = monthly_cost * 0.7
            monthly_savings = monthly_cost - ri_monthly_cost
            annual_savings = monthly_savings * 12
            
            recommendation = OptimizationRecommendation(
                id=f"reserved_instance_{service_key}",
                resource_type=service_key.split('_')[0],
                resource_id=service_key,
                current_cost_monthly=monthly_cost,
                optimized_cost_monthly=ri_monthly_cost,
                potential_savings_monthly=monthly_savings,
                potential_savings_annual=annual_savings,
                recommendation_type="reserved_instance",
                confidence_level="high",
                implementation_effort="low",
                risk_level="low",
                description=f"Purchase reserved instances for stable workload to achieve ~30% cost savings",
                implementation_steps=[
                    "Analyze usage patterns over the last 3-6 months",
                    "Choose appropriate RI term (1 or 3 years)",
                    "Purchase reserved instances through cloud provider console",
                    "Monitor RI utilization and coverage"
                ],
                business_impact=f"Guaranteed annual savings of ${annual_savings:.2f} for predictable workloads"
            )
            
            self.recommendations.append(recommendation)
    
    async def _generate_spot_instance_recommendations(self):
        """Generate spot instance recommendations for fault-tolerant workloads."""
        # This would identify workloads suitable for spot instances
        # For now, we'll create a generic recommendation
        
        batch_workloads = [metric for metric in self.cost_metrics 
                          if 'batch' in metric.tags.get('workload_type', '').lower()]
        
        for metric in batch_workloads:
            monthly_cost = metric.cost * 30
            
            # Assume 70% savings with spot instances
            spot_monthly_cost = monthly_cost * 0.3
            monthly_savings = monthly_cost - spot_monthly_cost
            annual_savings = monthly_savings * 12
            
            recommendation = OptimizationRecommendation(
                id=f"spot_instance_{metric.resource_id}",
                resource_type=metric.service,
                resource_id=metric.resource_id,
                current_cost_monthly=monthly_cost,
                optimized_cost_monthly=spot_monthly_cost,
                potential_savings_monthly=monthly_savings,
                potential_savings_annual=annual_savings,
                recommendation_type="spot_instance",
                confidence_level="medium",
                implementation_effort="medium",
                risk_level="medium",
                description="Convert fault-tolerant workload to spot instances for significant cost savings",
                implementation_steps=[
                    "Identify fault-tolerant and flexible workloads",
                    "Implement spot instance request strategy",
                    "Add interruption handling to applications",
                    "Monitor spot instance availability and pricing",
                    "Implement auto-scaling with mixed instance types"
                ],
                business_impact=f"Potential annual savings of ${annual_savings:.2f} for suitable workloads"
            )
            
            self.recommendations.append(recommendation)
    
    async def _generate_termination_recommendations(self):
        """Generate recommendations for terminating unused resources."""
        # Identify resources with zero or very low utilization
        zero_utilization_resources = [
            metric for metric in self.cost_metrics
            if metric.utilization is not None and metric.utilization < 0.01
        ]
        
        for metric in zero_utilization_resources:
            monthly_cost = metric.cost * 30
            
            recommendation = OptimizationRecommendation(
                id=f"terminate_{metric.resource_id}",
                resource_type=metric.service,
                resource_id=metric.resource_id,
                current_cost_monthly=monthly_cost,
                optimized_cost_monthly=0.0,
                potential_savings_monthly=monthly_cost,
                potential_savings_annual=monthly_cost * 12,
                recommendation_type="terminate",
                confidence_level="high",
                implementation_effort="low",
                risk_level="medium",
                description=f"Resource shows no utilization ({metric.utilization:.3%}). Consider termination if truly unused.",
                implementation_steps=[
                    "Verify resource is not needed for critical operations",
                    "Check for any dependencies or scheduled tasks",
                    "Create backup if data needs to be preserved",
                    "Terminate the resource",
                    "Monitor for any unexpected impacts"
                ],
                business_impact=f"Complete elimination of ${monthly_cost * 12:.2f} annual cost"
            )
            
            self.recommendations.append(recommendation)
    
    async def _generate_storage_optimization_recommendations(self):
        """Generate storage optimization recommendations."""
        # This would analyze storage costs and recommend lifecycle policies, compression, etc.
        # For now, we'll create a generic storage optimization recommendation
        
        storage_metrics = [metric for metric in self.cost_metrics 
                          if 'storage' in metric.service.lower() or 'ebs' in metric.service.lower()]
        
        total_storage_cost = sum(metric.cost for metric in storage_metrics) * 30
        
        if total_storage_cost > 100:  # Only recommend if storage cost is significant
            # Assume 25% savings through optimization
            optimized_cost = total_storage_cost * 0.75
            monthly_savings = total_storage_cost - optimized_cost
            annual_savings = monthly_savings * 12
            
            recommendation = OptimizationRecommendation(
                id="storage_optimization",
                resource_type="Storage",
                resource_id="all_storage_resources",
                current_cost_monthly=total_storage_cost,
                optimized_cost_monthly=optimized_cost,
                potential_savings_monthly=monthly_savings,
                potential_savings_annual=annual_savings,
                recommendation_type="storage_optimization",
                confidence_level="medium",
                implementation_effort="medium",
                risk_level="low",
                description="Implement storage lifecycle policies and optimization strategies",
                implementation_steps=[
                    "Analyze storage access patterns",
                    "Implement lifecycle policies for infrequent access data",
                    "Enable compression where applicable",
                    "Delete old snapshots and unused volumes",
                    "Consider storage class transitions"
                ],
                business_impact=f"Potential annual savings of ${annual_savings:.2f} through storage optimization"
            )
            
            self.recommendations.append(recommendation)
    
    def _update_budget_alerts(self):
        """Update budget alerts based on current spend and forecasts."""
        self.budget_alerts.clear()
        
        # Calculate total monthly spend
        monthly_spend = sum(metric.cost for metric in self.cost_metrics) * 30 / len(
            set(metric.timestamp.date() for metric in self.cost_metrics)
        ) if self.cost_metrics else 0
        
        # Create budget alerts for different cost centers (simplified)
        budgets = [
            {"name": "Total Infrastructure", "budget": 5000, "current": monthly_spend},
            {"name": "Development Environment", "budget": 1000, "current": monthly_spend * 0.2},
            {"name": "Production Environment", "budget": 3000, "current": monthly_spend * 0.6},
            {"name": "Staging Environment", "budget": 500, "current": monthly_spend * 0.1}
        ]
        
        for budget_config in budgets:
            current_spend = budget_config["current"]
            budget_amount = budget_config["budget"]
            spend_percentage = (current_spend / budget_amount) * 100
            
            # Determine alert level
            if spend_percentage >= 100:
                alert_level = "critical"
            elif spend_percentage >= 80:
                alert_level = "warning"
            elif spend_percentage >= 50:
                alert_level = "info"
            else:
                continue  # No alert needed
            
            # Simple forecasting (assume linear trend)
            forecasted_spend = current_spend * 1.1  # 10% growth assumption
            
            alert = BudgetAlert(
                name=budget_config["name"],
                budget_amount=budget_amount,
                currency="USD",
                threshold_percentage=spend_percentage,
                current_spend=current_spend,
                forecasted_spend=forecasted_spend,
                alert_level=alert_level,
                recipients=[self.config["budget_management"]["notification_email"]]
            )
            
            self.budget_alerts.append(alert)
    
    def generate_cost_report(self) -> Dict[str, Any]:
        """Generate comprehensive cost optimization report."""
        if not self.cost_metrics:
            return {"error": "No cost data available. Run analyze_costs() first."}
        
        # Calculate summary metrics
        total_current_monthly_cost = sum(rec.current_cost_monthly for rec in self.recommendations)
        total_potential_monthly_savings = sum(rec.potential_savings_monthly for rec in self.recommendations)
        total_potential_annual_savings = sum(rec.potential_savings_annual for rec in self.recommendations)
        
        # Categorize recommendations
        recommendations_by_type = {}
        for rec in self.recommendations:
            if rec.recommendation_type not in recommendations_by_type:
                recommendations_by_type[rec.recommendation_type] = []
            recommendations_by_type[rec.recommendation_type].append(rec)
        
        # Generate executive summary
        executive_summary = {
            "total_monthly_cost": sum(metric.cost for metric in self.cost_metrics) * 30,
            "potential_monthly_savings": total_potential_monthly_savings,
            "potential_annual_savings": total_potential_annual_savings,
            "savings_percentage": (total_potential_monthly_savings / max(total_current_monthly_cost, 1)) * 100,
            "total_recommendations": len(self.recommendations),
            "high_confidence_recommendations": len([r for r in self.recommendations if r.confidence_level == "high"]),
            "low_risk_recommendations": len([r for r in self.recommendations if r.risk_level == "low"])
        }
        
        report = {
            "report_metadata": {
                "generated_at": datetime.now().isoformat(),
                "optimizer_version": "1.0.0",
                "analysis_period": "Last 30 days"
            },
            "executive_summary": executive_summary,
            "recommendations": [asdict(rec) for rec in self.recommendations],
            "recommendations_by_type": {k: [asdict(rec) for rec in v] for k, v in recommendations_by_type.items()},
            "budget_alerts": [asdict(alert) for alert in self.budget_alerts],
            "cost_metrics_summary": self._analyze_cost_data(),
            "action_plan": self._generate_action_plan()
        }
        
        # Save report
        report_path = Path(self.config["report_settings"]["output_path"])
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self.logger.info(f"Cost optimization report saved to {report_path}")
        
        return report
    
    def _generate_action_plan(self) -> Dict[str, Any]:
        """Generate prioritized action plan for cost optimization."""
        # Sort recommendations by potential savings and confidence
        sorted_recommendations = sorted(
            self.recommendations,
            key=lambda x: (x.potential_annual_savings * (1 if x.confidence_level == "high" else 0.7)),
            reverse=True
        )
        
        # Create action plan phases
        phase_1 = [rec for rec in sorted_recommendations[:5] if rec.implementation_effort == "low"]
        phase_2 = [rec for rec in sorted_recommendations if rec not in phase_1 and rec.confidence_level == "high"]
        phase_3 = [rec for rec in sorted_recommendations if rec not in phase_1 and rec not in phase_2]
        
        return {
            "phase_1_quick_wins": {
                "description": "Low-effort, high-impact optimizations",
                "timeline": "1-2 weeks",
                "recommendations": [rec.id for rec in phase_1],
                "potential_savings": sum(rec.potential_annual_savings for rec in phase_1)
            },
            "phase_2_high_confidence": {
                "description": "High-confidence optimizations requiring more effort",
                "timeline": "1-2 months",
                "recommendations": [rec.id for rec in phase_2],
                "potential_savings": sum(rec.potential_annual_savings for rec in phase_2)
            },
            "phase_3_additional_optimizations": {
                "description": "Additional optimization opportunities",
                "timeline": "3-6 months",
                "recommendations": [rec.id for rec in phase_3],
                "potential_savings": sum(rec.potential_annual_savings for rec in phase_3)
            },
            "total_optimization_potential": {
                "annual_savings": sum(rec.potential_annual_savings for rec in self.recommendations),
                "implementation_priority": "Start with Phase 1 for immediate impact"
            }
        }


async def main():
    """Main function for standalone execution."""
    optimizer = CostOptimizer()
    
    # Perform cost analysis
    print("Performing cost analysis...")
    analysis_results = await optimizer.analyze_costs(days_back=30)
    
    # Generate report
    print("Generating cost optimization report...")
    report = optimizer.generate_cost_report()
    
    # Display summary
    print("\nCost Optimization Report Summary")
    print("=" * 50)
    
    summary = report["executive_summary"]
    print(f"Total Monthly Cost: ${summary['total_monthly_cost']:.2f}")
    print(f"Potential Monthly Savings: ${summary['potential_monthly_savings']:.2f}")
    print(f"Potential Annual Savings: ${summary['potential_annual_savings']:.2f}")
    print(f"Savings Percentage: {summary['savings_percentage']:.1f}%")
    print(f"Total Recommendations: {summary['total_recommendations']}")
    
    # Show top recommendations
    if report["recommendations"]:
        print("\nTop 3 Recommendations:")
        sorted_recs = sorted(
            report["recommendations"],
            key=lambda x: x["potential_annual_savings"],
            reverse=True
        )
        
        for i, rec in enumerate(sorted_recs[:3], 1):
            print(f"{i}. {rec['description']}")
            print(f"   Annual Savings: ${rec['potential_annual_savings']:.2f}")
            print(f"   Confidence: {rec['confidence_level']}, Risk: {rec['risk_level']}")


if __name__ == "__main__":
    asyncio.run(main())