#!/usr/bin/env python3
"""
Policy-as-Code Framework for Enterprise Governance

This module provides a comprehensive Policy-as-Code implementation for the
Agentic Startup Studio Boilerplate, enabling automated policy enforcement,
compliance monitoring, and governance across infrastructure and applications.

Features:
- Infrastructure policy enforcement (OPA/Gatekeeper)
- Security policy automation
- Compliance monitoring and reporting
- Cost governance and budget enforcement
- Data governance and privacy compliance
- CI/CD pipeline policy controls
- Real-time policy violation detection
"""

import json
import logging
import yaml
import asyncio
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
import subprocess
import requests
from kubernetes import client, config
import boto3


@dataclass
class PolicyRule:
    """Policy rule definition structure."""
    id: str
    name: str
    category: str  # 'security', 'cost', 'compliance', 'operational'
    severity: str  # 'low', 'medium', 'high', 'critical'
    description: str
    rule_logic: Dict[str, Any]
    enforcement_action: str  # 'warn', 'block', 'remediate'
    exceptions: List[str]
    created_by: str
    created_at: datetime
    last_updated: datetime
    enabled: bool


@dataclass
class PolicyViolation:
    """Policy violation record structure."""
    id: str
    policy_rule_id: str
    resource_type: str
    resource_id: str
    resource_namespace: Optional[str]
    violation_details: Dict[str, Any]
    severity: str
    detected_at: datetime
    status: str  # 'open', 'acknowledged', 'resolved', 'exception_granted'
    remediation_action: Optional[str]
    remediated_at: Optional[datetime]


@dataclass
class ComplianceReport:
    """Compliance report structure."""
    report_id: str
    report_type: str  # 'daily', 'weekly', 'monthly', 'on_demand'
    generated_at: datetime
    compliance_framework: str  # 'SOC2', 'GDPR', 'HIPAA', 'PCI-DSS'
    overall_score: float
    total_policies: int
    violations_by_severity: Dict[str, int]
    compliance_by_category: Dict[str, float]
    recommendations: List[str]


class PolicyAsCodeFramework:
    """Comprehensive Policy-as-Code framework for enterprise governance."""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the Policy-as-Code framework."""
        self.config = self._load_config(config_path)
        self.logger = self._setup_logging()
        self.policy_rules: List[PolicyRule] = []
        self.violations: List[PolicyViolation] = []
        
        # Initialize clients
        self._initialize_clients()
        
        # Load policy rules
        self._load_policy_rules()
    
    def _load_config(self, config_path: Optional[str]) -> Dict[str, Any]:
        """Load configuration from file or use defaults."""
        default_config = {
            "policy_engine": {
                "provider": "opa",  # Open Policy Agent
                "opa_server_url": "http://opa-server:8181",
                "gatekeeper_enabled": True,
                "policy_directory": "policies/"
            },
            "enforcement": {
                "default_action": "warn",  # warn, block, remediate
                "auto_remediation": True,
                "exception_approval_required": True,
                "violation_retention_days": 90
            },
            "compliance_frameworks": {
                "enabled": ["SOC2", "GDPR", "CIS"],
                "reporting_schedule": "weekly",
                "alert_thresholds": {
                    "critical": 0,
                    "high": 5,
                    "medium": 20
                }
            },
            "integrations": {
                "kubernetes": {
                    "enabled": True,
                    "namespaces": ["default", "production", "staging"]
                },
                "aws": {
                    "enabled": True,
                    "regions": ["us-east-1", "us-west-2"]
                },
                "ci_cd": {
                    "enabled": True,
                    "platforms": ["github", "gitlab"]
                }
            },
            "notifications": {
                "slack_webhook": None,
                "email_recipients": ["security@company.com"],
                "pagerduty_integration_key": None
            }
        }
        
        if config_path and Path(config_path).exists():
            with open(config_path) as f:
                user_config = yaml.safe_load(f)
                default_config.update(user_config)
                
        return default_config
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        logger = logging.getLogger("policy_as_code")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            
        return logger
    
    def _initialize_clients(self):
        """Initialize various service clients."""
        self.clients = {}
        
        # Kubernetes client
        if self.config["integrations"]["kubernetes"]["enabled"]:
            try:
                config.load_incluster_config()
            except:
                try:
                    config.load_kube_config()
                except:
                    self.logger.warning("Kubernetes config not available")
            
            try:
                self.clients["k8s"] = {
                    "core": client.CoreV1Api(),
                    "apps": client.AppsV1Api(),
                    "rbac": client.RbacAuthorizationV1Api(),
                    "networking": client.NetworkingV1Api()
                }
                self.logger.info("Kubernetes client initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize Kubernetes client: {e}")
        
        # AWS client
        if self.config["integrations"]["aws"]["enabled"]:
            try:
                self.clients["aws"] = {
                    "iam": boto3.client('iam'),
                    "ec2": boto3.client('ec2'),
                    "s3": boto3.client('s3'),
                    "config": boto3.client('config'),
                    "cloudtrail": boto3.client('cloudtrail')
                }
                self.logger.info("AWS clients initialized")
            except Exception as e:
                self.logger.error(f"Failed to initialize AWS clients: {e}")
    
    def _load_policy_rules(self):
        """Load policy rules from the policy directory."""
        policy_dir = Path(self.config["policy_engine"]["policy_directory"])
        
        if not policy_dir.exists():
            self.logger.warning(f"Policy directory {policy_dir} does not exist")
            self._create_default_policies()
            return
        
        for policy_file in policy_dir.glob("*.yaml"):
            try:
                with open(policy_file) as f:
                    policy_data = yaml.safe_load(f)
                    
                    for rule_data in policy_data.get("rules", []):
                        policy_rule = PolicyRule(
                            id=rule_data["id"],
                            name=rule_data["name"],
                            category=rule_data["category"],
                            severity=rule_data["severity"],
                            description=rule_data["description"],
                            rule_logic=rule_data["rule_logic"],
                            enforcement_action=rule_data.get("enforcement_action", "warn"),
                            exceptions=rule_data.get("exceptions", []),
                            created_by=rule_data.get("created_by", "system"),
                            created_at=datetime.fromisoformat(rule_data.get("created_at", datetime.now().isoformat())),
                            last_updated=datetime.fromisoformat(rule_data.get("last_updated", datetime.now().isoformat())),
                            enabled=rule_data.get("enabled", True)
                        )
                        
                        self.policy_rules.append(policy_rule)
                        
            except Exception as e:
                self.logger.error(f"Error loading policy file {policy_file}: {e}")
        
        self.logger.info(f"Loaded {len(self.policy_rules)} policy rules")
    
    def _create_default_policies(self):
        """Create default policy rules."""
        default_policies_dir = Path(self.config["policy_engine"]["policy_directory"])
        default_policies_dir.mkdir(parents=True, exist_ok=True)
        
        # Security policies
        security_policies = {
            "apiVersion": "v1",
            "kind": "PolicySet",
            "metadata": {"name": "security-policies"},
            "rules": [
                {
                    "id": "SEC001",
                    "name": "Require Security Context",
                    "category": "security",
                    "severity": "high",
                    "description": "All containers must run with a security context",
                    "rule_logic": {
                        "rego": """
                        deny[msg] {
                            input.kind == "Pod"
                            container := input.spec.containers[_]
                            not container.securityContext
                            msg := "Container must have securityContext defined"
                        }
                        """
                    },
                    "enforcement_action": "block"
                },
                {
                    "id": "SEC002", 
                    "name": "Prohibit Privileged Containers",
                    "category": "security",
                    "severity": "critical",
                    "description": "Containers must not run in privileged mode",
                    "rule_logic": {
                        "rego": """
                        deny[msg] {
                            input.kind == "Pod"
                            container := input.spec.containers[_]
                            container.securityContext.privileged == true
                            msg := "Privileged containers are not allowed"
                        }
                        """
                    },
                    "enforcement_action": "block"
                },
                {
                    "id": "SEC003",
                    "name": "Require Resource Limits",
                    "category": "security",
                    "severity": "medium",
                    "description": "All containers must have resource limits defined",
                    "rule_logic": {
                        "rego": """
                        deny[msg] {
                            input.kind == "Pod"
                            container := input.spec.containers[_]
                            not container.resources.limits
                            msg := "Container must have resource limits defined"
                        }
                        """
                    },
                    "enforcement_action": "warn"
                }
            ]
        }
        
        # Cost governance policies
        cost_policies = {
            "apiVersion": "v1",
            "kind": "PolicySet",
            "metadata": {"name": "cost-policies"},
            "rules": [
                {
                    "id": "COST001",
                    "name": "Limit CPU Requests",
                    "category": "cost",
                    "severity": "medium",
                    "description": "CPU requests should not exceed reasonable limits",
                    "rule_logic": {
                        "rego": """
                        deny[msg] {
                            input.kind == "Pod"
                            container := input.spec.containers[_]
                            cpu_request := container.resources.requests.cpu
                            cpu_limit_cores := 8
                            to_number(trim_suffix(cpu_request, "m")) / 1000 > cpu_limit_cores
                            msg := sprintf("CPU request %v exceeds limit of %v cores", [cpu_request, cpu_limit_cores])
                        }
                        """
                    },
                    "enforcement_action": "warn"
                },
                {
                    "id": "COST002",
                    "name": "Require Cost Center Tag",
                    "category": "cost",
                    "severity": "low",
                    "description": "All resources must have a cost center tag",
                    "rule_logic": {
                        "rego": """
                        deny[msg] {
                            not input.metadata.labels["cost-center"]
                            msg := "Resource must have cost-center label"
                        }
                        """
                    },
                    "enforcement_action": "warn"
                }
            ]
        }
        
        # Compliance policies
        compliance_policies = {
            "apiVersion": "v1",
            "kind": "PolicySet", 
            "metadata": {"name": "compliance-policies"},
            "rules": [
                {
                    "id": "COMP001",
                    "name": "Require Data Classification",
                    "category": "compliance",
                    "severity": "high",
                    "description": "All data processing workloads must have data classification",
                    "rule_logic": {
                        "rego": """
                        deny[msg] {
                            input.metadata.labels["data-processing"] == "true"
                            not input.metadata.labels["data-classification"]
                            msg := "Data processing workloads must have data-classification label"
                        }
                        """
                    },
                    "enforcement_action": "block"
                },
                {
                    "id": "COMP002",
                    "name": "Audit Log Requirements",
                    "category": "compliance",
                    "severity": "high",
                    "description": "Critical services must enable audit logging",
                    "rule_logic": {
                        "rego": """
                        deny[msg] {
                            input.metadata.labels["criticality"] == "high"
                            not input.metadata.annotations["audit-logging"]
                            msg := "High criticality services must enable audit logging"
                        }
                        """
                    },
                    "enforcement_action": "block"
                }
            ]
        }
        
        # Write policy files
        policy_files = [
            ("security-policies.yaml", security_policies),
            ("cost-policies.yaml", cost_policies),
            ("compliance-policies.yaml", compliance_policies)
        ]
        
        for filename, policy_data in policy_files:
            policy_path = default_policies_dir / filename
            with open(policy_path, 'w') as f:
                yaml.dump(policy_data, f, default_flow_style=False)
        
        self.logger.info("Created default policy files")
        
        # Reload policies
        self._load_policy_rules()
    
    async def evaluate_policies(self, resource_type: str = "all") -> Dict[str, Any]:
        """Evaluate policies against current resources."""
        self.logger.info(f"Starting policy evaluation for resource type: {resource_type}")
        
        evaluation_results = {
            "evaluation_id": f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "started_at": datetime.now().isoformat(),
            "resource_type": resource_type,
            "policies_evaluated": len([p for p in self.policy_rules if p.enabled]),
            "violations": [],
            "summary": {
                "total_resources_evaluated": 0,
                "violations_by_severity": {"critical": 0, "high": 0, "medium": 0, "low": 0},
                "violations_by_category": {}
            }
        }
        
        try:
            # Evaluate Kubernetes resources
            if resource_type in ["all", "kubernetes"] and "k8s" in self.clients:
                k8s_violations = await self._evaluate_kubernetes_policies()
                evaluation_results["violations"].extend(k8s_violations)
            
            # Evaluate AWS resources
            if resource_type in ["all", "aws"] and "aws" in self.clients:
                aws_violations = await self._evaluate_aws_policies()
                evaluation_results["violations"].extend(aws_violations)
            
            # Evaluate CI/CD pipelines
            if resource_type in ["all", "cicd"]:
                cicd_violations = await self._evaluate_cicd_policies()
                evaluation_results["violations"].extend(cicd_violations)
            
            # Update summary
            evaluation_results["summary"] = self._calculate_evaluation_summary(evaluation_results["violations"])
            
            # Store violations
            self.violations.extend([
                PolicyViolation(
                    id=f"violation_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{i}",
                    policy_rule_id=v["policy_rule_id"],
                    resource_type=v["resource_type"],
                    resource_id=v["resource_id"],
                    resource_namespace=v.get("resource_namespace"),
                    violation_details=v["violation_details"],
                    severity=v["severity"],
                    detected_at=datetime.now(),
                    status="open",
                    remediation_action=None,
                    remediated_at=None
                )
                for i, v in enumerate(evaluation_results["violations"])
            ])
            
            # Send notifications for critical violations
            critical_violations = [v for v in evaluation_results["violations"] if v["severity"] == "critical"]
            if critical_violations:
                await self._send_violation_notifications(critical_violations)
            
            evaluation_results["completed_at"] = datetime.now().isoformat()
            
        except Exception as e:
            self.logger.error(f"Error during policy evaluation: {e}")
            evaluation_results["error"] = str(e)
        
        return evaluation_results
    
    async def _evaluate_kubernetes_policies(self) -> List[Dict]:
        """Evaluate policies against Kubernetes resources."""
        violations = []
        
        try:
            k8s_core = self.clients["k8s"]["core"]
            
            # Get all namespaces to evaluate
            namespaces = self.config["integrations"]["kubernetes"]["namespaces"]
            
            for namespace in namespaces:
                # Evaluate Pods
                pods = k8s_core.list_namespaced_pod(namespace=namespace)
                
                for pod in pods.items:
                    pod_violations = self._evaluate_resource_against_policies(
                        resource_type="Pod",
                        resource_data=pod.to_dict(),
                        resource_id=pod.metadata.name,
                        resource_namespace=namespace
                    )
                    violations.extend(pod_violations)
                
                # Evaluate Services
                services = k8s_core.list_namespaced_service(namespace=namespace)
                
                for service in services.items:
                    service_violations = self._evaluate_resource_against_policies(
                        resource_type="Service",
                        resource_data=service.to_dict(),
                        resource_id=service.metadata.name,
                        resource_namespace=namespace
                    )
                    violations.extend(service_violations)
        
        except Exception as e:
            self.logger.error(f"Error evaluating Kubernetes policies: {e}")
        
        return violations
    
    async def _evaluate_aws_policies(self) -> List[Dict]:
        """Evaluate policies against AWS resources."""
        violations = []
        
        try:
            # Evaluate EC2 instances
            ec2_client = self.clients["aws"]["ec2"]
            
            for region in self.config["integrations"]["aws"]["regions"]:
                # Set region for client
                ec2_regional = boto3.client('ec2', region_name=region)
                
                # Get EC2 instances
                response = ec2_regional.describe_instances()
                
                for reservation in response['Reservations']:
                    for instance in reservation['Instances']:
                        instance_violations = self._evaluate_resource_against_policies(
                            resource_type="EC2Instance",
                            resource_data=instance,
                            resource_id=instance['InstanceId'],
                            resource_namespace=region
                        )
                        violations.extend(instance_violations)
            
            # Evaluate S3 buckets
            s3_client = self.clients["aws"]["s3"]
            
            buckets = s3_client.list_buckets()
            
            for bucket in buckets['Buckets']:
                bucket_name = bucket['Name']
                
                # Get bucket details
                try:
                    bucket_location = s3_client.get_bucket_location(Bucket=bucket_name)
                    bucket_tagging = s3_client.get_bucket_tagging(Bucket=bucket_name)
                    
                    bucket_data = {
                        'BucketName': bucket_name,
                        'CreationDate': bucket['CreationDate'],
                        'Region': bucket_location.get('LocationConstraint', 'us-east-1'),
                        'Tags': bucket_tagging.get('TagSet', [])
                    }
                    
                    bucket_violations = self._evaluate_resource_against_policies(
                        resource_type="S3Bucket",
                        resource_data=bucket_data,
                        resource_id=bucket_name,
                        resource_namespace=bucket_data['Region']
                    )
                    violations.extend(bucket_violations)
                    
                except Exception as e:
                    # Skip buckets we can't access
                    self.logger.warning(f"Could not evaluate S3 bucket {bucket_name}: {e}")
        
        except Exception as e:
            self.logger.error(f"Error evaluating AWS policies: {e}")
        
        return violations
    
    async def _evaluate_cicd_policies(self) -> List[Dict]:
        """Evaluate policies against CI/CD pipelines."""
        violations = []
        
        # This would integrate with GitHub Actions, GitLab CI, Jenkins, etc.
        # For now, we'll create placeholder violations
        
        try:
            # Check for required CI/CD policies
            required_checks = [
                "security_scan",
                "vulnerability_check", 
                "license_check",
                "code_quality",
                "test_coverage"
            ]
            
            # This would be replaced with actual CI/CD system integration
            pipeline_data = {
                "pipeline_id": "sample_pipeline",
                "checks_configured": ["security_scan", "test_coverage"],
                "repository": "agentic-startup-studio-boilerplate"
            }
            
            missing_checks = set(required_checks) - set(pipeline_data["checks_configured"])
            
            if missing_checks:
                violations.append({
                    "policy_rule_id": "CICD001",
                    "resource_type": "CI/CD Pipeline",
                    "resource_id": pipeline_data["pipeline_id"],
                    "resource_namespace": None,
                    "violation_details": {
                        "missing_checks": list(missing_checks),
                        "repository": pipeline_data["repository"]
                    },
                    "severity": "medium"
                })
        
        except Exception as e:
            self.logger.error(f"Error evaluating CI/CD policies: {e}")
        
        return violations
    
    def _evaluate_resource_against_policies(self, resource_type: str, resource_data: Dict,
                                          resource_id: str, resource_namespace: Optional[str]) -> List[Dict]:
        """Evaluate a single resource against all applicable policies."""
        violations = []
        
        for policy in self.policy_rules:
            if not policy.enabled:
                continue
            
            try:
                # Check if policy applies to this resource type
                if self._policy_applies_to_resource(policy, resource_type, resource_data):
                    violation = self._evaluate_single_policy(policy, resource_data, resource_type, resource_id, resource_namespace)
                    if violation:
                        violations.append(violation)
            
            except Exception as e:
                self.logger.error(f"Error evaluating policy {policy.id} against resource {resource_id}: {e}")
        
        return violations
    
    def _policy_applies_to_resource(self, policy: PolicyRule, resource_type: str, resource_data: Dict) -> bool:
        """Check if a policy applies to a specific resource."""
        # This would contain logic to determine if a policy applies to a resource
        # For now, we'll use simple type matching
        
        policy_logic = policy.rule_logic
        
        # Check if policy specifies resource types
        if "resource_types" in policy_logic:
            return resource_type in policy_logic["resource_types"]
        
        # If no specific resource types, assume it applies to all
        return True
    
    def _evaluate_single_policy(self, policy: PolicyRule, resource_data: Dict, 
                               resource_type: str, resource_id: str, 
                               resource_namespace: Optional[str]) -> Optional[Dict]:
        """Evaluate a single policy against a resource."""
        try:
            # This is a simplified policy evaluation
            # In a real implementation, this would use OPA or another policy engine
            
            rule_logic = policy.rule_logic
            
            # Example: Check for required labels
            if "required_labels" in rule_logic:
                required_labels = rule_logic["required_labels"]
                resource_labels = resource_data.get("metadata", {}).get("labels", {})
                
                missing_labels = set(required_labels) - set(resource_labels.keys())
                if missing_labels:
                    return {
                        "policy_rule_id": policy.id,
                        "resource_type": resource_type,
                        "resource_id": resource_id,
                        "resource_namespace": resource_namespace,
                        "violation_details": {
                            "missing_labels": list(missing_labels),
                            "required_labels": required_labels
                        },
                        "severity": policy.severity
                    }
            
            # Example: Check for resource limits
            if "require_resource_limits" in rule_logic and rule_logic["require_resource_limits"]:
                if resource_type == "Pod":
                    containers = resource_data.get("spec", {}).get("containers", [])
                    for container in containers:
                        if not container.get("resources", {}).get("limits"):
                            return {
                                "policy_rule_id": policy.id,
                                "resource_type": resource_type,
                                "resource_id": resource_id,
                                "resource_namespace": resource_namespace,
                                "violation_details": {
                                    "container_name": container.get("name", "unknown"),
                                    "issue": "missing resource limits"
                                },
                                "severity": policy.severity
                            }
            
            # Example: Check for privileged containers
            if "deny_privileged" in rule_logic and rule_logic["deny_privileged"]:
                if resource_type == "Pod":
                    containers = resource_data.get("spec", {}).get("containers", [])
                    for container in containers:
                        security_context = container.get("securityContext", {})
                        if security_context.get("privileged", False):
                            return {
                                "policy_rule_id": policy.id,
                                "resource_type": resource_type,
                                "resource_id": resource_id,
                                "resource_namespace": resource_namespace,
                                "violation_details": {
                                    "container_name": container.get("name", "unknown"),
                                    "issue": "privileged container detected"
                                },
                                "severity": policy.severity
                            }
            
        except Exception as e:
            self.logger.error(f"Error in policy evaluation logic for {policy.id}: {e}")
        
        return None
    
    def _calculate_evaluation_summary(self, violations: List[Dict]) -> Dict[str, Any]:
        """Calculate summary statistics for policy evaluation."""
        summary = {
            "total_resources_evaluated": len(set(v["resource_id"] for v in violations)),
            "total_violations": len(violations),
            "violations_by_severity": {"critical": 0, "high": 0, "medium": 0, "low": 0},
            "violations_by_category": {}
        }
        
        for violation in violations:
            severity = violation.get("severity", "unknown")
            if severity in summary["violations_by_severity"]:
                summary["violations_by_severity"][severity] += 1
            
            # Get policy category
            policy = next((p for p in self.policy_rules if p.id == violation["policy_rule_id"]), None)
            if policy:
                category = policy.category
                if category not in summary["violations_by_category"]:
                    summary["violations_by_category"][category] = 0
                summary["violations_by_category"][category] += 1
        
        return summary
    
    async def _send_violation_notifications(self, violations: List[Dict]):
        """Send notifications for policy violations."""
        try:
            notification_config = self.config["notifications"]
            
            # Prepare notification message
            message = self._format_violation_notification(violations)
            
            # Send Slack notification
            if notification_config.get("slack_webhook"):
                await self._send_slack_notification(notification_config["slack_webhook"], message)
            
            # Send email notification
            if notification_config.get("email_recipients"):
                await self._send_email_notification(notification_config["email_recipients"], message)
            
            # Send PagerDuty alert
            if notification_config.get("pagerduty_integration_key"):
                await self._send_pagerduty_alert(notification_config["pagerduty_integration_key"], violations)
        
        except Exception as e:
            self.logger.error(f"Error sending violation notifications: {e}")
    
    def _format_violation_notification(self, violations: List[Dict]) -> str:
        """Format violation notification message."""
        message = f"🚨 **Policy Violations Detected** 🚨\n\n"
        message += f"**Total Violations:** {len(violations)}\n"
        
        severity_counts = {}
        for violation in violations:
            severity = violation.get("severity", "unknown")
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        message += "**By Severity:**\n"
        for severity, count in severity_counts.items():
            emoji = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(severity, "⚪")
            message += f"  {emoji} {severity.title()}: {count}\n"
        
        message += "\n**Top Violations:**\n"
        for i, violation in enumerate(violations[:5], 1):
            message += f"{i}. **{violation['resource_type']}** `{violation['resource_id']}` - {violation.get('violation_details', {}).get('issue', 'Policy violation')}\n"
        
        if len(violations) > 5:
            message += f"\n...and {len(violations) - 5} more violations.\n"
        
        message += f"\n**Generated at:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}"
        
        return message
    
    async def _send_slack_notification(self, webhook_url: str, message: str):
        """Send Slack notification."""
        try:
            payload = {
                "text": "Policy Violations Detected",
                "attachments": [
                    {
                        "color": "danger",
                        "text": message,
                        "mrkdwn_in": ["text"]
                    }
                ]
            }
            
            response = requests.post(webhook_url, json=payload)
            response.raise_for_status()
            
        except Exception as e:
            self.logger.error(f"Error sending Slack notification: {e}")
    
    async def _send_email_notification(self, recipients: List[str], message: str):
        """Send email notification."""
        # This would integrate with your email service
        self.logger.info(f"Would send email notification to {recipients}")
    
    async def _send_pagerduty_alert(self, integration_key: str, violations: List[Dict]):
        """Send PagerDuty alert."""
        # This would integrate with PagerDuty API
        self.logger.info(f"Would send PagerDuty alert for {len(violations)} violations")
    
    def generate_compliance_report(self, framework: str = "SOC2") -> ComplianceReport:
        """Generate compliance report for specified framework."""
        self.logger.info(f"Generating compliance report for {framework}")
        
        # Calculate compliance metrics
        total_policies = len([p for p in self.policy_rules if p.enabled])
        
        # Get violations from last 30 days
        recent_violations = [
            v for v in self.violations 
            if v.detected_at > datetime.now() - timedelta(days=30)
        ]
        
        violations_by_severity = {}
        for violation in recent_violations:
            severity = violation.severity
            violations_by_severity[severity] = violations_by_severity.get(severity, 0) + 1
        
        compliance_by_category = {}
        for category in ["security", "operational", "cost", "compliance"]:
            category_policies = [p for p in self.policy_rules if p.category == category and p.enabled]
            category_violations = [v for v in recent_violations if any(p.id == v.policy_rule_id and p.category == category for p in self.policy_rules)]
            
            if category_policies:
                compliance_score = max(0, 100 - (len(category_violations) / len(category_policies)) * 100)
                compliance_by_category[category] = compliance_score
        
        # Calculate overall compliance score
        overall_score = sum(compliance_by_category.values()) / len(compliance_by_category) if compliance_by_category else 0
        
        # Generate recommendations
        recommendations = []
        
        critical_violations = [v for v in recent_violations if v.severity == "critical"]
        if critical_violations:
            recommendations.append(f"Address {len(critical_violations)} critical policy violations immediately")
        
        if overall_score < 80:
            recommendations.append("Overall compliance score is below target (80%). Review policy implementation.")
        
        for category, score in compliance_by_category.items():
            if score < 70:
                recommendations.append(f"Improve {category} compliance (currently {score:.1f}%)")
        
        report = ComplianceReport(
            report_id=f"compliance_{framework.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            report_type="on_demand",
            generated_at=datetime.now(),
            compliance_framework=framework,
            overall_score=overall_score,
            total_policies=total_policies,
            violations_by_severity=violations_by_severity,
            compliance_by_category=compliance_by_category,
            recommendations=recommendations
        )
        
        return report
    
    def save_report(self, report: Union[Dict, ComplianceReport], file_path: str):
        """Save report to file."""
        try:
            report_data = asdict(report) if isinstance(report, ComplianceReport) else report
            
            # Convert datetime objects to ISO format strings
            def serialize_datetime(obj):
                if isinstance(obj, datetime):
                    return obj.isoformat()
                return obj
            
            def serialize_dict(d):
                if isinstance(d, dict):
                    return {k: serialize_dict(v) for k, v in d.items()}
                elif isinstance(d, list):
                    return [serialize_dict(item) for item in d]
                else:
                    return serialize_datetime(d)
            
            serialized_data = serialize_dict(report_data)
            
            report_path = Path(file_path)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            
            with open(report_path, 'w') as f:
                json.dump(serialized_data, f, indent=2)
            
            self.logger.info(f"Report saved to {report_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving report to {file_path}: {e}")


async def main():
    """Main function for standalone execution."""
    # Initialize Policy-as-Code framework
    policy_framework = PolicyAsCodeFramework()
    
    print("Policy-as-Code Framework")
    print("=" * 50)
    
    # Evaluate policies
    print("\n1. Evaluating policies...")
    evaluation_results = await policy_framework.evaluate_policies()
    
    print(f"Evaluation completed:")
    print(f"  - Policies evaluated: {evaluation_results['policies_evaluated']}")
    print(f"  - Total violations: {evaluation_results['summary']['total_violations']}")
    print(f"  - Critical violations: {evaluation_results['summary']['violations_by_severity']['critical']}")
    print(f"  - High violations: {evaluation_results['summary']['violations_by_severity']['high']}")
    
    # Generate compliance report
    print("\n2. Generating compliance report...")
    compliance_report = policy_framework.generate_compliance_report("SOC2")
    
    print(f"Compliance Report:")
    print(f"  - Overall score: {compliance_report.overall_score:.1f}%")
    print(f"  - Total policies: {compliance_report.total_policies}")
    print(f"  - Recommendations: {len(compliance_report.recommendations)}")
    
    # Save reports
    print("\n3. Saving reports...")
    policy_framework.save_report(evaluation_results, "reports/policy_evaluation.json")
    policy_framework.save_report(compliance_report, "reports/compliance_report.json")
    
    print("\nPolicy-as-Code framework execution completed!")


if __name__ == "__main__":
    asyncio.run(main())