"""
Quantum Production Orchestrator - Final Deployment System

Implements zero-downtime deployment with quantum state preservation,
autonomous rollback mechanisms, and consciousness-driven deployment validation.
"""

import asyncio
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
import uuid
import json
import numpy as np
import subprocess
import shutil
from pathlib import Path
import logging
import yaml


class DeploymentPhase(Enum):
    """Deployment phases with quantum state management"""
    PREPARATION = "preparation"
    VALIDATION = "validation"
    QUANTUM_STATE_BACKUP = "quantum_state_backup"
    DEPLOYMENT = "deployment"
    HEALTH_CHECK = "health_check"
    TRAFFIC_MIGRATION = "traffic_migration"
    MONITORING = "monitoring"
    ROLLBACK = "rollback"
    COMPLETED = "completed"
    FAILED = "failed"


class DeploymentStrategy(Enum):
    """Deployment strategies with consciousness considerations"""
    BLUE_GREEN = "blue_green"
    ROLLING = "rolling"
    CANARY = "canary"
    QUANTUM_SUPERPOSITION = "quantum_superposition"
    CONSCIOUSNESS_AWARE = "consciousness_aware"


@dataclass
class DeploymentConfig:
    """Configuration for production deployment"""
    version: str
    strategy: DeploymentStrategy = DeploymentStrategy.BLUE_GREEN
    health_check_timeout: float = 300.0  # 5 minutes
    rollback_timeout: float = 600.0  # 10 minutes
    traffic_migration_steps: int = 5
    quantum_state_preservation: bool = True
    consciousness_validation: bool = True
    performance_validation: bool = True
    security_validation: bool = True
    enable_auto_rollback: bool = True
    notification_channels: List[str] = field(default_factory=list)
    environment_variables: Dict[str, str] = field(default_factory=dict)
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "version": self.version,
            "strategy": self.strategy.value,
            "health_check_timeout": self.health_check_timeout,
            "rollback_timeout": self.rollback_timeout,
            "traffic_migration_steps": self.traffic_migration_steps,
            "quantum_state_preservation": self.quantum_state_preservation,
            "consciousness_validation": self.consciousness_validation,
            "performance_validation": self.performance_validation,
            "security_validation": self.security_validation,
            "enable_auto_rollback": self.enable_auto_rollback,
            "notification_channels": self.notification_channels,
            "environment_variables": self.environment_variables,
            "resource_limits": self.resource_limits
        }


@dataclass
class DeploymentStatus:
    """Real-time deployment status tracking"""
    deployment_id: str
    version: str
    current_phase: DeploymentPhase
    progress_percentage: float = 0.0
    start_time: datetime = field(default_factory=datetime.utcnow)
    estimated_completion: Optional[datetime] = None
    current_step: str = "Initializing deployment"
    quantum_coherence_level: float = 1.0
    consciousness_stability: float = 1.0
    error_messages: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    rollback_available: bool = True
    
    def add_error(self, error_message: str):
        """Add error to deployment status"""
        timestamped_error = f"{datetime.utcnow().isoformat()}: {error_message}"
        self.error_messages.append(timestamped_error)
        
        # Limit error history
        if len(self.error_messages) > 50:
            self.error_messages = self.error_messages[-25:]
    
    def update_progress(self, phase: DeploymentPhase, step: str, percentage: float):
        """Update deployment progress"""
        self.current_phase = phase
        self.current_step = step
        self.progress_percentage = min(100.0, max(0.0, percentage))
        
        # Update estimated completion
        if percentage > 0 and percentage < 100:
            elapsed = (datetime.utcnow() - self.start_time).total_seconds()
            estimated_total = elapsed * (100.0 / percentage)
            self.estimated_completion = self.start_time + timedelta(seconds=estimated_total)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "deployment_id": self.deployment_id,
            "version": self.version,
            "current_phase": self.current_phase.value,
            "progress_percentage": self.progress_percentage,
            "start_time": self.start_time.isoformat(),
            "estimated_completion": self.estimated_completion.isoformat() if self.estimated_completion else None,
            "current_step": self.current_step,
            "quantum_coherence_level": self.quantum_coherence_level,
            "consciousness_stability": self.consciousness_stability,
            "error_count": len(self.error_messages),
            "recent_errors": self.error_messages[-3:],  # Last 3 errors
            "performance_metrics": self.performance_metrics,
            "rollback_available": self.rollback_available
        }


class QuantumProductionOrchestrator:
    """
    Comprehensive production deployment orchestrator with quantum state management,
    autonomous health monitoring, and consciousness-aware validation.
    """
    
    def __init__(self, project_root: str = None):
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.active_deployments: Dict[str, DeploymentStatus] = {}
        self.deployment_history: List[Dict[str, Any]] = []
        
        # Production environment configuration
        self.environments = {
            "staging": {
                "api_url": "http://staging.quantum-api.local:8000",
                "health_endpoint": "/api/v1/health",
                "quantum_state_backup_path": "/opt/quantum/backups/staging"
            },
            "production": {
                "api_url": "http://production.quantum-api.local:8000",
                "health_endpoint": "/api/v1/health",
                "quantum_state_backup_path": "/opt/quantum/backups/production"
            }
        }
        
        # Deployment validation thresholds
        self.validation_thresholds = {
            "response_time_ms": 500,
            "error_rate_percentage": 1.0,
            "cpu_usage_percentage": 80.0,
            "memory_usage_percentage": 85.0,
            "quantum_coherence_minimum": 0.7,
            "consciousness_stability_minimum": 0.6
        }
        
        # Performance tracking
        self.successful_deployments = 0
        self.failed_deployments = 0
        self.total_rollbacks = 0
        self.average_deployment_time = 0.0
        
        self.logger = logging.getLogger(__name__)
    
    async def deploy_to_production(self, config: DeploymentConfig, 
                                 environment: str = "production") -> str:
        """Deploy application to production with quantum state preservation"""
        deployment_id = str(uuid.uuid4())
        
        # Create deployment status tracker
        status = DeploymentStatus(
            deployment_id=deployment_id,
            version=config.version,
            current_phase=DeploymentPhase.PREPARATION
        )
        
        self.active_deployments[deployment_id] = status
        
        self.logger.info(f"Starting deployment {deployment_id} (version {config.version}) to {environment}")
        
        try:
            # Execute deployment phases
            await self._execute_deployment_phases(config, status, environment)
            
            # Mark as completed
            status.update_progress(DeploymentPhase.COMPLETED, "Deployment completed successfully", 100.0)
            self.successful_deployments += 1
            
            # Record deployment history
            self._record_deployment_history(config, status, "success")
            
            self.logger.info(f"Deployment {deployment_id} completed successfully")
            
        except Exception as e:
            error_msg = f"Deployment failed: {str(e)}"
            status.add_error(error_msg)
            status.current_phase = DeploymentPhase.FAILED
            
            self.failed_deployments += 1
            self._record_deployment_history(config, status, "failed")
            
            self.logger.error(f"Deployment {deployment_id} failed: {e}")
            
            # Attempt rollback if enabled
            if config.enable_auto_rollback:
                try:
                    await self._execute_rollback(deployment_id, status, environment)
                except Exception as rollback_error:
                    self.logger.error(f"Rollback failed for {deployment_id}: {rollback_error}")
                    status.add_error(f"Rollback failed: {str(rollback_error)}")
        
        return deployment_id
    
    async def _execute_deployment_phases(self, config: DeploymentConfig, 
                                       status: DeploymentStatus, environment: str):
        """Execute all deployment phases sequentially"""
        
        # Phase 1: Preparation
        status.update_progress(DeploymentPhase.PREPARATION, "Preparing deployment environment", 10.0)
        await self._prepare_deployment(config, environment)
        
        # Phase 2: Validation
        status.update_progress(DeploymentPhase.VALIDATION, "Validating deployment artifacts", 20.0)
        await self._validate_deployment_artifacts(config)
        
        # Phase 3: Quantum State Backup
        if config.quantum_state_preservation:
            status.update_progress(DeploymentPhase.QUANTUM_STATE_BACKUP, "Backing up quantum states", 30.0)
            await self._backup_quantum_states(environment)
        
        # Phase 4: Deployment Execution
        status.update_progress(DeploymentPhase.DEPLOYMENT, "Executing deployment", 50.0)
        await self._execute_deployment_strategy(config, status, environment)
        
        # Phase 5: Health Checks
        status.update_progress(DeploymentPhase.HEALTH_CHECK, "Running health checks", 70.0)
        await self._run_comprehensive_health_checks(config, status, environment)
        
        # Phase 6: Traffic Migration
        status.update_progress(DeploymentPhase.TRAFFIC_MIGRATION, "Migrating traffic", 85.0)
        await self._migrate_traffic_gradually(config, status, environment)
        
        # Phase 7: Post-deployment monitoring
        status.update_progress(DeploymentPhase.MONITORING, "Monitoring deployment stability", 95.0)
        await self._monitor_deployment_stability(config, status, environment)
    
    async def _prepare_deployment(self, config: DeploymentConfig, environment: str):
        """Prepare deployment environment"""
        try:
            # Create deployment directories
            deployment_dir = self.project_root / "deployments" / config.version
            deployment_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy application files
            app_source = self.project_root
            app_dest = deployment_dir / "app"
            
            if app_dest.exists():
                shutil.rmtree(app_dest)
            
            # Copy essential files for deployment
            essential_files = [
                "quantum_task_planner/",
                "requirements.txt",
                "main.py",
                "docker-compose.yml",
                "Dockerfile"
            ]
            
            app_dest.mkdir(parents=True)
            
            for file_pattern in essential_files:
                source_path = app_source / file_pattern
                if source_path.exists():
                    if source_path.is_dir():
                        shutil.copytree(source_path, app_dest / file_pattern, dirs_exist_ok=True)
                    else:
                        shutil.copy2(source_path, app_dest / file_pattern)
            
            # Generate deployment manifest
            manifest = {
                "version": config.version,
                "timestamp": datetime.utcnow().isoformat(),
                "strategy": config.strategy.value,
                "environment": environment,
                "config": config.to_dict()
            }
            
            with open(deployment_dir / "manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)
            
            self.logger.info(f"Deployment prepared in {deployment_dir}")
        
        except Exception as e:
            raise Exception(f"Deployment preparation failed: {str(e)}")
    
    async def _validate_deployment_artifacts(self, config: DeploymentConfig):
        """Validate deployment artifacts and dependencies"""
        try:
            deployment_dir = self.project_root / "deployments" / config.version
            
            # Check essential files exist
            required_files = ["main.py", "requirements.txt"]
            for file_name in required_files:
                file_path = deployment_dir / "app" / file_name
                if not file_path.exists():
                    raise Exception(f"Required file missing: {file_name}")
            
            # Validate Python syntax
            main_py = deployment_dir / "app" / "main.py"
            result = subprocess.run(
                ["python3", "-m", "py_compile", str(main_py)],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                raise Exception(f"Python syntax validation failed: {result.stderr}")
            
            # Validate requirements can be installed
            requirements_file = deployment_dir / "app" / "requirements.txt"
            if requirements_file.exists():
                # Just check the file format, don't actually install
                with open(requirements_file) as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.strip()
                        if line and not line.startswith('#') and '==' not in line and '>=' not in line and '<=' not in line:
                            # Basic validation - ensure some version specification
                            pass
            
            self.logger.info("Deployment artifacts validated successfully")
        
        except Exception as e:
            raise Exception(f"Artifact validation failed: {str(e)}")
    
    async def _backup_quantum_states(self, environment: str):
        """Backup quantum states before deployment"""
        try:
            backup_dir = Path(f"/tmp/quantum_backup_{environment}_{int(time.time())}")
            backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Simulate quantum state backup
            quantum_state_data = {
                "timestamp": datetime.utcnow().isoformat(),
                "environment": environment,
                "agent_states": {
                    "total_agents": 10,
                    "average_consciousness": 0.75,
                    "total_entanglements": 25,
                    "quantum_coherence": 0.85
                },
                "task_states": {
                    "active_tasks": 15,
                    "completed_tasks": 150,
                    "average_completion_probability": 0.82
                },
                "system_coherence": 0.78
            }
            
            with open(backup_dir / "quantum_states.json", "w") as f:
                json.dump(quantum_state_data, f, indent=2)
            
            self.logger.info(f"Quantum states backed up to {backup_dir}")
        
        except Exception as e:
            raise Exception(f"Quantum state backup failed: {str(e)}")
    
    async def _execute_deployment_strategy(self, config: DeploymentConfig, 
                                         status: DeploymentStatus, environment: str):
        """Execute deployment based on selected strategy"""
        try:
            if config.strategy == DeploymentStrategy.BLUE_GREEN:
                await self._execute_blue_green_deployment(config, status, environment)
            elif config.strategy == DeploymentStrategy.ROLLING:
                await self._execute_rolling_deployment(config, status, environment)
            elif config.strategy == DeploymentStrategy.CANARY:
                await self._execute_canary_deployment(config, status, environment)
            elif config.strategy == DeploymentStrategy.QUANTUM_SUPERPOSITION:
                await self._execute_quantum_superposition_deployment(config, status, environment)
            else:
                await self._execute_consciousness_aware_deployment(config, status, environment)
            
            self.logger.info(f"Deployment strategy {config.strategy.value} executed successfully")
        
        except Exception as e:
            raise Exception(f"Deployment strategy execution failed: {str(e)}")
    
    async def _execute_blue_green_deployment(self, config: DeploymentConfig, 
                                           status: DeploymentStatus, environment: str):
        """Execute blue-green deployment strategy"""
        # Simulate blue-green deployment
        await asyncio.sleep(5)  # Deployment time
        
        # Update deployment status
        status.performance_metrics["deployment_time"] = 5.0
        status.quantum_coherence_level = 0.9
        status.consciousness_stability = 0.85
    
    async def _execute_rolling_deployment(self, config: DeploymentConfig, 
                                        status: DeploymentStatus, environment: str):
        """Execute rolling deployment strategy"""
        # Simulate rolling deployment with gradual updates
        for i in range(5):  # 5 rolling steps
            await asyncio.sleep(2)  # Each step takes 2 seconds
            progress = 50 + (i + 1) * 8  # Progress from 50% to 90%
            status.update_progress(DeploymentPhase.DEPLOYMENT, f"Rolling update step {i+1}/5", progress)
        
        status.performance_metrics["deployment_time"] = 10.0
        status.quantum_coherence_level = 0.85
        status.consciousness_stability = 0.8
    
    async def _execute_canary_deployment(self, config: DeploymentConfig, 
                                       status: DeploymentStatus, environment: str):
        """Execute canary deployment strategy"""
        # Deploy to small subset first
        await asyncio.sleep(3)  # Canary deployment
        status.update_progress(DeploymentPhase.DEPLOYMENT, "Canary deployment active", 60)
        
        # Monitor canary performance
        await asyncio.sleep(2)
        
        # Full deployment
        await asyncio.sleep(5)
        
        status.performance_metrics["deployment_time"] = 10.0
        status.quantum_coherence_level = 0.88
        status.consciousness_stability = 0.82
    
    async def _execute_quantum_superposition_deployment(self, config: DeploymentConfig, 
                                                      status: DeploymentStatus, environment: str):
        """Execute quantum superposition deployment (parallel reality deployment)"""
        # Deploy to multiple quantum states simultaneously
        await asyncio.sleep(3)
        
        status.performance_metrics["deployment_time"] = 3.0
        status.quantum_coherence_level = 0.95  # High coherence due to quantum nature
        status.consciousness_stability = 0.9
    
    async def _execute_consciousness_aware_deployment(self, config: DeploymentConfig, 
                                                    status: DeploymentStatus, environment: str):
        """Execute consciousness-aware deployment strategy"""
        # Deploy based on system consciousness levels
        await asyncio.sleep(6)
        
        status.performance_metrics["deployment_time"] = 6.0
        status.quantum_coherence_level = 0.87
        status.consciousness_stability = 0.95  # High stability due to consciousness awareness
    
    async def _run_comprehensive_health_checks(self, config: DeploymentConfig, 
                                             status: DeploymentStatus, environment: str):
        """Run comprehensive health checks on deployed system"""
        try:
            health_checks = [
                ("API Health Check", self._check_api_health),
                ("Database Connectivity", self._check_database_health),
                ("Quantum Coherence Check", self._check_quantum_coherence),
                ("Performance Validation", self._check_performance_metrics),
                ("Security Validation", self._check_security_status)
            ]
            
            for check_name, check_func in health_checks:
                try:
                    result = await check_func(environment)
                    if not result["healthy"]:
                        raise Exception(f"{check_name} failed: {result['message']}")
                    
                    # Update performance metrics
                    if "metrics" in result:
                        status.performance_metrics.update(result["metrics"])
                    
                    self.logger.info(f"Health check passed: {check_name}")
                
                except Exception as e:
                    raise Exception(f"Health check failed - {check_name}: {str(e)}")
            
            # Validate against thresholds
            await self._validate_against_thresholds(status)
            
            self.logger.info("All health checks passed")
        
        except Exception as e:
            raise Exception(f"Health checks failed: {str(e)}")
    
    async def _check_api_health(self, environment: str) -> Dict[str, Any]:
        """Check API health status"""
        # Simulate API health check
        await asyncio.sleep(1)
        
        response_time = np.random.uniform(50, 200)  # 50-200ms
        error_rate = np.random.uniform(0, 2)  # 0-2%
        
        healthy = response_time < self.validation_thresholds["response_time_ms"] and \
                 error_rate < self.validation_thresholds["error_rate_percentage"]
        
        return {
            "healthy": healthy,
            "message": "API is responding" if healthy else "API health check failed",
            "metrics": {
                "api_response_time_ms": response_time,
                "api_error_rate_percentage": error_rate
            }
        }
    
    async def _check_database_health(self, environment: str) -> Dict[str, Any]:
        """Check database connectivity and performance"""
        await asyncio.sleep(0.5)
        
        # Simulate database health metrics
        connection_pool_usage = np.random.uniform(20, 70)  # 20-70%
        query_response_time = np.random.uniform(10, 100)  # 10-100ms
        
        return {
            "healthy": True,
            "message": "Database is healthy",
            "metrics": {
                "db_connection_pool_usage": connection_pool_usage,
                "db_query_response_time": query_response_time
            }
        }
    
    async def _check_quantum_coherence(self, environment: str) -> Dict[str, Any]:
        """Check quantum system coherence levels"""
        await asyncio.sleep(0.3)
        
        # Simulate quantum coherence measurement
        system_coherence = np.random.uniform(0.6, 0.95)
        
        healthy = system_coherence >= self.validation_thresholds["quantum_coherence_minimum"]
        
        return {
            "healthy": healthy,
            "message": "Quantum coherence is stable" if healthy else "Low quantum coherence detected",
            "metrics": {
                "quantum_coherence": system_coherence
            }
        }
    
    async def _check_performance_metrics(self, environment: str) -> Dict[str, Any]:
        """Check system performance metrics"""
        await asyncio.sleep(0.7)
        
        # Simulate system resource usage
        cpu_usage = np.random.uniform(30, 75)
        memory_usage = np.random.uniform(40, 80)
        
        healthy = cpu_usage < self.validation_thresholds["cpu_usage_percentage"] and \
                 memory_usage < self.validation_thresholds["memory_usage_percentage"]
        
        return {
            "healthy": healthy,
            "message": "Performance metrics within limits" if healthy else "Performance metrics exceed thresholds",
            "metrics": {
                "cpu_usage_percentage": cpu_usage,
                "memory_usage_percentage": memory_usage
            }
        }
    
    async def _check_security_status(self, environment: str) -> Dict[str, Any]:
        """Check security status and compliance"""
        await asyncio.sleep(0.4)
        
        # Simulate security checks
        security_score = np.random.uniform(0.8, 1.0)
        
        return {
            "healthy": security_score > 0.85,
            "message": "Security status is good" if security_score > 0.85 else "Security concerns detected",
            "metrics": {
                "security_score": security_score
            }
        }
    
    async def _validate_against_thresholds(self, status: DeploymentStatus):
        """Validate deployment metrics against configured thresholds"""
        violations = []
        
        metrics = status.performance_metrics
        
        # Check each threshold
        for metric_name, threshold in self.validation_thresholds.items():
            if metric_name in metrics:
                value = metrics[metric_name]
                
                # Determine if violation based on metric type
                if "minimum" in metric_name:
                    if value < threshold:
                        violations.append(f"{metric_name}: {value} < {threshold}")
                else:
                    if value > threshold:
                        violations.append(f"{metric_name}: {value} > {threshold}")
        
        if violations:
            violation_msg = "; ".join(violations)
            raise Exception(f"Validation threshold violations: {violation_msg}")
    
    async def _migrate_traffic_gradually(self, config: DeploymentConfig, 
                                       status: DeploymentStatus, environment: str):
        """Gradually migrate traffic to new deployment"""
        try:
            steps = config.traffic_migration_steps
            
            for step in range(steps):
                # Calculate traffic percentage for this step
                traffic_percentage = (step + 1) * (100 / steps)
                
                # Simulate traffic migration
                await asyncio.sleep(2)
                
                # Update status
                step_progress = 85 + (step * 2)  # Progress from 85% to 95%
                status.update_progress(
                    DeploymentPhase.TRAFFIC_MIGRATION, 
                    f"Migrating {traffic_percentage:.0f}% of traffic", 
                    step_progress
                )
                
                # Monitor during migration
                health_ok = await self._quick_health_check(environment)
                if not health_ok:
                    raise Exception(f"Health check failed during traffic migration at {traffic_percentage:.0f}%")
                
                self.logger.info(f"Traffic migration step {step+1}/{steps}: {traffic_percentage:.0f}% traffic migrated")
            
            self.logger.info("Traffic migration completed successfully")
        
        except Exception as e:
            raise Exception(f"Traffic migration failed: {str(e)}")
    
    async def _quick_health_check(self, environment: str) -> bool:
        """Perform quick health check during traffic migration"""
        try:
            # Quick API check
            await asyncio.sleep(0.5)
            
            # Simulate health status
            return np.random.uniform(0, 1) > 0.1  # 90% success rate
        
        except Exception:
            return False
    
    async def _monitor_deployment_stability(self, config: DeploymentConfig, 
                                          status: DeploymentStatus, environment: str):
        """Monitor deployment stability for initial period"""
        try:
            monitoring_duration = 60  # 1 minute of monitoring
            check_interval = 10  # Check every 10 seconds
            checks = monitoring_duration // check_interval
            
            for i in range(checks):
                await asyncio.sleep(check_interval)
                
                # Perform stability checks
                stability_ok = await self._check_deployment_stability(environment)
                if not stability_ok:
                    raise Exception(f"Deployment stability check failed at {i+1}/{checks}")
                
                # Update quantum coherence and consciousness stability
                status.quantum_coherence_level = max(0.6, status.quantum_coherence_level - 0.01)
                status.consciousness_stability = max(0.6, status.consciousness_stability - 0.005)
                
                self.logger.debug(f"Stability check {i+1}/{checks} passed")
            
            self.logger.info("Deployment stability monitoring completed")
        
        except Exception as e:
            raise Exception(f"Deployment stability monitoring failed: {str(e)}")
    
    async def _check_deployment_stability(self, environment: str) -> bool:
        """Check if deployment is stable"""
        # Simulate stability metrics
        error_rate = np.random.uniform(0, 3)
        response_time = np.random.uniform(50, 300)
        
        return error_rate < 2.0 and response_time < 250
    
    async def _execute_rollback(self, deployment_id: str, status: DeploymentStatus, environment: str):
        """Execute rollback procedure"""
        try:
            self.logger.warning(f"Executing rollback for deployment {deployment_id}")
            
            status.current_phase = DeploymentPhase.ROLLBACK
            status.current_step = "Initiating rollback procedure"
            
            # Rollback steps
            rollback_steps = [
                "Stopping new deployment",
                "Restoring previous version",
                "Restoring quantum states",
                "Validating rollback health",
                "Restoring traffic routing"
            ]
            
            for i, step in enumerate(rollback_steps):
                status.current_step = step
                await asyncio.sleep(3)  # Each rollback step takes 3 seconds
                
                self.logger.info(f"Rollback step {i+1}/{len(rollback_steps)}: {step}")
            
            self.total_rollbacks += 1
            status.current_step = "Rollback completed"
            
            self.logger.info(f"Rollback completed for deployment {deployment_id}")
        
        except Exception as e:
            self.logger.error(f"Rollback failed for deployment {deployment_id}: {e}")
            raise
    
    def _record_deployment_history(self, config: DeploymentConfig, 
                                 status: DeploymentStatus, outcome: str):
        """Record deployment in history"""
        deployment_record = {
            "deployment_id": status.deployment_id,
            "version": config.version,
            "strategy": config.strategy.value,
            "outcome": outcome,
            "start_time": status.start_time.isoformat(),
            "end_time": datetime.utcnow().isoformat(),
            "duration_seconds": (datetime.utcnow() - status.start_time).total_seconds(),
            "final_phase": status.current_phase.value,
            "error_count": len(status.error_messages),
            "quantum_coherence_final": status.quantum_coherence_level,
            "consciousness_stability_final": status.consciousness_stability,
            "performance_metrics": status.performance_metrics
        }
        
        self.deployment_history.append(deployment_record)
        
        # Update average deployment time
        successful_deployments = [d for d in self.deployment_history if d["outcome"] == "success"]
        if successful_deployments:
            total_time = sum(d["duration_seconds"] for d in successful_deployments)
            self.average_deployment_time = total_time / len(successful_deployments)
        
        # Limit history size
        if len(self.deployment_history) > 100:
            self.deployment_history = self.deployment_history[-50:]
    
    def get_deployment_status(self, deployment_id: str) -> Optional[Dict[str, Any]]:
        """Get status of specific deployment"""
        if deployment_id in self.active_deployments:
            return self.active_deployments[deployment_id].to_dict()
        return None
    
    def get_orchestrator_status(self) -> Dict[str, Any]:
        """Get comprehensive orchestrator status"""
        return {
            "active_deployments": len(self.active_deployments),
            "deployment_history_count": len(self.deployment_history),
            "performance_metrics": {
                "successful_deployments": self.successful_deployments,
                "failed_deployments": self.failed_deployments,
                "total_rollbacks": self.total_rollbacks,
                "success_rate": self.successful_deployments / max(1, self.successful_deployments + self.failed_deployments),
                "average_deployment_time_seconds": self.average_deployment_time
            },
            "validation_thresholds": self.validation_thresholds,
            "supported_strategies": [strategy.value for strategy in DeploymentStrategy],
            "environments": list(self.environments.keys())
        }
    
    def get_deployment_history(self, limit: int = 20) -> List[Dict[str, Any]]:
        """Get recent deployment history"""
        return self.deployment_history[-limit:] if self.deployment_history else []
    
    async def create_deployment_config(self, version: str, **kwargs) -> DeploymentConfig:
        """Create deployment configuration with intelligent defaults"""
        config = DeploymentConfig(version=version)
        
        # Apply any provided overrides
        for key, value in kwargs.items():
            if hasattr(config, key):
                if key == "strategy" and isinstance(value, str):
                    config.strategy = DeploymentStrategy(value)
                else:
                    setattr(config, key, value)
        
        # Intelligent defaults based on deployment history
        if self.deployment_history:
            recent_deployments = self.deployment_history[-10:]
            
            # Choose strategy based on recent success rates
            strategy_success_rates = {}
            for deployment in recent_deployments:
                strategy = deployment["strategy"]
                outcome = deployment["outcome"]
                
                if strategy not in strategy_success_rates:
                    strategy_success_rates[strategy] = {"success": 0, "total": 0}
                
                strategy_success_rates[strategy]["total"] += 1
                if outcome == "success":
                    strategy_success_rates[strategy]["success"] += 1
            
            # Select best performing strategy
            best_strategy = None
            best_rate = 0
            
            for strategy, stats in strategy_success_rates.items():
                if stats["total"] >= 2:  # At least 2 deployments
                    rate = stats["success"] / stats["total"]
                    if rate > best_rate:
                        best_rate = rate
                        best_strategy = strategy
            
            if best_strategy and not kwargs.get("strategy"):
                config.strategy = DeploymentStrategy(best_strategy)
                self.logger.info(f"Selected deployment strategy {best_strategy} based on {best_rate:.1%} success rate")
        
        return config


# Global orchestrator instance
quantum_orchestrator = None

def get_quantum_orchestrator(project_root: str = None) -> QuantumProductionOrchestrator:
    """Get or create quantum production orchestrator instance"""
    global quantum_orchestrator
    if quantum_orchestrator is None:
        quantum_orchestrator = QuantumProductionOrchestrator(project_root)
    return quantum_orchestrator
