"""
Comprehensive test suite for the entire Quantum Task Planner system
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from quantum_task_planner.core.quantum_task import QuantumTask, TaskState, TaskPriority
from quantum_task_planner.core.quantum_scheduler import QuantumTaskScheduler
from quantum_task_planner.core.quantum_optimizer import QuantumProbabilityOptimizer
from quantum_task_planner.core.entanglement_manager import TaskEntanglementManager, EntanglementType

from quantum_task_planner.utils.security import SecurityManager, create_default_security_config
from quantum_task_planner.utils.health_checks import HealthCheckManager, SystemResourcesHealthCheck
from quantum_task_planner.utils.logging import QuantumLogger
from quantum_task_planner.utils.validation import TaskValidator, QuantumValidator

from quantum_task_planner.performance.cache import QuantumCache
from quantum_task_planner.performance.concurrent import QuantumWorkerPool
from quantum_task_planner.performance.scaling import AutoScaler, QuantumLoadBalancer

from quantum_task_planner.distributed.quantum_sync import QuantumStateTracker, DistributedQuantumCoordinator


class TestQuantumTaskCore:
    """Test core quantum task functionality"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.task = QuantumTask(
            title="Test Task",
            description="A test quantum task",
            priority=TaskPriority.HIGH,
            complexity_factor=2.0
        )
    
    def test_quantum_task_creation(self):
        """Test quantum task creation and initial state"""
        assert self.task.title == "Test Task"
        assert self.task.priority == TaskPriority.HIGH
        assert self.task.state == TaskState.PENDING
        assert 0.7 <= self.task.quantum_coherence <= 1.0
        assert self.task.task_id is not None
    
    def test_quantum_measurement(self):
        """Test quantum state measurement"""
        initial_coherence = self.task.quantum_coherence
        measurement = self.task.measure_quantum_state()
        
        assert measurement.measured_state in [s.value for s in TaskState]
        assert 0.0 < measurement.measurement_probability <= 1.0
        assert self.task.quantum_coherence < initial_coherence
    
    def test_quantum_decoherence(self):
        """Test quantum decoherence over time"""
        initial_coherence = self.task.quantum_coherence
        self.task.apply_decoherence(60.0)  # 1 minute
        
        assert self.task.quantum_coherence < initial_coherence
        assert self.task.quantum_coherence >= 0.0


class TestQuantumScheduler:
    """Test quantum task scheduler"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.scheduler = QuantumTaskScheduler()
        
        # Create test tasks
        self.tasks = [
            QuantumTask(
                title=f"Task {i}",
                description=f"Test task {i}",
                priority=TaskPriority.HIGH if i % 2 == 0 else TaskPriority.LOW,
                complexity_factor=float(i + 1)
            )
            for i in range(5)
        ]
    
    def test_add_tasks(self):
        """Test adding tasks to scheduler"""
        for task in self.tasks:
            self.scheduler.add_task(task)
        
        assert len(self.scheduler.tasks) == 5
        assert all(tid in self.scheduler.tasks for tid in [t.task_id for t in self.tasks])
    
    def test_quantum_scheduling_algorithm(self):
        """Test quantum annealing scheduling"""
        for task in self.tasks:
            self.scheduler.add_task(task)
        
        # Run quantum scheduling
        schedule = self.scheduler.schedule_tasks()
        
        assert isinstance(schedule, list)
        assert len(schedule) <= len(self.tasks)
        
        # Verify schedule order respects quantum optimization
        if len(schedule) > 1:
            # Higher priority tasks should generally come first
            high_priority_positions = [
                i for i, task_id in enumerate(schedule)
                if self.scheduler.tasks[task_id].priority == TaskPriority.HIGH
            ]
            low_priority_positions = [
                i for i, task_id in enumerate(schedule)
                if self.scheduler.tasks[task_id].priority == TaskPriority.LOW
            ]
            
            if high_priority_positions and low_priority_positions:
                assert min(high_priority_positions) <= max(low_priority_positions)
    
    @pytest.mark.asyncio
    async def test_async_scheduling(self):
        """Test asynchronous scheduling operations"""
        for task in self.tasks:
            self.scheduler.add_task(task)
        
        schedule = await self.scheduler.schedule_tasks_async()
        assert isinstance(schedule, list)
        assert len(schedule) >= 0


class TestQuantumOptimizer:
    """Test quantum optimization algorithms"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.optimizer = QuantumProbabilityOptimizer()
        self.tasks = [
            QuantumTask(
                title=f"Task {i}",
                description=f"Optimization test task {i}",
                priority=TaskPriority.MEDIUM,
                complexity_factor=float(i + 1) * 0.5
            )
            for i in range(3)
        ]
    
    def test_genetic_algorithm_optimization(self):
        """Test genetic algorithm optimization"""
        # Add objectives
        objectives = self.optimizer.create_standard_objectives()
        for obj in objectives:
            self.optimizer.add_objective(obj)
        
        # Run optimization
        results = self.optimizer.optimize_genetic_algorithm(
            self.tasks, 
            max_iterations=10,
            population_size=20
        )
        
        assert isinstance(results, dict)
        assert "best_fitness" in results
        assert "optimization_history" in results
        assert "final_allocation" in results
        
        # Fitness should be valid
        assert 0.0 <= results["best_fitness"] <= 100.0
    
    @pytest.mark.asyncio
    async def test_async_optimization(self):
        """Test asynchronous optimization"""
        task_ids = [task.task_id for task in self.tasks]
        
        results = await self.optimizer.optimize_async(task_ids)
        
        assert isinstance(results, dict)
        assert "improvement" in results or "error" in results


class TestEntanglementManager:
    """Test quantum entanglement management"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.entanglement_manager = TaskEntanglementManager()
        self.tasks = [
            QuantumTask(
                title=f"Entangled Task {i}",
                description=f"Task for entanglement testing {i}",
                priority=TaskPriority.MEDIUM
            )
            for i in range(4)
        ]
    
    def test_create_entanglement(self):
        """Test creating quantum entanglement between tasks"""
        task_ids = [task.task_id for task in self.tasks[:2]]
        
        bond = self.entanglement_manager.create_entanglement_bond(
            task_ids,
            EntanglementType.BELL_STATE,
            strength=0.8
        )
        
        assert bond is not None
        assert bond.entanglement_type == EntanglementType.BELL_STATE
        assert bond.strength == 0.8
        assert len(bond.task_ids) == 2
        assert bond.bond_id in self.entanglement_manager.entanglement_bonds
    
    def test_entanglement_correlation(self):
        """Test entanglement correlation effects"""
        task_ids = [task.task_id for task in self.tasks[:2]]
        
        # Create entanglement
        bond = self.entanglement_manager.create_entanglement_bond(
            task_ids,
            EntanglementType.CORRELATION,
            strength=0.9
        )
        
        # Measure one task
        initial_coherence_0 = self.tasks[0].quantum_coherence
        initial_coherence_1 = self.tasks[1].quantum_coherence
        
        self.tasks[0].measure_quantum_state()
        
        # Apply correlation effect
        self.entanglement_manager.apply_entanglement_correlation(bond.bond_id, self.tasks)
        
        # Both tasks should be affected due to entanglement
        assert self.tasks[0].quantum_coherence < initial_coherence_0
        assert self.tasks[1].quantum_coherence != initial_coherence_1  # Should be correlated
    
    @pytest.mark.asyncio
    async def test_decoherence_application(self):
        """Test quantum decoherence application"""
        # Create some entangled tasks
        for i in range(0, len(self.tasks), 2):
            if i + 1 < len(self.tasks):
                task_ids = [self.tasks[i].task_id, self.tasks[i+1].task_id]
                self.entanglement_manager.create_entanglement_bond(
                    task_ids, EntanglementType.BELL_STATE
                )
        
        initial_bond_count = len(self.entanglement_manager.entanglement_bonds)
        
        # Apply decoherence
        await self.entanglement_manager.apply_decoherence(300.0)  # 5 minutes
        
        # Some bonds might have broken due to decoherence
        final_bond_count = len(self.entanglement_manager.entanglement_bonds)
        assert final_bond_count <= initial_bond_count


class TestSecurityFeatures:
    """Test security and authentication features"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.security_config = create_default_security_config()
        self.security_manager = SecurityManager(self.security_config)
    
    def test_password_validation(self):
        """Test password strength validation"""
        # Valid password
        valid_password = "StrongP@ssw0rd123"
        assert self.security_manager.password_validator.validate_password_strength(valid_password)
        
        # Weak password
        with pytest.raises(Exception):  # Should raise validation error
            self.security_manager.password_validator.validate_password_strength("weak")
    
    def test_jwt_token_management(self):
        """Test JWT token creation and verification"""
        user_id = "test_user"
        permissions = ["read", "write"]
        
        # Create token
        token = self.security_manager.jwt_manager.create_token(user_id, permissions)
        assert isinstance(token, str)
        assert len(token) > 10
        
        # Verify token
        payload = self.security_manager.jwt_manager.verify_token(token)
        assert payload["user_id"] == user_id
        assert payload["permissions"] == permissions
    
    def test_rate_limiting(self):
        """Test rate limiting functionality"""
        client_ip = "192.168.1.1"
        
        # Should not be rate limited initially
        assert not self.security_manager.rate_limiter.is_rate_limited(client_ip)
        
        # Exceed rate limit
        for _ in range(self.security_config.rate_limit_requests + 1):
            self.security_manager.rate_limiter.is_rate_limited(client_ip)
        
        # Should now be rate limited
        assert self.security_manager.rate_limiter.is_rate_limited(client_ip)
    
    def test_input_sanitization(self):
        """Test input sanitization"""
        malicious_input = "<script>alert('xss')</script>"
        sanitized = self.security_manager.sanitize_input(malicious_input, "html")
        
        assert "<script>" not in sanitized
        assert "alert" not in sanitized


class TestHealthChecks:
    """Test health check and monitoring system"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.health_manager = HealthCheckManager()
    
    def test_system_resources_health_check(self):
        """Test system resources health check"""
        health_check = SystemResourcesHealthCheck()
        self.health_manager.add_health_check(health_check)
        
        assert len(self.health_manager.health_checks) == 1
        assert "system_resources" in self.health_manager.health_checks
    
    @pytest.mark.asyncio
    async def test_health_check_execution(self):
        """Test health check execution"""
        health_check = SystemResourcesHealthCheck()
        self.health_manager.add_health_check(health_check)
        
        results = await self.health_manager.check_all_health()
        
        assert isinstance(results, dict)
        assert "system_resources" in results
        
        result = results["system_resources"]
        assert hasattr(result, 'status')
        assert hasattr(result, 'message')
        assert hasattr(result, 'timestamp')


class TestPerformanceFeatures:
    """Test performance optimization features"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.cache = QuantumCache()
        self.worker_pool = QuantumWorkerPool(max_workers=4, worker_type="thread")
    
    def test_quantum_cache(self):
        """Test quantum-aware caching"""
        # Set cache entry
        key = "test_key"
        value = {"data": "test_value", "quantum_coherence": 0.8}
        
        self.cache.set(key, value, quantum_coherence=0.8)
        
        # Get cache entry
        cached_value = self.cache.get(key)
        assert cached_value == value
        
        # Check cache statistics
        stats = self.cache.get_stats()
        assert stats["entries"] == 1
        assert stats["hit_rate"] > 0.0
    
    @pytest.mark.asyncio
    async def test_quantum_worker_pool(self):
        """Test quantum-aware worker pool"""
        def test_task(x, y):
            return x + y
        
        # Submit quantum task
        task_id = self.worker_pool.submit_quantum_task(
            test_task, 5, 10,
            quantum_coherence=0.9,
            priority=2.0
        )
        
        assert isinstance(task_id, str)
        
        # Get result
        result = self.worker_pool.get_task_result(task_id, timeout=5.0)
        assert result == 15
        
        # Check pool statistics
        stats = self.worker_pool.get_pool_stats()
        assert stats["total_tasks_processed"] > 0
    
    def test_auto_scaler(self):
        """Test auto-scaling functionality"""
        auto_scaler = AutoScaler()
        
        # Check initial state
        status = auto_scaler.get_scaling_status()
        assert status["current_instances"] >= 1
        assert isinstance(status["rules_enabled"], int)
        
        # Check scaling rules
        assert len(auto_scaler.scaling_rules) > 0
        
        # All rules should have valid thresholds
        for rule in auto_scaler.scaling_rules:
            assert rule.threshold_up > rule.threshold_down
            assert rule.min_instances <= rule.max_instances
    
    def test_load_balancer(self):
        """Test quantum load balancer"""
        load_balancer = QuantumLoadBalancer()
        
        # Register workers
        workers = [
            ("worker_1", "http://localhost:8001", 0.9),
            ("worker_2", "http://localhost:8002", 0.8),
            ("worker_3", "http://localhost:8003", 0.7)
        ]
        
        for worker_id, endpoint, coherence in workers:
            load_balancer.register_worker(worker_id, endpoint, coherence)
        
        # Select worker for task
        selected_worker = load_balancer.select_worker(task_quantum_coherence=0.85)
        assert selected_worker in ["worker_1", "worker_2", "worker_3"]
        
        # Check load distribution
        distribution = load_balancer.get_load_distribution()
        assert distribution["workers"] == 3


class TestDistributedFeatures:
    """Test distributed quantum coordination"""
    
    def setup_method(self):
        """Setup test fixtures"""
        self.node_id = "test_node_1"
        self.state_tracker = QuantumStateTracker(self.node_id)
        self.coordinator = DistributedQuantumCoordinator(self.node_id)
    
    def test_quantum_state_tracking(self):
        """Test distributed quantum state tracking"""
        task_id = "test_task_1"
        coherence = 0.8
        state_probs = {"pending": 0.7, "in_progress": 0.3}
        
        # Update local state
        self.state_tracker.update_local_state(
            task_id, coherence, state_probs
        )
        
        assert task_id in self.state_tracker.local_states
        
        snapshot = self.state_tracker.local_states[task_id]
        assert snapshot.quantum_coherence == coherence
        assert snapshot.state_probabilities == state_probs
    
    def test_distributed_state_consensus(self):
        """Test distributed state consensus"""
        task_id = "consensus_test_task"
        
        # Add local state
        self.state_tracker.update_local_state(
            task_id, 0.8, {"pending": 1.0}
        )
        
        # Simulate remote state
        from quantum_task_planner.distributed.quantum_sync import QuantumStateSnapshot
        remote_snapshot = QuantumStateSnapshot(
            node_id="remote_node",
            task_id=task_id,
            quantum_coherence=0.9,
            state_probabilities={"pending": 0.5, "in_progress": 0.5},
            entanglement_bonds=[],
            timestamp=datetime.utcnow()
        )
        
        self.state_tracker.remote_states["remote_node"][task_id] = remote_snapshot
        
        # Get distributed state
        distributed_state = self.state_tracker.get_distributed_state(task_id)
        
        assert distributed_state is not None
        assert distributed_state["node_count"] == 2
        assert 0.8 <= distributed_state["consensus_coherence"] <= 0.9
    
    @pytest.mark.asyncio
    async def test_distributed_operation(self):
        """Test distributed quantum operation execution"""
        task_ids = ["task_1", "task_2"]
        operation_params = {"strength": 0.8, "type": "correlation"}
        
        # Mock the cluster joining
        with patch.object(self.coordinator, '_broadcast_message', new_callable=AsyncMock):
            await self.coordinator.join_cluster()
        
        # Execute distributed entanglement operation
        with patch.object(self.coordinator, '_collect_node_responses', new_callable=AsyncMock) as mock_collect:
            mock_collect.return_value = {"node_1": {"prepared": True}}
            
            result = await self.coordinator.execute_distributed_operation(
                "quantum_entanglement",
                task_ids,
                operation_params
            )
            
            assert result["success"]
            assert "operation_id" in result


class TestValidationSystem:
    """Test comprehensive validation system"""
    
    def test_task_validation(self):
        """Test task field validation"""
        # Valid task data
        TaskValidator.validate_title("Valid Task Title")
        TaskValidator.validate_description("A valid task description")
        TaskValidator.validate_priority("high")
        TaskValidator.validate_complexity_factor(2.5)
        
        # Invalid task data
        with pytest.raises(Exception):
            TaskValidator.validate_title("")  # Empty title
        
        with pytest.raises(Exception):
            TaskValidator.validate_priority("invalid")  # Invalid priority
        
        with pytest.raises(Exception):
            TaskValidator.validate_complexity_factor(0.0)  # Invalid complexity
    
    def test_quantum_validation(self):
        """Test quantum-specific validation"""
        # Valid quantum values
        QuantumValidator.validate_quantum_coherence(0.8)
        QuantumValidator.validate_probability_amplitude(0.5 + 0.3j)
        QuantumValidator.validate_state_probabilities({"state1": 0.7, "state2": 0.3})
        QuantumValidator.validate_entanglement_strength(0.9)
        
        # Invalid quantum values
        with pytest.raises(Exception):
            QuantumValidator.validate_quantum_coherence(1.5)  # > 1.0
        
        with pytest.raises(Exception):
            QuantumValidator.validate_state_probabilities({"state1": 1.5})  # > 1.0
        
        with pytest.raises(Exception):
            QuantumValidator.validate_entanglement_strength(-0.1)  # < 0.0


class TestIntegrationScenarios:
    """Test end-to-end integration scenarios"""
    
    def setup_method(self):
        """Setup comprehensive test environment"""
        self.scheduler = QuantumTaskScheduler()
        self.optimizer = QuantumProbabilityOptimizer()
        self.entanglement_manager = TaskEntanglementManager()
        self.security_manager = SecurityManager(create_default_security_config())
    
    def test_complete_task_workflow(self):
        """Test complete task creation, scheduling, and optimization workflow"""
        # Create tasks
        tasks = [
            QuantumTask(
                title=f"Workflow Task {i}",
                description=f"Integration test task {i}",
                priority=TaskPriority.HIGH if i % 2 == 0 else TaskPriority.MEDIUM,
                complexity_factor=float(i + 1)
            )
            for i in range(3)
        ]
        
        # Add tasks to scheduler
        for task in tasks:
            self.scheduler.add_task(task)
        
        assert len(self.scheduler.tasks) == 3
        
        # Create entanglement between first two tasks
        entangled_task_ids = [tasks[0].task_id, tasks[1].task_id]
        bond = self.entanglement_manager.create_entanglement_bond(
            entangled_task_ids,
            EntanglementType.BELL_STATE
        )
        
        assert bond is not None
        
        # Schedule tasks
        schedule = self.scheduler.schedule_tasks()
        assert len(schedule) <= 3
        
        # Add optimization objectives
        objectives = self.optimizer.create_standard_objectives()
        for obj in objectives:
            self.optimizer.add_objective(obj)
        
        # Run optimization
        results = self.optimizer.optimize_genetic_algorithm(
            tasks, max_iterations=5, population_size=10
        )
        
        assert "best_fitness" in results
        assert results["best_fitness"] >= 0.0
    
    @pytest.mark.asyncio
    async def test_async_integration_workflow(self):
        """Test asynchronous integration workflow"""
        # Create task
        task = QuantumTask(
            title="Async Integration Task",
            description="Test async workflow",
            priority=TaskPriority.HIGH
        )
        
        self.scheduler.add_task(task)
        
        # Async scheduling
        schedule = await self.scheduler.schedule_tasks_async()
        assert isinstance(schedule, list)
        
        # Async optimization
        results = await self.optimizer.optimize_async([task.task_id])
        assert isinstance(results, dict)
        
        # Async decoherence
        await self.entanglement_manager.apply_decoherence(30.0)
        
        # Verify task state after async operations
        assert task.quantum_coherence >= 0.0
        assert task.quantum_coherence <= 1.0


if __name__ == "__main__":
    pytest.main([
        __file__, 
        "-v", 
        "--cov=quantum_task_planner",
        "--cov-report=html",
        "--cov-report=term-missing",
        "--cov-fail-under=85"
    ])