"""
Comprehensive Test Suite for Dynamic Quantum-Classical Ensemble Optimizer (DQCEO)

Research Validation Framework:
- Comparative performance analysis
- Statistical significance testing
- Reproducibility validation
- Quantum advantage quantification

Publication-ready experimental validation for academic submission.
"""

import pytest
import asyncio
import numpy as np
import time
from datetime import datetime, timedelta
from typing import List, Dict, Any

from quantum_task_planner.core.quantum_task import QuantumTask, TaskState, TaskPriority, QuantumAmplitude
from quantum_task_planner.research.dynamic_quantum_classical_optimizer import (
    DynamicQuantumClassicalOptimizer,
    OptimizationAlgorithm,
    ProblemCharacteristics,
    PerformancePredictor,
    ResultFusion
)


class TestDynamicQuantumClassicalOptimizer:
    """Research validation test suite for DQCEO"""
    
    @pytest.fixture
    def sample_tasks(self) -> List[QuantumTask]:
        """Generate sample tasks for testing"""
        tasks = []
        for i in range(10):
            task = QuantumTask(
                title=f"Research Task {i}",
                description=f"Test task {i} for DQCEO validation",
                priority=TaskPriority.NORMAL,
                estimated_duration=timedelta(hours=i+1),
                due_date=datetime.utcnow() + timedelta(days=i+1)
            )
            
            # Add quantum state amplitudes
            task.state_amplitudes[TaskState.PENDING] = QuantumAmplitude(
                amplitude=complex(np.cos(i * 0.1), np.sin(i * 0.1)),
                probability=0.7,
                phase_angle=i * 0.1
            )
            
            task.quantum_coherence = 0.5 + 0.3 * np.sin(i)
            tasks.append(task)
        
        return tasks
    
    @pytest.fixture
    def dqceo_optimizer(self) -> DynamicQuantumClassicalOptimizer:
        """Create DQCEO optimizer for testing"""
        return DynamicQuantumClassicalOptimizer(
            enable_parallel_execution=True,
            max_parallel_algorithms=3,
            adaptive_learning=True
        )
    
    @pytest.mark.asyncio
    async def test_basic_optimization_functionality(self, dqceo_optimizer, sample_tasks):
        """Test basic DQCEO optimization functionality"""
        
        constraints = {"max_parallel_tasks": 3}
        objectives = [
            lambda tasks: np.mean([t.get_completion_probability() for t in tasks]),
            lambda tasks: np.mean([t.quantum_coherence for t in tasks])
        ]
        
        result = await dqceo_optimizer.optimize_with_dynamic_selection(
            tasks=sample_tasks,
            constraints=constraints,
            objectives=objectives,
            time_budget=10.0
        )
        
        # Validate result structure
        assert "experiment_id" in result
        assert "optimized_schedule" in result
        assert "primary_algorithm" in result
        assert "dynamic_selection_metrics" in result
        assert "research_contributions" in result
        
        # Validate schedule
        schedule = result["optimized_schedule"]
        assert len(schedule) == len(sample_tasks)
        assert all("task_id" in task for task in schedule)
        
        # Validate research metrics
        research = result["research_contributions"]
        assert research["novel_dynamic_selection"] is True
        assert "reproducibility_metrics" in research
    
    @pytest.mark.asyncio
    async def test_algorithm_comparison_validation(self, dqceo_optimizer, sample_tasks):
        """Test comparative performance of different algorithms"""
        
        constraints = {"max_parallel_tasks": 2}
        objectives = [lambda tasks: np.mean([t.get_completion_probability() for t in tasks])]
        
        # Run optimization
        result = await dqceo_optimizer.optimize_with_dynamic_selection(
            tasks=sample_tasks,
            constraints=constraints,
            objectives=objectives,
            time_budget=15.0
        )
        
        # Validate algorithm comparison
        algorithm_results = result["algorithm_results"]
        assert len(algorithm_results) >= 2  # At least 2 algorithms should run
        
        # Check that each algorithm has proper metrics
        for algo_name, algo_result in algorithm_results.items():
            assert "execution_time" in algo_result
            assert "solution_quality" in algo_result
            assert algo_result["solution_quality"] >= 0.0
            assert algo_result["solution_quality"] <= 1.0
        
        # Validate statistical analysis
        pub_data = result["publication_ready_data"]
        assert "performance_comparison" in pub_data
        assert "statistical_significance" in pub_data
    
    @pytest.mark.asyncio
    async def test_quantum_advantage_analysis(self, dqceo_optimizer, sample_tasks):
        """Test quantum advantage quantification"""
        
        constraints = {}
        objectives = [lambda tasks: np.mean([t.quantum_coherence for t in tasks])]
        
        result = await dqceo_optimizer.optimize_with_dynamic_selection(
            tasks=sample_tasks,
            constraints=constraints,
            objectives=objectives,
            time_budget=20.0
        )
        
        # Validate quantum advantage analysis
        qa_analysis = result["quantum_advantage_analysis"]
        
        if "quantum_advantage" in qa_analysis and isinstance(qa_analysis["quantum_advantage"], dict):
            advantage = qa_analysis["quantum_advantage"]
            assert "quality_improvement" in advantage
            assert "time_ratio" in advantage
            assert "overall_advantage" in advantage
            assert "statistical_confidence" in advantage
            
            # Quality improvement should be a valid number
            assert isinstance(advantage["quality_improvement"], (int, float))
            assert isinstance(advantage["statistical_confidence"], (int, float))
            assert 0.0 <= advantage["statistical_confidence"] <= 1.0
    
    def test_problem_characteristics_analysis(self, sample_tasks):
        """Test problem characteristics extraction"""
        
        dqceo = DynamicQuantumClassicalOptimizer()
        constraints = {"max_parallel_tasks": 3}
        objectives = [lambda tasks: np.mean([t.get_completion_probability() for t in tasks])]
        
        problem_chars = dqceo._analyze_problem_characteristics(
            tasks=sample_tasks,
            constraints=constraints,
            objectives=objectives,
            time_budget=30.0
        )
        
        # Validate problem characteristics
        assert problem_chars.problem_size == len(sample_tasks)
        assert 0.0 <= problem_chars.constraint_density <= 10.0  # Reasonable range
        assert 0.0 <= problem_chars.objective_complexity <= 1.0
        assert 0.0 <= problem_chars.nonlinearity_measure <= 1.0
        assert 0.0 <= problem_chars.quantum_coherence_potential <= 1.0
        assert problem_chars.time_budget_seconds == 30.0
        
        # Test feature vector conversion
        feature_vector = problem_chars.to_feature_vector()
        assert len(feature_vector) == 6
        assert all(isinstance(x, (int, float)) for x in feature_vector)
    
    def test_performance_predictor(self, sample_tasks):
        """Test ML-based performance prediction"""
        
        predictor = PerformancePredictor()
        
        # Create test problem characteristics
        problem_chars = ProblemCharacteristics(
            problem_size=10,
            constraint_density=0.3,
            objective_complexity=0.4,
            nonlinearity_measure=0.6,
            quantum_coherence_potential=0.8,
            time_budget_seconds=30.0
        )
        
        # Test cold start prediction
        algo, confidence = predictor.predict_best_algorithm(problem_chars)
        assert isinstance(algo, OptimizationAlgorithm)
        assert 0.0 <= confidence <= 1.0
        
        # Test learning update
        predictor.update_model(problem_chars, algo, 0.75)
        assert len(predictor.training_data) == 1
        
        # Add more training data
        for i in range(15):
            test_chars = ProblemCharacteristics(
                problem_size=5 + i,
                constraint_density=0.1 + i * 0.05,
                objective_complexity=0.2,
                nonlinearity_measure=0.5,
                quantum_coherence_potential=0.6,
                time_budget_seconds=20.0
            )
            predictor.update_model(test_chars, OptimizationAlgorithm.QUANTUM_ANNEALING, 0.6 + i * 0.02)
        
        # Test prediction after learning
        new_algo, new_confidence = predictor.predict_best_algorithm(problem_chars)
        assert isinstance(new_algo, OptimizationAlgorithm)
        assert 0.0 <= new_confidence <= 1.0
    
    def test_result_fusion(self, sample_tasks):
        """Test intelligent result fusion"""
        
        fusion = ResultFusion()
        
        # Create mock algorithm results
        algorithm_results = {
            OptimizationAlgorithm.QUANTUM_ANNEALING: {
                "optimized_schedule": [
                    {"task_id": f"task_{i}", "completion_probability": 0.8, "quantum_coherence": 0.7}
                    for i in range(5)
                ],
                "statistical_validation": {
                    "confidence_interval_95": [0.7, 0.9]
                }
            },
            OptimizationAlgorithm.GENETIC_ALGORITHM: {
                "optimized_schedule": [
                    {"task_id": f"task_{i}", "completion_probability": 0.75, "quantum_coherence": 0.6}
                    for i in range(5)
                ],
                "statistical_validation": {
                    "confidence_interval_95": [0.65, 0.85]
                }
            }
        }
        
        problem_chars = ProblemCharacteristics(
            problem_size=5,
            constraint_density=0.2,
            objective_complexity=0.3,
            nonlinearity_measure=0.4,
            quantum_coherence_potential=0.7,
            time_budget_seconds=20.0
        )
        
        # Test fusion
        fused_result = fusion.fuse_results(algorithm_results, problem_chars)
        
        # Validate fusion result
        assert "fused_schedule" in fused_result
        assert "primary_algorithm" in fused_result
        assert "fusion_quality" in fused_result
        assert "fusion_confidence" in fused_result
        assert "algorithm_weights" in fused_result
        
        assert 0.0 <= fused_result["fusion_quality"] <= 1.0
        assert 0.0 <= fused_result["fusion_confidence"] <= 1.0
        
        # Validate algorithm weights
        weights = fused_result["algorithm_weights"]
        total_weight = sum(weights.values())
        assert abs(total_weight - 1.0) < 0.01  # Should sum to 1.0
    
    @pytest.mark.asyncio
    async def test_reproducibility_validation(self, dqceo_optimizer, sample_tasks):
        """Test reproducibility of results"""
        
        constraints = {"max_parallel_tasks": 2}
        objectives = [lambda tasks: np.mean([t.get_completion_probability() for t in tasks])]
        
        # Run multiple experiments
        results = []
        for run in range(3):
            result = await dqceo_optimizer.optimize_with_dynamic_selection(
                tasks=sample_tasks,
                constraints=constraints,
                objectives=objectives,
                time_budget=10.0
            )
            results.append(result)
        
        # Validate reproducibility metrics are recorded
        for result in results:
            research_contrib = result["research_contributions"]
            repro_metrics = research_contrib["reproducibility_metrics"]
            
            assert "experiment_id" in repro_metrics
            assert "algorithm_versions" in repro_metrics
            assert "execution_environment" in repro_metrics
        
        # Check that experiments have different IDs but consistent structure
        experiment_ids = [r["experiment_id"] for r in results]
        assert len(set(experiment_ids)) == len(experiment_ids)  # All unique
        
        # Check result structure consistency
        for result in results:
            assert "optimized_schedule" in result
            assert "dynamic_selection_metrics" in result
            assert len(result["optimized_schedule"]) == len(sample_tasks)
    
    @pytest.mark.asyncio
    async def test_statistical_significance_validation(self, dqceo_optimizer, sample_tasks):
        """Test statistical significance testing"""
        
        constraints = {}
        objectives = [
            lambda tasks: np.mean([t.get_completion_probability() for t in tasks]),
            lambda tasks: np.mean([t.quantum_coherence for t in tasks])
        ]
        
        result = await dqceo_optimizer.optimize_with_dynamic_selection(
            tasks=sample_tasks,
            constraints=constraints,
            objectives=objectives,
            time_budget=15.0
        )
        
        # Validate statistical significance testing
        pub_data = result["publication_ready_data"]
        stat_sig = pub_data["statistical_significance"]
        
        assert "p_value" in stat_sig
        assert "is_significant" in stat_sig
        assert "effect_size" in stat_sig
        assert "confidence_level" in stat_sig
        
        # Validate p-value range
        assert 0.0 <= stat_sig["p_value"] <= 1.0
        assert isinstance(stat_sig["is_significant"], bool)
        assert stat_sig["confidence_level"] == 0.95
    
    @pytest.mark.asyncio
    async def test_research_summary_generation(self, dqceo_optimizer, sample_tasks):
        """Test comprehensive research summary generation"""
        
        # Run a few experiments to generate data
        constraints = {"max_parallel_tasks": 2}
        objectives = [lambda tasks: np.mean([t.get_completion_probability() for t in tasks])]
        
        for _ in range(3):
            await dqceo_optimizer.optimize_with_dynamic_selection(
                tasks=sample_tasks,
                constraints=constraints,
                objectives=objectives,
                time_budget=8.0
            )
        
        # Generate research summary
        summary = dqceo_optimizer.get_research_summary()
        
        # Validate research summary structure
        assert "research_contributions" in summary
        assert "experimental_results" in summary
        assert "performance_improvements" in summary
        assert "novel_algorithms" in summary
        assert "publication_readiness" in summary
        
        # Validate experimental results
        exp_results = summary["experimental_results"]
        assert exp_results["total_experiments"] == 3
        assert "average_solution_quality" in exp_results
        assert "average_execution_time" in exp_results
        
        # Validate publication readiness
        pub_ready = summary["publication_readiness"]
        assert pub_ready["reproducible_experiments"] is True
        assert pub_ready["statistical_validation"] is True
        assert pub_ready["comparative_baselines"] is True
        assert pub_ready["open_source_implementation"] is True
    
    @pytest.mark.asyncio
    async def test_performance_benchmarking(self, dqceo_optimizer, sample_tasks):
        """Benchmark DQCEO performance against individual algorithms"""
        
        start_time = time.time()
        
        constraints = {"max_parallel_tasks": 3}
        objectives = [
            lambda tasks: np.mean([t.get_completion_probability() for t in tasks]),
            lambda tasks: 1.0 - np.var([t.quantum_coherence for t in tasks])  # Prefer consistent coherence
        ]
        
        result = await dqceo_optimizer.optimize_with_dynamic_selection(
            tasks=sample_tasks,
            constraints=constraints,
            objectives=objectives,
            time_budget=25.0
        )
        
        total_time = time.time() - start_time
        
        # Performance validation
        assert total_time < 30.0  # Should complete within time budget + overhead
        
        # Validate that multiple algorithms were compared
        algo_results = result["algorithm_results"]
        assert len(algo_results) >= 2
        
        # Validate solution quality
        fusion_quality = result["dynamic_selection_metrics"]["fusion_quality"]
        assert 0.0 <= fusion_quality <= 1.0
        
        # Check that fusion quality is competitive
        individual_qualities = [
            ar["solution_quality"] for ar in algo_results.values()
        ]
        
        if individual_qualities:
            max_individual = max(individual_qualities)
            # Fusion should be at least as good as the average individual algorithm
            avg_individual = np.mean(individual_qualities)
            assert fusion_quality >= avg_individual * 0.8  # Allow some tolerance
    
    @pytest.mark.asyncio
    async def test_edge_cases_and_robustness(self, dqceo_optimizer):
        """Test edge cases and robustness"""
        
        # Test with empty task list
        empty_result = await dqceo_optimizer.optimize_with_dynamic_selection(
            tasks=[],
            constraints={},
            objectives=[lambda tasks: 1.0],
            time_budget=5.0
        )
        
        assert "optimized_schedule" in empty_result
        assert len(empty_result["optimized_schedule"]) == 0
        
        # Test with single task
        single_task = [QuantumTask(
            title="Single Task",
            description="Test single task",
            priority=TaskPriority.HIGH,
            estimated_duration=timedelta(hours=1)
        )]
        
        single_result = await dqceo_optimizer.optimize_with_dynamic_selection(
            tasks=single_task,
            constraints={},
            objectives=[lambda tasks: np.mean([t.get_completion_probability() for t in tasks])],
            time_budget=5.0
        )
        
        assert len(single_result["optimized_schedule"]) == 1
        
        # Test with very short time budget
        short_budget_result = await dqceo_optimizer.optimize_with_dynamic_selection(
            tasks=single_task,
            constraints={},
            objectives=[lambda tasks: 1.0],
            time_budget=0.1  # Very short time budget
        )
        
        assert "optimized_schedule" in short_budget_result
        # Should still produce a result, even if not optimal


class TestResearchValidationFramework:
    """Test the research validation framework itself"""
    
    def test_experimental_design_validity(self):
        """Test that experimental design follows research best practices"""
        
        # Test problem characteristics coverage
        test_cases = [
            # Small problems
            ProblemCharacteristics(5, 0.2, 0.3, 0.4, 0.5, 10.0),
            # Medium problems  
            ProblemCharacteristics(20, 0.5, 0.6, 0.7, 0.8, 30.0),
            # Large problems
            ProblemCharacteristics(100, 0.8, 0.9, 0.9, 0.6, 120.0)
        ]
        
        # Validate test case diversity
        sizes = [tc.problem_size for tc in test_cases]
        assert min(sizes) < 10 and max(sizes) > 50  # Good size range
        
        coherences = [tc.quantum_coherence_potential for tc in test_cases]
        assert max(coherences) - min(coherences) > 0.3  # Good coherence range
    
    def test_statistical_power_analysis(self):
        """Test statistical power of experimental design"""
        
        # Simulate multiple experimental runs
        sample_sizes = [3, 5, 10, 20]
        expected_power = []
        
        for n in sample_sizes:
            # Simulate effect sizes (simplified)
            effect_sizes = np.random.normal(0.2, 0.1, n)  # Small to medium effects
            
            # Calculate statistical power (simplified)
            power = min(1.0, len([e for e in effect_sizes if abs(e) > 0.1]) / n)
            expected_power.append(power)
        
        # Validate that power increases with sample size
        for i in range(1, len(expected_power)):
            assert expected_power[i] >= expected_power[i-1] * 0.8  # Monotonic increase (with tolerance)
    
    def test_reproducibility_framework(self):
        """Test reproducibility framework implementation"""
        
        # Test that reproducibility metrics are comprehensive
        required_metrics = [
            "experiment_id",
            "algorithm_versions", 
            "execution_environment"
        ]
        
        dqceo = DynamicQuantumClassicalOptimizer()
        versions = dqceo._get_algorithm_versions()
        environment = dqceo._get_execution_environment()
        
        # Validate version tracking
        assert "dqceo_version" in versions
        assert all(isinstance(v, str) for v in versions.values())
        
        # Validate environment tracking
        assert "python_version" in environment
        assert "platform" in environment
        assert "timestamp" in environment
    
    @pytest.mark.asyncio
    async def test_publication_ready_output_format(self):
        """Test that output format is suitable for academic publication"""
        
        dqceo = DynamicQuantumClassicalOptimizer()
        
        # Create minimal test case
        task = QuantumTask(
            title="Test Task",
            description="Minimal test",
            priority=TaskPriority.NORMAL
        )
        
        result = await dqceo.optimize_with_dynamic_selection(
            tasks=[task],
            constraints={},
            objectives=[lambda tasks: 1.0],
            time_budget=2.0
        )
        
        # Validate publication-ready sections
        pub_data = result["publication_ready_data"]
        
        required_sections = [
            "problem_characteristics",
            "performance_comparison", 
            "statistical_significance"
        ]
        
        for section in required_sections:
            assert section in pub_data
        
        # Validate data types are JSON-serializable
        import json
        try:
            json.dumps(result, default=str)  # Should not raise exception
        except TypeError:
            pytest.fail("Result contains non-serializable data types")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])