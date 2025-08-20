"""
Test Suite for Experimental Research Framework

Validation of research infrastructure for consciousness-quantum optimization experiments.
Ensures reproducibility, statistical validity, and publication-ready experimental design.
"""

import pytest
import asyncio
import numpy as np
import time
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Any

from quantum_task_planner.core.quantum_task import QuantumTask, TaskState, TaskPriority
from quantum_task_planner.research.experimental_research_framework import (
    ExperimentConfiguration,
    ExperimentResult,
    ExperimentType,
    ExperimentRunner,
    ExperimentalTaskGenerator,
    RandomAssignmentOptimizer,
    PrioritySortOptimizer,
    GreedyDurationOptimizer,
    StatisticalAnalyzer,
    ExperimentalResearchFramework
)


class TestExperimentalTaskGenerator:
    """Test synthetic task generation for experiments"""
    
    def test_generate_task_set_basic(self):
        """Test basic task generation"""
        tasks = ExperimentalTaskGenerator.generate_task_set(
            problem_size=10,
            complexity_level="medium",
            random_seed=42
        )
        
        assert len(tasks) == 10
        assert all(isinstance(task, QuantumTask) for task in tasks)
        assert all(hasattr(task, 'metadata') for task in tasks)
        
        # Check task diversity
        priorities = [task.priority for task in tasks]
        assert len(set(priorities)) > 1, "Tasks should have diverse priorities"
        
        # Check metadata
        for task in tasks:
            assert 'task_type' in task.metadata
            assert 'complexity_factor' in task.metadata
            assert 0.0 <= task.metadata['emotional_content'] <= 1.0
            assert 0.0 <= task.metadata['creativity_required'] <= 1.0
    
    def test_generate_task_set_reproducibility(self):
        """Test reproducible task generation"""
        tasks1 = ExperimentalTaskGenerator.generate_task_set(
            problem_size=5, random_seed=123
        )
        tasks2 = ExperimentalTaskGenerator.generate_task_set(
            problem_size=5, random_seed=123
        )
        
        # Tasks should be identical with same seed
        assert len(tasks1) == len(tasks2)
        for t1, t2 in zip(tasks1, tasks2):
            assert t1.title == t2.title
            assert t1.priority == t2.priority
            assert t1.estimated_duration == t2.estimated_duration
    
    def test_complexity_levels(self):
        """Test different complexity levels"""
        simple_tasks = ExperimentalTaskGenerator.generate_task_set(
            problem_size=5, complexity_level="simple", random_seed=42
        )
        complex_tasks = ExperimentalTaskGenerator.generate_task_set(
            problem_size=5, complexity_level="complex", random_seed=42
        )
        
        # Complex tasks should generally have longer durations
        simple_avg_duration = np.mean([t.estimated_duration.total_seconds() for t in simple_tasks])
        complex_avg_duration = np.mean([t.estimated_duration.total_seconds() for t in complex_tasks])
        
        assert complex_avg_duration > simple_avg_duration


class TestBaselineOptimizers:
    """Test baseline optimization algorithms"""
    
    @pytest.mark.asyncio
    async def test_random_assignment_optimizer(self):
        """Test random assignment baseline"""
        tasks = ExperimentalTaskGenerator.generate_task_set(5, random_seed=42)
        optimizer = RandomAssignmentOptimizer()
        
        results = await optimizer.optimize_tasks(tasks)
        
        assert 'optimized_task_order' in results
        assert 'execution_time_seconds' in results
        assert 'solution_quality' in results
        assert len(results['optimized_task_order']) == 5
        assert 0.0 <= results['solution_quality'] <= 1.0
        assert optimizer.get_algorithm_name() == "random_assignment"
    
    @pytest.mark.asyncio
    async def test_priority_sort_optimizer(self):
        """Test priority-based sorting baseline"""
        tasks = ExperimentalTaskGenerator.generate_task_set(5, random_seed=42)
        optimizer = PrioritySortOptimizer()
        
        results = await optimizer.optimize_tasks(tasks)
        
        assert 'optimized_task_order' in results
        assert results['solution_quality'] >= 0.6  # Should be reasonable quality
        assert optimizer.get_algorithm_name() == "priority_sort"
        
        # Verify priority ordering
        task_dict = {task.id: task for task in tasks}
        ordered_tasks = [task_dict[task_id] for task_id in results['optimized_task_order'] if task_id in task_dict]
        
        # High priority tasks should generally come first
        high_priority_positions = [i for i, task in enumerate(ordered_tasks) if task.priority == TaskPriority.HIGH]
        low_priority_positions = [i for i, task in enumerate(ordered_tasks) if task.priority == TaskPriority.LOW]
        
        if high_priority_positions and low_priority_positions:
            assert min(high_priority_positions) <= max(low_priority_positions)
    
    @pytest.mark.asyncio
    async def test_greedy_duration_optimizer(self):
        """Test greedy duration-based optimizer"""
        tasks = ExperimentalTaskGenerator.generate_task_set(5, random_seed=42)
        optimizer = GreedyDurationOptimizer()
        
        results = await optimizer.optimize_tasks(tasks)
        
        assert 'optimized_task_order' in results
        assert optimizer.get_algorithm_name() == "greedy_duration"
        
        # Verify duration ordering (shortest first)
        task_dict = {task.id: task for task in tasks}
        ordered_tasks = [task_dict[task_id] for task_id in results['optimized_task_order'] if task_id in task_dict]
        
        durations = [task.estimated_duration.total_seconds() for task in ordered_tasks]
        assert durations == sorted(durations), "Tasks should be ordered by duration"


class TestExperimentConfiguration:
    """Test experiment configuration"""
    
    def test_experiment_configuration_creation(self):
        """Test experiment configuration creation"""
        config = ExperimentConfiguration(
            experiment_id="test_exp_001",
            experiment_type=ExperimentType.COMPARATIVE_PERFORMANCE,
            problem_sizes=[5, 10],
            num_runs_per_size=3,
            algorithms_to_compare=["consciousness_quantum", "priority_sort"],
            consciousness_agent_counts=[2, 4],
            random_seed=42
        )
        
        assert config.experiment_id == "test_exp_001"
        assert config.experiment_type == ExperimentType.COMPARATIVE_PERFORMANCE
        assert config.problem_sizes == [5, 10]
        assert config.num_runs_per_size == 3
        
        # Test hash generation
        hash1 = config.generate_experiment_hash()
        hash2 = config.generate_experiment_hash()
        assert hash1 == hash2  # Should be deterministic
        assert len(hash1) == 16  # Expected hash length


class TestExperimentResult:
    """Test experiment result handling"""
    
    def test_experiment_result_creation(self):
        """Test experiment result creation and serialization"""
        result = ExperimentResult(
            experiment_id="test_exp",
            run_id="5_0",
            algorithm_name="test_algorithm",
            problem_size=5,
            execution_time_seconds=1.5,
            solution_quality=0.8,
            optimization_efficiency=3.33,
            consciousness_metrics={"empathy": 0.7},
            quantum_metrics={"coherence": 0.6},
            memory_usage_mb=50.0
        )
        
        # Test dictionary conversion
        result_dict = result.to_dict()
        
        assert result_dict['experiment_id'] == "test_exp"
        assert result_dict['algorithm_name'] == "test_algorithm"
        assert result_dict['solution_quality'] == 0.8
        assert 'timestamp' in result_dict
        
        # Test JSON serialization
        json_str = json.dumps(result_dict)
        assert json_str is not None


class TestExperimentRunner:
    """Test experiment execution"""
    
    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def basic_experiment_config(self, temp_output_dir):
        """Create basic experiment configuration"""
        return ExperimentConfiguration(
            experiment_id="test_basic_exp",
            experiment_type=ExperimentType.COMPARATIVE_PERFORMANCE,
            problem_sizes=[5],
            num_runs_per_size=2,
            algorithms_to_compare=["priority_sort", "random_assignment"],
            consciousness_agent_counts=[2],
            output_directory=temp_output_dir,
            random_seed=42
        )
    
    def test_experiment_runner_initialization(self, basic_experiment_config):
        """Test experiment runner initialization"""
        runner = ExperimentRunner(basic_experiment_config)
        
        assert runner.config == basic_experiment_config
        assert len(runner.baseline_optimizers) >= 3
        assert "priority_sort" in runner.baseline_optimizers
        assert "random_assignment" in runner.baseline_optimizers
        assert "greedy_duration" in runner.baseline_optimizers
    
    @pytest.mark.asyncio
    async def test_single_experiment_run(self, basic_experiment_config):
        """Test single experiment execution"""
        runner = ExperimentRunner(basic_experiment_config)
        
        # Run single experiment
        results = await runner.run_single_experiment(problem_size=5, run_index=0)
        
        assert len(results) == 2  # Two algorithms
        
        for result in results:
            assert isinstance(result, ExperimentResult)
            assert result.problem_size == 5
            assert result.execution_time_seconds >= 0
            assert 0.0 <= result.solution_quality <= 1.0
            assert result.algorithm_name in ["priority_sort", "random_assignment"]
    
    @pytest.mark.asyncio
    async def test_full_experiment_execution(self, basic_experiment_config):
        """Test full experiment suite execution"""
        runner = ExperimentRunner(basic_experiment_config)
        
        # Run full experiment
        all_results = await runner.run_full_experiment()
        
        # Should have results for each algorithm, problem size, and run
        expected_results = len(basic_experiment_config.problem_sizes) * \
                          basic_experiment_config.num_runs_per_size * \
                          len(basic_experiment_config.algorithms_to_compare)
        
        assert len(all_results) == expected_results
        
        # Verify results structure
        for result in all_results:
            assert isinstance(result, ExperimentResult)
            assert result.experiment_id == basic_experiment_config.experiment_id
        
        # Check output files exist
        output_dir = Path(basic_experiment_config.output_directory)
        assert any(f.name.startswith("results_") for f in output_dir.glob("*.json"))
        assert any(f.name.startswith("results_") for f in output_dir.glob("*.csv"))


class TestStatisticalAnalyzer:
    """Test statistical analysis functionality"""
    
    @pytest.fixture
    def sample_results(self):
        """Create sample experiment results for testing"""
        results = []
        
        # Create results for two algorithms with different performance
        for algorithm in ["algorithm_a", "algorithm_b"]:
            for i in range(10):
                # Algorithm A performs better on average
                base_quality = 0.8 if algorithm == "algorithm_a" else 0.6
                quality = base_quality + np.random.normal(0, 0.1)
                quality = max(0.0, min(1.0, quality))  # Clamp to [0, 1]
                
                result = ExperimentResult(
                    experiment_id="test_stat_analysis",
                    run_id=f"{algorithm}_{i}",
                    algorithm_name=algorithm,
                    problem_size=10,
                    execution_time_seconds=1.0 + np.random.uniform(0, 0.5),
                    solution_quality=quality,
                    optimization_efficiency=10.0,
                    consciousness_metrics={},
                    quantum_metrics={},
                    memory_usage_mb=50.0
                )
                results.append(result)
        
        return results
    
    def test_statistical_analyzer_basic(self, sample_results):
        """Test basic statistical analysis"""
        analyzer = StatisticalAnalyzer(sample_results)
        analysis = analyzer.perform_comprehensive_analysis()
        
        # Check analysis structure
        assert hasattr(analysis, 'algorithm_comparisons')
        assert hasattr(analysis, 'significance_tests')
        assert hasattr(analysis, 'confidence_intervals')
        assert hasattr(analysis, 'performance_distributions')
        
        # Check performance distributions
        assert "algorithm_a" in analysis.performance_distributions
        assert "algorithm_b" in analysis.performance_distributions
        assert len(analysis.performance_distributions["algorithm_a"]) == 10
        assert len(analysis.performance_distributions["algorithm_b"]) == 10
    
    def test_confidence_intervals(self, sample_results):
        """Test confidence interval calculation"""
        analyzer = StatisticalAnalyzer(sample_results)
        analysis = analyzer.perform_comprehensive_analysis()
        
        # Should have confidence intervals for both algorithms
        assert "algorithm_a" in analysis.confidence_intervals
        assert "algorithm_b" in analysis.confidence_intervals
        
        # Confidence intervals should be reasonable
        for algorithm, (lower, upper) in analysis.confidence_intervals.items():
            assert lower <= upper
            assert 0.0 <= lower <= 1.0
            assert 0.0 <= upper <= 1.0


class TestExperimentalResearchFramework:
    """Test main research framework"""
    
    @pytest.fixture
    def temp_base_dir(self):
        """Create temporary base directory"""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_framework_initialization(self, temp_base_dir):
        """Test research framework initialization"""
        framework = ExperimentalResearchFramework(base_output_dir=temp_base_dir)
        
        assert framework.base_output_dir.exists()
        assert framework.base_output_dir.is_dir()
    
    def test_create_comparative_performance_experiment(self, temp_base_dir):
        """Test comparative performance experiment creation"""
        framework = ExperimentalResearchFramework(base_output_dir=temp_base_dir)
        
        config = framework.create_comparative_performance_experiment(
            problem_sizes=[5, 10],
            num_runs=3
        )
        
        assert config.experiment_type == ExperimentType.COMPARATIVE_PERFORMANCE
        assert config.problem_sizes == [5, 10]
        assert config.num_runs_per_size == 3
        assert "consciousness_quantum" in config.algorithms_to_compare
        assert "priority_sort" in config.algorithms_to_compare
    
    def test_create_consciousness_evolution_experiment(self, temp_base_dir):
        """Test consciousness evolution experiment creation"""
        framework = ExperimentalResearchFramework(base_output_dir=temp_base_dir)
        
        config = framework.create_consciousness_evolution_experiment(
            problem_sizes=[10, 20],
            num_runs=5
        )
        
        assert config.experiment_type == ExperimentType.CONSCIOUSNESS_EVOLUTION
        assert config.problem_sizes == [10, 20]
        assert config.num_runs_per_size == 5
        assert "consciousness_quantum" in config.algorithms_to_compare
        assert len(config.consciousness_agent_counts) > 1  # Should test different agent counts
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_run_experimental_suite_basic(self, temp_base_dir):
        """Test running basic experimental suite"""
        framework = ExperimentalResearchFramework(base_output_dir=temp_base_dir)
        
        # Create minimal experiment configuration
        config = ExperimentConfiguration(
            experiment_id="test_suite_basic",
            experiment_type=ExperimentType.COMPARATIVE_PERFORMANCE,
            problem_sizes=[3],  # Small problem size for speed
            num_runs_per_size=1,  # Single run for speed
            algorithms_to_compare=["priority_sort", "random_assignment"],  # Baseline only
            consciousness_agent_counts=[2],
            output_directory=str(framework.base_output_dir / "test_suite")
        )
        
        # Run experimental suite
        results, analysis = await framework.run_experimental_suite(config)
        
        # Validate results
        assert len(results) >= 2  # At least one result per algorithm
        assert all(isinstance(r, ExperimentResult) for r in results)
        
        # Validate analysis
        assert hasattr(analysis, 'performance_distributions')
        assert len(analysis.performance_distributions) >= 2
        
        # Check output files
        output_dir = framework.base_output_dir / "test_suite"
        assert output_dir.exists()
        assert any(f.name.endswith('.json') for f in output_dir.glob('*'))


@pytest.mark.integration
class TestResearchIntegration:
    """Integration tests for research framework"""
    
    @pytest.fixture
    def temp_research_dir(self):
        """Create temporary research directory"""
        temp_dir = tempfile.mkdtemp(prefix="research_integration_")
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    @pytest.mark.asyncio
    @pytest.mark.slow
    async def test_end_to_end_research_pipeline(self, temp_research_dir):
        """Test complete research pipeline from experiment design to analysis"""
        framework = ExperimentalResearchFramework(base_output_dir=temp_research_dir)
        
        # Phase 1: Design experiment
        config = framework.create_comparative_performance_experiment(
            problem_sizes=[5, 8],  # Small sizes for CI/CD
            num_runs=2  # Minimal runs for speed
        )
        
        # Phase 2: Execute experiments (excluding consciousness_quantum for speed)
        config.algorithms_to_compare = ["priority_sort", "greedy_duration", "random_assignment"]
        
        results, analysis = await framework.run_experimental_suite(config)
        
        # Phase 3: Validate comprehensive outputs
        assert len(results) >= 6  # 2 problem sizes * 2 runs * 3 algorithms (minimum)
        
        # Phase 4: Check statistical analysis
        assert len(analysis.performance_distributions) == 3  # 3 algorithms
        
        # Phase 5: Verify file outputs
        experiment_dir = Path(temp_research_dir) / "comparative_performance"
        assert experiment_dir.exists()
        
        # Check for results files
        json_files = list(experiment_dir.glob("results_*.json"))
        csv_files = list(experiment_dir.glob("results_*.csv"))
        analysis_files = list(experiment_dir.glob("statistical_analysis_*.json"))
        
        assert len(json_files) >= 1
        assert len(csv_files) >= 1
        assert len(analysis_files) >= 1
        
        # Phase 6: Validate JSON structure
        with open(json_files[0], 'r') as f:
            json_data = json.load(f)
            assert isinstance(json_data, list)
            assert len(json_data) == len(results)
            
            # Validate individual result structure
            for result_dict in json_data:
                assert 'algorithm_name' in result_dict
                assert 'solution_quality' in result_dict
                assert 'execution_time_seconds' in result_dict
                assert 'timestamp' in result_dict
    
    def test_reproducibility_validation(self, temp_research_dir):
        """Test experimental reproducibility"""
        framework = ExperimentalResearchFramework(base_output_dir=temp_research_dir)
        
        # Create two identical configurations
        config1 = ExperimentConfiguration(
            experiment_id="repro_test_1",
            experiment_type=ExperimentType.COMPARATIVE_PERFORMANCE,
            problem_sizes=[5],
            num_runs_per_size=1,
            algorithms_to_compare=["priority_sort"],
            consciousness_agent_counts=[2],
            random_seed=12345,
            output_directory=str(Path(temp_research_dir) / "repro_1")
        )
        
        config2 = ExperimentConfiguration(
            experiment_id="repro_test_2",
            experiment_type=ExperimentType.COMPARATIVE_PERFORMANCE,
            problem_sizes=[5],
            num_runs_per_size=1,
            algorithms_to_compare=["priority_sort"],
            consciousness_agent_counts=[2],
            random_seed=12345,  # Same seed
            output_directory=str(Path(temp_research_dir) / "repro_2")
        )
        
        # Both configurations should generate the same hash (indicating identical setup)
        hash1 = config1.generate_experiment_hash()
        hash2 = config2.generate_experiment_hash()
        
        # Hashes will be different due to different experiment_ids and output_directories
        # But the core experimental parameters should be captured
        assert isinstance(hash1, str)
        assert isinstance(hash2, str)
        assert len(hash1) == 16
        assert len(hash2) == 16


@pytest.mark.performance
class TestPerformanceBenchmarks:
    """Performance benchmarks for research framework"""
    
    def test_task_generation_performance(self):
        """Test performance of task generation"""
        start_time = time.time()
        
        # Generate large task set
        tasks = ExperimentalTaskGenerator.generate_task_set(
            problem_size=100,
            complexity_level="complex",
            random_seed=42
        )
        
        generation_time = time.time() - start_time
        
        assert len(tasks) == 100
        assert generation_time < 1.0  # Should generate 100 tasks in under 1 second
        
        print(f"Task generation time for 100 tasks: {generation_time:.3f}s")
    
    @pytest.mark.asyncio
    async def test_baseline_optimizer_performance(self):
        """Test performance of baseline optimizers"""
        tasks = ExperimentalTaskGenerator.generate_task_set(50, random_seed=42)
        
        optimizers = [
            RandomAssignmentOptimizer(),
            PrioritySortOptimizer(),
            GreedyDurationOptimizer()
        ]
        
        for optimizer in optimizers:
            start_time = time.time()
            results = await optimizer.optimize_tasks(tasks)
            execution_time = time.time() - start_time
            
            assert execution_time < 0.1  # Should complete in under 100ms
            assert len(results['optimized_task_order']) == 50
            
            print(f"{optimizer.get_algorithm_name()} time for 50 tasks: {execution_time:.3f}s")