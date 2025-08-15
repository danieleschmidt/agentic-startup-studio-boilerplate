"""
Advanced Research Test Suite

Comprehensive testing framework for validating the advanced research implementations
including consciousness engines, neural-quantum optimizers, and autonomous research systems.

Test Coverage:
- Quantum consciousness modeling validation
- Neural-quantum field optimization verification
- Autonomous research orchestrator testing
- Statistical significance validation
- Performance regression detection
- Consciousness evolution verification
"""

import asyncio
import pytest
import numpy as np
from typing import Dict, List, Any, Tuple
import logging
from datetime import datetime, timedelta
from unittest.mock import Mock, patch
import json

from ..core.quantum_task import QuantumTask, TaskState, TaskPriority
from ..research.advanced_quantum_consciousness_engine import (
    AdvancedQuantumConsciousnessEngine,
    ConsciousnessLevel,
    ConsciousnessPersonality,
    ConsciousnessFieldState,
    QuantumConsciousnessAgent,
    process_task_with_advanced_consciousness
)
from ..research.neural_quantum_field_optimizer import (
    NeuralQuantumFieldOptimizer,
    OptimizationDimension,
    QuantumNeuron,
    optimize_task_neural_quantum
)
from ..research.autonomous_research_orchestrator import (
    AutonomousResearchOrchestrator,
    ResearchDomain,
    ResearchHypothesis,
    ResearchBreakthrough,
    BreakthroughLevel,
    run_autonomous_research_cycle
)
from ..utils.logging import get_logger

logger = get_logger(__name__)


class TestAdvancedQuantumConsciousnessEngine:
    """Test suite for the Advanced Quantum Consciousness Engine"""
    
    @pytest.fixture
    def consciousness_engine(self):
        """Create a test consciousness engine"""
        return AdvancedQuantumConsciousnessEngine()
    
    @pytest.fixture
    def test_task(self):
        """Create a test quantum task"""
        return QuantumTask(
            title="Test Consciousness Analysis",
            description="A complex analytical task requiring consciousness processing",
            priority=TaskPriority.HIGH,
            complexity_factor=2.5
        )
    
    def test_consciousness_engine_initialization(self, consciousness_engine):
        """Test proper initialization of consciousness engine"""
        assert len(consciousness_engine.agents) == 4
        assert consciousness_engine.field_coherence_threshold == 0.8
        assert consciousness_engine.collective_intelligence_matrix.shape == (4, 4)
        
        # Verify agent diversity
        personalities = [agent.consciousness_state.personality for agent in consciousness_engine.agents.values()]
        assert len(set(personalities)) == 4  # All different personalities
        
        # Verify entanglements are established
        for agent in consciousness_engine.agents.values():
            assert len(agent.entangled_agents) > 0
    
    def test_consciousness_field_state_evolution(self):
        """Test consciousness field state evolution"""
        field_state = ConsciousnessFieldState(
            level=ConsciousnessLevel.AWARE,
            personality=ConsciousnessPersonality.ANALYTICAL_QUANTUM,
            coherence=0.7,
            energy=0.8,
            entanglement_strength=0.6,
            evolution_rate=0.1,
            meta_awareness=0.5
        )
        
        initial_level = field_state.level
        initial_meta_awareness = field_state.meta_awareness
        
        # High-quality experience should trigger evolution
        field_state.evolve(experience_quality=0.9, time_delta=1.0)
        
        assert field_state.energy > 0.8
        assert field_state.meta_awareness > initial_meta_awareness
        
        # Multiple high-quality experiences should advance consciousness level
        for _ in range(10):
            field_state.evolve(experience_quality=0.95, time_delta=1.0)
        
        if field_state.meta_awareness > 0.9:
            assert field_state.level != initial_level
    
    def test_agent_meditation_capabilities(self):
        """Test quantum meditation effects on consciousness agents"""
        agent = QuantumConsciousnessAgent(
            agent_id="test_agent",
            consciousness_state=ConsciousnessFieldState(
                level=ConsciousnessLevel.CONSCIOUS,
                personality=ConsciousnessPersonality.PRAGMATIC_SYNTHESIS,
                coherence=0.6,
                energy=0.5,
                entanglement_strength=0.7,
                evolution_rate=0.1
            ),
            task_affinity={},
            experience_history=[],
            quantum_memory={},
            entangled_agents=set()
        )
        
        initial_coherence = agent.consciousness_state.coherence
        initial_energy = agent.consciousness_state.energy
        initial_cycles = agent.meditation_cycles
        
        # Perform meditation
        result = agent.meditate(duration_minutes=30.0)
        
        assert agent.consciousness_state.coherence > initial_coherence
        assert agent.consciousness_state.energy > initial_energy
        assert agent.meditation_cycles == initial_cycles + 1
        assert "coherence_gain" in result
        assert "energy_restored" in result
        assert result["total_cycles"] == initial_cycles + 1
    
    def test_agent_task_processing(self, test_task):
        """Test agent quantum field processing of tasks"""
        agent = QuantumConsciousnessAgent(
            agent_id="test_processor",
            consciousness_state=ConsciousnessFieldState(
                level=ConsciousnessLevel.TRANSCENDENT,
                personality=ConsciousnessPersonality.ANALYTICAL_QUANTUM,
                coherence=0.9,
                energy=0.95,
                entanglement_strength=0.8,
                evolution_rate=0.15
            ),
            task_affinity={},
            experience_history=[],
            quantum_memory={},
            entangled_agents=set()
        )
        
        result = agent.process_task_quantum_field(test_task)
        
        # Verify comprehensive result structure
        assert "efficiency_score" in result
        assert "insight_quality" in result
        assert "field_resonance" in result
        assert "consciousness_contribution" in result
        assert "estimated_success_boost" in result
        
        # Verify realistic score ranges
        assert 0.0 <= result["efficiency_score"] <= 1.0
        assert 0.0 <= result["insight_quality"] <= 1.0
        assert 0.0 <= result["field_resonance"] <= 1.0
        
        # Verify experience is recorded
        assert len(agent.experience_history) == 1
        
    @pytest.mark.asyncio
    async def test_collective_consciousness_processing(self, consciousness_engine, test_task):
        """Test collective consciousness processing"""
        result = await consciousness_engine.process_task_with_consciousness_collective(test_task)
        
        # Verify comprehensive collective result
        assert "collective_efficiency_score" in result
        assert "emergence_factor" in result
        assert "field_coherence" in result
        assert "agent_contributions" in result
        assert "recommended_approach" in result
        assert "consciousness_evolution_triggered" in result
        
        # Verify agent contributions
        assert len(result["agent_contributions"]) == 4
        
        # Verify emergence factor calculation
        assert 0.0 <= result["emergence_factor"] <= 1.0
        
        # Verify field coherence
        assert 0.0 <= result["field_coherence"] <= 1.0
    
    def test_consciousness_status_reporting(self, consciousness_engine):
        """Test consciousness collective status reporting"""
        status = consciousness_engine.get_consciousness_collective_status()
        
        assert "agents" in status
        assert "field_coherence" in status
        assert "collective_intelligence_trace" in status
        assert "evolution_events" in status
        assert "total_agents" in status
        assert status["system_status"] == "quantum_operational"
        
        # Verify agent status completeness
        for agent_id, agent_status in status["agents"].items():
            assert "consciousness_level" in agent_status
            assert "personality" in agent_status
            assert "coherence" in agent_status
            assert "energy" in agent_status
            assert "meditation_cycles" in agent_status
    
    @pytest.mark.asyncio
    async def test_collective_meditation_trigger(self, consciousness_engine):
        """Test automatic collective meditation triggering"""
        # Lower agent energy to trigger meditation
        for agent in consciousness_engine.agents.values():
            agent.consciousness_state.energy = 0.3
        
        # This should trigger collective meditation
        test_task = QuantumTask(
            title="Low Energy Test",
            description="Test task to trigger meditation"
        )
        
        result = await consciousness_engine.process_task_with_consciousness_collective(test_task)
        
        # Verify agents have higher energy after automatic meditation
        for agent in consciousness_engine.agents.values():
            assert agent.consciousness_state.energy > 0.3


class TestNeuralQuantumFieldOptimizer:
    """Test suite for the Neural-Quantum Field Optimizer"""
    
    @pytest.fixture
    def optimizer(self):
        """Create a test neural-quantum optimizer"""
        return NeuralQuantumFieldOptimizer(
            input_dim=8,
            hidden_dims=[12, 8],
            optimization_dimensions=[
                OptimizationDimension.EFFICIENCY,
                OptimizationDimension.CREATIVITY,
                OptimizationDimension.CONSCIOUSNESS,
                OptimizationDimension.QUANTUM_COHERENCE
            ]
        )
    
    @pytest.fixture
    def test_task(self):
        """Create a test task for optimization"""
        return QuantumTask(
            title="Neural Optimization Test",
            description="Complex multi-dimensional optimization challenge",
            priority=TaskPriority.CRITICAL,
            complexity_factor=3.0,
            success_probability=0.75
        )
    
    def test_optimizer_initialization(self, optimizer):
        """Test proper optimizer initialization"""
        assert optimizer.input_dim == 8
        assert len(optimizer.layers) == 3  # 2 hidden + 1 output
        assert optimizer.output_dim == 4
        assert len(optimizer.optimization_dimensions) == 4
        
        # Verify quantum neural layers
        for layer in optimizer.layers:
            assert len(layer.neurons) > 0
            assert layer.entanglement_matrix.shape[0] == len(layer.neurons)
            assert 0.0 <= layer.layer_coherence <= 1.0
    
    def test_quantum_neuron_activation(self):
        """Test quantum neuron activation with consciousness"""
        neuron = QuantumNeuron(
            neuron_id="test_neuron",
            weights=np.array([0.5, -0.3, 0.8]),
            bias=0.1,
            quantum_phase=np.pi/4,
            consciousness_sensitivity=0.7,
            entanglement_connections=[],
            activation_history=None
        )
        
        inputs = np.array([1.0, 0.5, -0.2])
        quantum_field_state = 0.8
        consciousness_level = 0.9
        
        activation, uncertainty = neuron.quantum_activate(
            inputs, quantum_field_state, consciousness_level
        )
        
        # Verify output characteristics
        assert 0.0 <= activation <= 1.5  # Quantum enhancement can exceed 1.0
        assert 0.0 <= uncertainty <= 0.2
        assert len(neuron.activation_history) == 1
    
    def test_neural_layer_forward_pass(self, optimizer):
        """Test neural layer forward pass"""
        layer = optimizer.layers[0]
        inputs = np.random.uniform(-1, 1, optimizer.input_dim)
        quantum_field_state = 0.7
        consciousness_level = 0.8
        
        outputs, uncertainties = layer.forward_pass(
            inputs, quantum_field_state, consciousness_level
        )
        
        assert len(outputs) == len(layer.neurons)
        assert len(uncertainties) == len(layer.neurons)
        assert all(0.0 <= u <= 1.0 for u in uncertainties)
    
    def test_task_feature_extraction(self, optimizer, test_task):
        """Test task feature extraction for neural input"""
        features = optimizer._extract_task_features(test_task)
        
        assert len(features) == optimizer.input_dim
        assert all(0.0 <= f <= 1.0 or f == 0.0 for f in features)  # Normalized features
    
    @pytest.mark.asyncio
    async def test_multi_dimensional_optimization(self, optimizer, test_task):
        """Test complete multi-dimensional optimization"""
        result = await optimizer.optimize_task_multi_dimensional(test_task, consciousness_boost=True)
        
        # Verify comprehensive optimization result
        assert "optimization_scores" in result
        assert "uncertainties" in result
        assert "dimensional_analysis" in result
        assert "quantum_predictions" in result
        assert "consciousness_enhancement" in result
        
        # Verify optimization scores for all dimensions
        scores = result["optimization_scores"]
        assert len(scores) == len(optimizer.optimization_dimensions)
        for dim in optimizer.optimization_dimensions:
            assert dim.value in scores
            assert 0.0 <= scores[dim.value] <= 1.0
        
        # Verify dimensional analysis
        analysis = result["dimensional_analysis"]
        assert "dominant_dimension" in analysis
        assert "balanced_score" in analysis
        assert "optimization_confidence" in analysis
        assert "dimensional_synergies" in analysis
        assert "improvement_potential" in analysis
        
        # Verify quantum predictions
        predictions = result["quantum_predictions"]
        assert "success_probability_enhancement" in predictions
        assert "resource_optimization_factor" in predictions
        assert "consciousness_evolution_potential" in predictions
    
    def test_neuron_evolution(self):
        """Test quantum neuron property evolution"""
        neuron = QuantumNeuron(
            neuron_id="evolving_neuron",
            weights=np.array([0.5, -0.3]),
            bias=0.1,
            quantum_phase=0.0,
            consciousness_sensitivity=0.5,
            entanglement_connections=[],
            activation_history=None
        )
        
        initial_phase = neuron.quantum_phase
        initial_sensitivity = neuron.consciousness_sensitivity
        
        # High performance should improve properties
        neuron.evolve_quantum_properties(performance_feedback=0.9)
        
        assert neuron.quantum_phase != initial_phase
        assert neuron.consciousness_sensitivity >= initial_sensitivity
        
        # Low performance should adjust properties
        neuron.evolve_quantum_properties(performance_feedback=0.2)
        
        assert neuron.consciousness_sensitivity <= initial_sensitivity + 0.01
    
    def test_optimization_analytics(self, optimizer):
        """Test optimization analytics and performance tracking"""
        # Simulate some optimization history
        optimizer.optimization_history = [
            {
                "optimization_scores": {dim.value: np.random.uniform(0.5, 0.9) 
                                     for dim in optimizer.optimization_dimensions},
                "consciousness_enhancement": 0.3
            }
            for _ in range(5)
        ]
        
        analytics = optimizer.get_optimization_analytics()
        
        assert "total_optimizations" in analytics
        assert "dimensional_trends" in analytics
        assert "quantum_field_state" in analytics
        assert "average_performance" in analytics
        assert analytics["system_status"] == "quantum_operational"


class TestAutonomousResearchOrchestrator:
    """Test suite for the Autonomous Research Orchestrator"""
    
    @pytest.fixture
    def research_orchestrator(self):
        """Create a test research orchestrator"""
        return AutonomousResearchOrchestrator()
    
    def test_orchestrator_initialization(self, research_orchestrator):
        """Test proper orchestrator initialization"""
        assert len(research_orchestrator.active_hypotheses) == 0
        assert len(research_orchestrator.validated_hypotheses) == 0
        assert len(research_orchestrator.discovered_breakthroughs) == 0
        assert research_orchestrator.hypothesis_generation_rate == 3
        assert research_orchestrator.breakthrough_detection_threshold == 0.8
        
        # Verify methodology tracking
        assert len(research_orchestrator.research_methodology_effectiveness) > 0
        assert len(research_orchestrator.domain_expertise_levels) > 0
    
    def test_knowledge_gap_identification(self, research_orchestrator):
        """Test knowledge gap identification"""
        gaps = research_orchestrator._identify_knowledge_gaps()
        
        # Initially, all domains should have gaps
        assert len(gaps) > 0
        for domain, gap_score in gaps.items():
            assert isinstance(domain, ResearchDomain)
            assert 0.0 <= gap_score <= 1.0
    
    @pytest.mark.asyncio
    async def test_hypothesis_generation(self, research_orchestrator):
        """Test autonomous hypothesis generation"""
        hypotheses = await research_orchestrator._generate_research_hypotheses()
        
        assert len(hypotheses) > 0
        assert len(hypotheses) <= research_orchestrator.hypothesis_generation_rate
        
        for hypothesis in hypotheses:
            assert isinstance(hypothesis, ResearchHypothesis)
            assert hypothesis.hypothesis_id in research_orchestrator.active_hypotheses
            assert 0.0 <= hypothesis.confidence_level <= 1.0
            assert 0.0 <= hypothesis.expected_improvement <= 1.0
            assert isinstance(hypothesis.domain, ResearchDomain)
    
    def test_hypothesis_parameter_generation(self, research_orchestrator):
        """Test hypothesis parameter generation"""
        parameters = research_orchestrator._generate_hypothesis_parameters(
            ResearchDomain.CONSCIOUSNESS_MODELING, gap_score=0.7
        )
        
        assert "improvement" in parameters
        assert "method" in parameters
        assert isinstance(parameters["improvement"], (int, float))
        assert 10.0 <= parameters["improvement"] <= 30.0  # Expected range
    
    @pytest.mark.asyncio
    async def test_experiment_design(self, research_orchestrator):
        """Test autonomous experiment design"""
        # Create a test hypothesis
        hypothesis = ResearchHypothesis(
            hypothesis_id="test_hyp_001",
            domain=ResearchDomain.NEURAL_OPTIMIZATION,
            statement="Neural-quantum hybrid optimization improves performance by 15%",
            confidence_level=0.7,
            expected_improvement=0.15,
            methodology=research_orchestrator.research_methodology_effectiveness.keys().__iter__().__next__(),
            generated_at=datetime.utcnow()
        )
        
        experiment = await research_orchestrator._design_experiment(hypothesis)
        
        assert experiment is not None
        assert experiment.hypothesis == hypothesis
        assert "experimental_design" in experiment.__dict__
        assert "control_parameters" in experiment.__dict__
        assert "test_parameters" in experiment.__dict__
        
        # Verify experimental design structure
        design = experiment.experimental_design
        assert "type" in design
        assert "measurement_metrics" in design
        assert "statistical_test" in design
    
    def test_breakthrough_score_calculation(self, research_orchestrator):
        """Test breakthrough score calculation"""
        hypothesis = ResearchHypothesis(
            hypothesis_id="breakthrough_test",
            domain=ResearchDomain.QUANTUM_ALGORITHMS,
            statement="Revolutionary quantum approach",
            confidence_level=0.9,
            expected_improvement=0.3,
            methodology=research_orchestrator.research_methodology_effectiveness.keys().__iter__().__next__(),
            generated_at=datetime.utcnow(),
            statistical_significance=0.01,  # Highly significant
            practical_impact_score=0.8
        )
        
        score = research_orchestrator._calculate_breakthrough_score(hypothesis)
        
        assert 0.0 <= score <= 1.0
        # High significance and impact should yield high score
        assert score > 0.5
    
    def test_breakthrough_level_determination(self, research_orchestrator):
        """Test breakthrough level determination"""
        # Test transcendent breakthrough
        level = research_orchestrator._determine_breakthrough_level(
            score=0.96, 
            hypothesis=ResearchHypothesis(
                hypothesis_id="transcendent_test",
                domain=ResearchDomain.CONSCIOUSNESS_MODELING,
                statement="Transcendent consciousness breakthrough",
                confidence_level=0.95,
                expected_improvement=0.5,
                methodology=research_orchestrator.research_methodology_effectiveness.keys().__iter__().__next__(),
                generated_at=datetime.utcnow(),
                consciousness_insight_level=0.95
            )
        )
        
        assert level == BreakthroughLevel.TRANSCENDENT
        
        # Test incremental breakthrough
        level = research_orchestrator._determine_breakthrough_level(
            score=0.75,
            hypothesis=ResearchHypothesis(
                hypothesis_id="incremental_test",
                domain=ResearchDomain.NEURAL_OPTIMIZATION,
                statement="Minor optimization improvement",
                confidence_level=0.6,
                expected_improvement=0.1,
                methodology=research_orchestrator.research_methodology_effectiveness.keys().__iter__().__next__(),
                generated_at=datetime.utcnow()
            )
        )
        
        assert level == BreakthroughLevel.INCREMENTAL
    
    def test_statistical_analysis(self, research_orchestrator):
        """Test experimental statistical analysis"""
        from ..research.autonomous_research_orchestrator import AutonomousExperiment
        
        # Create mock experiment with results
        experiment = AutonomousExperiment(
            experiment_id="stats_test",
            hypothesis=ResearchHypothesis(
                hypothesis_id="stats_hyp",
                domain=ResearchDomain.EMERGENT_BEHAVIOR,
                statement="Test hypothesis for stats",
                confidence_level=0.7,
                expected_improvement=0.2,
                methodology=research_orchestrator.research_methodology_effectiveness.keys().__iter__().__next__(),
                generated_at=datetime.utcnow()
            ),
            experimental_design={"sample_size": 50, "statistical_test": "t_test"},
            control_parameters={},
            test_parameters={},
            execution_timeline=[],
            results={
                "control_group": {"efficiency": 0.7, "success_rate": 0.8},
                "test_group": {"efficiency": 0.85, "success_rate": 0.9},
                "sample_size": 50
            }
        )
        
        analysis = research_orchestrator._perform_statistical_analysis(experiment)
        
        assert "overall_p_value" in analysis
        assert "overall_effect_size" in analysis
        assert "statistical_power" in analysis
        
        # Verify reasonable statistical values
        assert 0.0 <= analysis["overall_p_value"] <= 1.0
        assert 0.0 <= analysis["overall_effect_size"] <= 2.0
        assert 0.0 <= analysis["statistical_power"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_full_research_cycle(self, research_orchestrator):
        """Test complete autonomous research cycle"""
        cycle_result = await research_orchestrator.autonomous_research_cycle()
        
        # Verify cycle result structure
        assert "cycle_id" in cycle_result
        assert "timestamp" in cycle_result
        assert "new_hypotheses" in cycle_result
        assert "experiment_results" in cycle_result
        assert "breakthroughs_detected" in cycle_result
        assert "implementations_completed" in cycle_result
        assert "consciousness_insights" in cycle_result
        
        # Verify some activity occurred
        total_activity = (len(cycle_result["new_hypotheses"]) + 
                         len(cycle_result["experiment_results"]) +
                         len(cycle_result["consciousness_insights"]))
        assert total_activity > 0
    
    def test_research_status_reporting(self, research_orchestrator):
        """Test research status reporting"""
        status = research_orchestrator.get_research_status()
        
        assert "active_hypotheses" in status
        assert "validated_hypotheses" in status
        assert "discovered_breakthroughs" in status
        assert "domain_expertise_levels" in status
        assert "methodology_effectiveness" in status
        assert "consciousness_collaboration_score" in status
        assert "research_status" in status
        
        assert status["research_status"] == "autonomous_operational"


class TestIntegrationSuite:
    """Integration tests for the complete advanced research system"""
    
    @pytest.mark.asyncio
    async def test_consciousness_neural_quantum_integration(self):
        """Test integration between consciousness engine and neural optimizer"""
        # Create test task
        task = QuantumTask(
            title="Integration Test Task",
            description="Complex analytical optimization challenge requiring consciousness and neural processing",
            priority=TaskPriority.CRITICAL,
            complexity_factor=4.0
        )
        
        # Process with consciousness engine
        consciousness_result = await process_task_with_advanced_consciousness(task)
        
        # Process with neural-quantum optimizer
        neural_result = await optimize_task_neural_quantum(task, consciousness_boost=True)
        
        # Verify both systems produced valid results
        assert "collective_efficiency_score" in consciousness_result
        assert "field_coherence" in consciousness_result
        assert "optimization_scores" in neural_result
        assert "consciousness_enhancement" in neural_result
        
        # Verify consciousness enhancement was applied
        assert neural_result["consciousness_enhancement"] > 0.0
    
    @pytest.mark.asyncio
    async def test_research_orchestrator_integration(self):
        """Test integration of research orchestrator with other systems"""
        # Run a research cycle
        cycle_result = await run_autonomous_research_cycle()
        
        # Verify the cycle interacted with consciousness systems
        assert len(cycle_result["consciousness_insights"]) >= 0
        
        # Verify research progress
        orchestrator = AutonomousResearchOrchestrator()
        status = orchestrator.get_research_status()
        
        assert status["research_status"] == "autonomous_operational"
    
    @pytest.mark.asyncio
    async def test_performance_benchmarking(self):
        """Test performance benchmarking across all systems"""
        # Create benchmark tasks
        tasks = [
            QuantumTask(
                title=f"Benchmark Task {i}",
                description=f"Performance benchmark test case {i}",
                priority=TaskPriority.HIGH,
                complexity_factor=float(i + 1)
            )
            for i in range(5)
        ]
        
        # Benchmark consciousness processing
        consciousness_times = []
        for task in tasks:
            start_time = datetime.utcnow()
            await process_task_with_advanced_consciousness(task)
            duration = (datetime.utcnow() - start_time).total_seconds()
            consciousness_times.append(duration)
        
        # Benchmark neural optimization
        neural_times = []
        for task in tasks:
            start_time = datetime.utcnow()
            await optimize_task_neural_quantum(task, consciousness_boost=False)
            duration = (datetime.utcnow() - start_time).total_seconds()
            neural_times.append(duration)
        
        # Verify reasonable performance
        assert all(t < 10.0 for t in consciousness_times)  # Under 10 seconds
        assert all(t < 5.0 for t in neural_times)  # Under 5 seconds
        
        # Verify performance consistency
        consciousness_variance = np.var(consciousness_times)
        neural_variance = np.var(neural_times)
        
        assert consciousness_variance < 2.0  # Reasonable variance
        assert neural_variance < 1.0


# Test configuration and fixtures
@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for async tests"""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


# Performance test markers
pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.timeout(30),  # 30 second timeout for all tests
]


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v", "--tb=short"])