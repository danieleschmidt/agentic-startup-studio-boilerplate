#!/usr/bin/env python3
"""
Terragon Autonomous SDLC - Continuous Learning Engine

This engine learns from execution outcomes to improve future
value discovery, scoring, and execution decisions.
"""

import json
import yaml
import numpy as np
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import statistics

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class LearningMetrics:
    """Metrics for tracking learning and adaptation."""
    estimation_accuracy: float
    value_prediction_accuracy: float
    execution_success_rate: float
    average_cycle_time: float
    false_positive_rate: float
    model_confidence: float
    adaptation_cycles: int
    last_calibration: str


@dataclass
class PredictionPattern:
    """Pattern learned from execution history."""
    pattern_type: str
    conditions: Dict[str, Any]
    adjustment_factor: float
    confidence: float
    samples: int
    last_updated: str


class ContinuousLearningEngine:
    """Engine for continuous learning and model adaptation."""
    
    def __init__(self, config_path: str = ".terragon/value-config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.learning_data: Dict = {}
        self.patterns: List[PredictionPattern] = []
        self.metrics: LearningMetrics = self._initialize_metrics()
        self._setup_paths()
        self._load_learning_data()
    
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
        
        self.learning_file = terragon_dir / "learning-data.json"
        self.patterns_file = terragon_dir / "learned-patterns.json"
        self.metrics_file = terragon_dir / "learning-metrics.json"
        self.execution_log = terragon_dir / "execution-log.json"
    
    def _initialize_metrics(self) -> LearningMetrics:
        """Initialize learning metrics with defaults."""
        return LearningMetrics(
            estimation_accuracy=0.5,
            value_prediction_accuracy=0.5,
            execution_success_rate=0.8,
            average_cycle_time=300.0,  # 5 minutes
            false_positive_rate=0.2,
            model_confidence=0.5,
            adaptation_cycles=0,
            last_calibration=datetime.now().isoformat()
        )
    
    def _load_learning_data(self):
        """Load existing learning data and patterns."""
        if self.learning_file.exists():
            try:
                with open(self.learning_file, 'r') as f:
                    self.learning_data = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load learning data: {e}")
                self.learning_data = {}
        
        if self.patterns_file.exists():
            try:
                with open(self.patterns_file, 'r') as f:
                    patterns_data = json.load(f)
                    self.patterns = [
                        PredictionPattern(**pattern) for pattern in patterns_data
                    ]
            except Exception as e:
                logger.warning(f"Failed to load patterns: {e}")
                self.patterns = []
        
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file, 'r') as f:
                    metrics_data = json.load(f)
                    self.metrics = LearningMetrics(**metrics_data)
            except Exception as e:
                logger.warning(f"Failed to load metrics: {e}")
    
    def learn_from_execution(self, execution_record: Dict):
        """Learn from a single execution record."""
        logger.info("Processing execution record for learning...")
        
        item = execution_record.get('item', {})
        result = execution_record.get('result', {})
        learning_data = execution_record.get('learning_data', {})
        
        # Update effort estimation accuracy
        self._update_effort_estimation(learning_data)
        
        # Update value prediction accuracy
        self._update_value_prediction(item, result)
        
        # Update execution success patterns
        self._update_success_patterns(item, result)
        
        # Update category-specific learning
        self._update_category_learning(item, result)
        
        # Detect new patterns
        self._detect_patterns(execution_record)
        
        # Update overall metrics
        self._update_metrics()
        
        # Save learning data
        self._save_learning_data()
        
        logger.info("Learning update completed")
    
    def _update_effort_estimation(self, learning_data: Dict):
        """Update effort estimation model based on actual vs predicted."""
        estimated = learning_data.get('estimated_effort', 0)
        actual = learning_data.get('actual_effort', 0)
        
        if estimated > 0 and actual > 0:
            # Calculate accuracy ratio
            accuracy_ratio = min(actual / estimated, estimated / actual)
            
            # Update estimation accuracy using exponential moving average
            alpha = self.config.get('learning', {}).get('learning_rate', 0.1)
            self.metrics.estimation_accuracy = (
                (1 - alpha) * self.metrics.estimation_accuracy + 
                alpha * accuracy_ratio
            )
            
            # Store historical data
            if 'effort_estimation' not in self.learning_data:
                self.learning_data['effort_estimation'] = []
            
            self.learning_data['effort_estimation'].append({
                'timestamp': datetime.now().isoformat(),
                'estimated': estimated,
                'actual': actual,
                'ratio': accuracy_ratio
            })
            
            # Keep only recent data (last 100 records)
            if len(self.learning_data['effort_estimation']) > 100:
                self.learning_data['effort_estimation'] = \
                    self.learning_data['effort_estimation'][-100:]
    
    def _update_value_prediction(self, item: Dict, result: Dict):
        """Update value prediction accuracy."""
        predicted_impact = item.get('impact_score', 0)
        success = result.get('success', False)
        
        # Simple binary success as actual impact measure
        actual_impact = 10.0 if success else 0.0
        
        if predicted_impact > 0:
            # Normalize and calculate accuracy
            normalized_predicted = min(predicted_impact, 10.0)
            accuracy = 1.0 - abs(normalized_predicted - actual_impact) / 10.0
            
            # Update value prediction accuracy
            alpha = self.config.get('learning', {}).get('learning_rate', 0.1)
            self.metrics.value_prediction_accuracy = (
                (1 - alpha) * self.metrics.value_prediction_accuracy + 
                alpha * accuracy
            )
            
            # Store historical data
            if 'value_prediction' not in self.learning_data:
                self.learning_data['value_prediction'] = []
            
            self.learning_data['value_prediction'].append({
                'timestamp': datetime.now().isoformat(),
                'predicted_impact': predicted_impact,
                'actual_success': success,
                'accuracy': accuracy
            })
            
            # Keep only recent data
            if len(self.learning_data['value_prediction']) > 100:
                self.learning_data['value_prediction'] = \
                    self.learning_data['value_prediction'][-100:]
    
    def _update_success_patterns(self, item: Dict, result: Dict):
        """Update patterns related to execution success."""
        category = item.get('category', 'unknown')
        success = result.get('success', False)
        
        # Update category success rates
        if 'category_success' not in self.learning_data:
            self.learning_data['category_success'] = {}
        
        if category not in self.learning_data['category_success']:
            self.learning_data['category_success'][category] = {
                'successes': 0,
                'total': 0
            }
        
        self.learning_data['category_success'][category]['total'] += 1
        if success:
            self.learning_data['category_success'][category]['successes'] += 1
        
        # Calculate overall success rate
        total_executions = sum(
            cat_data['total'] 
            for cat_data in self.learning_data['category_success'].values()
        )
        total_successes = sum(
            cat_data['successes'] 
            for cat_data in self.learning_data['category_success'].values()
        )
        
        if total_executions > 0:
            self.metrics.execution_success_rate = total_successes / total_executions
    
    def _update_category_learning(self, item: Dict, result: Dict):
        """Update category-specific learning data."""
        category = item.get('category', 'unknown')
        
        if 'category_patterns' not in self.learning_data:
            self.learning_data['category_patterns'] = {}
        
        if category not in self.learning_data['category_patterns']:
            self.learning_data['category_patterns'][category] = {
                'avg_effort': 0.0,
                'avg_impact': 0.0,
                'success_rate': 0.0,
                'count': 0
            }
        
        cat_data = self.learning_data['category_patterns'][category]
        count = cat_data['count']
        
        # Update averages using incremental mean
        estimated_effort = item.get('estimated_effort', 0)
        impact_score = item.get('impact_score', 0)
        success = result.get('success', False)
        
        cat_data['avg_effort'] = (
            (cat_data['avg_effort'] * count + estimated_effort) / (count + 1)
        )
        cat_data['avg_impact'] = (
            (cat_data['avg_impact'] * count + impact_score) / (count + 1)
        )
        cat_data['success_rate'] = (
            (cat_data['success_rate'] * count + (1.0 if success else 0.0)) / (count + 1)
        )
        cat_data['count'] = count + 1
    
    def _detect_patterns(self, execution_record: Dict):
        """Detect new patterns from execution data."""
        item = execution_record.get('item', {})
        result = execution_record.get('result', {})
        
        # Pattern: High-scoring items that fail
        if item.get('composite_score', 0) > 50 and not result.get('success', True):
            self._update_pattern(
                'high_score_failure',
                {'min_score': 50},
                0.7  # Reduce confidence in high scores
            )
        
        # Pattern: Low-effort items that succeed
        if item.get('estimated_effort', 0) < 2 and result.get('success', False):
            self._update_pattern(
                'low_effort_success',
                {'max_effort': 2},
                1.2  # Boost confidence in low-effort items
            )
        
        # Pattern: Security items success rate
        if item.get('category') == 'security':
            success_rate = self._calculate_category_success_rate('security')
            self._update_pattern(
                'security_reliability',
                {'category': 'security'},
                1.5 if success_rate > 0.8 else 0.8
            )
    
    def _update_pattern(self, pattern_type: str, conditions: Dict, adjustment: float):
        """Update or create a learned pattern."""
        # Find existing pattern
        existing_pattern = None
        for pattern in self.patterns:
            if (pattern.pattern_type == pattern_type and 
                pattern.conditions == conditions):
                existing_pattern = pattern
                break
        
        if existing_pattern:
            # Update existing pattern
            existing_pattern.samples += 1
            alpha = 0.1
            existing_pattern.adjustment_factor = (
                (1 - alpha) * existing_pattern.adjustment_factor + 
                alpha * adjustment
            )
            existing_pattern.confidence = min(
                existing_pattern.confidence + 0.05, 
                0.95
            )
            existing_pattern.last_updated = datetime.now().isoformat()
        else:
            # Create new pattern
            new_pattern = PredictionPattern(
                pattern_type=pattern_type,
                conditions=conditions,
                adjustment_factor=adjustment,
                confidence=0.5,
                samples=1,
                last_updated=datetime.now().isoformat()
            )
            self.patterns.append(new_pattern)
    
    def _calculate_category_success_rate(self, category: str) -> float:
        """Calculate success rate for a specific category."""
        cat_data = self.learning_data.get('category_success', {}).get(category)
        if cat_data and cat_data['total'] > 0:
            return cat_data['successes'] / cat_data['total']
        return 0.5  # Default neutral rate
    
    def _update_metrics(self):
        """Update overall learning metrics."""
        # Update cycle time from execution log
        if self.execution_log.exists():
            try:
                with open(self.execution_log, 'r') as f:
                    executions = json.load(f)
                
                if executions:
                    # Calculate average cycle time from recent executions
                    recent_times = []
                    for execution in executions[-20:]:  # Last 20 executions
                        exec_time = execution.get('result', {}).get('execution_time', 0)
                        if exec_time > 0:
                            recent_times.append(exec_time)
                    
                    if recent_times:
                        self.metrics.average_cycle_time = statistics.mean(recent_times)
            except Exception as e:
                logger.warning(f"Failed to update cycle time metrics: {e}")
        
        # Update model confidence based on recent accuracy
        recent_accuracy = (
            self.metrics.estimation_accuracy * 0.4 +
            self.metrics.value_prediction_accuracy * 0.4 +
            self.metrics.execution_success_rate * 0.2
        )
        self.metrics.model_confidence = recent_accuracy
        
        # Update adaptation cycles
        self.metrics.adaptation_cycles += 1
    
    def apply_learned_adjustments(self, work_item: Dict) -> Dict:
        """Apply learned patterns to adjust work item scoring."""
        adjusted_item = work_item.copy()
        
        # Apply pattern-based adjustments
        for pattern in self.patterns:
            if self._pattern_matches(pattern, work_item):
                # Apply adjustment to composite score
                current_score = adjusted_item.get('composite_score', 0)
                adjustment = pattern.adjustment_factor * pattern.confidence
                adjusted_item['composite_score'] = current_score * adjustment
                
                logger.debug(f"Applied pattern {pattern.pattern_type}: "
                           f"{current_score:.1f} -> {adjusted_item['composite_score']:.1f}")
        
        # Apply category-specific adjustments
        category = work_item.get('category', 'unknown')
        cat_patterns = self.learning_data.get('category_patterns', {}).get(category)
        
        if cat_patterns and cat_patterns['count'] > 5:  # Sufficient data
            # Adjust effort estimation
            learned_avg_effort = cat_patterns['avg_effort']
            current_effort = adjusted_item.get('estimated_effort', 0)
            
            if learned_avg_effort > 0:
                effort_ratio = learned_avg_effort / max(current_effort, 0.1)
                # Dampen extreme adjustments
                effort_adjustment = 1.0 + (effort_ratio - 1.0) * 0.3
                adjusted_item['estimated_effort'] = current_effort * effort_adjustment
            
            # Adjust impact based on category success rate
            success_rate = cat_patterns['success_rate']
            impact_adjustment = 0.5 + success_rate * 0.5  # Range: 0.5 to 1.0
            current_impact = adjusted_item.get('impact_score', 0)
            adjusted_item['impact_score'] = current_impact * impact_adjustment
        
        return adjusted_item
    
    def _pattern_matches(self, pattern: PredictionPattern, work_item: Dict) -> bool:
        """Check if a pattern matches a work item."""
        for key, value in pattern.conditions.items():
            if key == 'min_score':
                if work_item.get('composite_score', 0) < value:
                    return False
            elif key == 'max_effort':
                if work_item.get('estimated_effort', 0) > value:
                    return False
            elif key == 'category':
                if work_item.get('category') != value:
                    return False
            else:
                if work_item.get(key) != value:
                    return False
        return True
    
    def calibrate_model(self):
        """Perform model recalibration based on learning data."""
        logger.info("Starting model calibration...")
        
        # Recalibrate effort estimation
        self._calibrate_effort_estimation()
        
        # Recalibrate value prediction
        self._calibrate_value_prediction()
        
        # Clean up old patterns
        self._cleanup_patterns()
        
        # Update calibration timestamp
        self.metrics.last_calibration = datetime.now().isoformat()
        
        # Save updated data
        self._save_learning_data()
        
        logger.info("Model calibration completed")
    
    def _calibrate_effort_estimation(self):
        """Recalibrate effort estimation model."""
        effort_data = self.learning_data.get('effort_estimation', [])
        
        if len(effort_data) < 10:
            return  # Insufficient data
        
        # Calculate recent accuracy trend
        recent_data = effort_data[-20:]  # Last 20 estimations
        recent_ratios = [item['ratio'] for item in recent_data]
        
        if recent_ratios:
            mean_ratio = statistics.mean(recent_ratios)
            
            # Update estimation confidence
            if mean_ratio > 0.8:
                self.metrics.estimation_accuracy = min(
                    self.metrics.estimation_accuracy * 1.05, 0.95
                )
            elif mean_ratio < 0.6:
                self.metrics.estimation_accuracy = max(
                    self.metrics.estimation_accuracy * 0.95, 0.3
                )
    
    def _calibrate_value_prediction(self):
        """Recalibrate value prediction model."""
        value_data = self.learning_data.get('value_prediction', [])
        
        if len(value_data) < 10:
            return  # Insufficient data
        
        # Calculate recent prediction accuracy
        recent_data = value_data[-20:]
        recent_accuracies = [item['accuracy'] for item in recent_data]
        
        if recent_accuracies:
            mean_accuracy = statistics.mean(recent_accuracies)
            self.metrics.value_prediction_accuracy = mean_accuracy
    
    def _cleanup_patterns(self):
        """Remove outdated or low-confidence patterns."""
        cutoff_date = datetime.now() - timedelta(days=30)
        
        # Remove old patterns with low confidence
        self.patterns = [
            pattern for pattern in self.patterns
            if (pattern.confidence > 0.3 and 
                datetime.fromisoformat(pattern.last_updated) > cutoff_date)
        ]
    
    def _save_learning_data(self):
        """Save all learning data to disk."""
        try:
            # Save main learning data
            with open(self.learning_file, 'w') as f:
                json.dump(self.learning_data, f, indent=2)
            
            # Save patterns
            patterns_data = [asdict(pattern) for pattern in self.patterns]
            with open(self.patterns_file, 'w') as f:
                json.dump(patterns_data, f, indent=2)
            
            # Save metrics
            with open(self.metrics_file, 'w') as f:
                json.dump(asdict(self.metrics), f, indent=2)
            
            logger.debug("Learning data saved successfully")
            
        except Exception as e:
            logger.error(f"Failed to save learning data: {e}")
    
    def get_learning_report(self) -> Dict:
        """Generate a comprehensive learning report."""
        return {
            'metrics': asdict(self.metrics),
            'total_patterns': len(self.patterns),
            'category_insights': self.learning_data.get('category_patterns', {}),
            'recent_accuracy_trend': self._get_recent_accuracy_trend(),
            'top_patterns': [
                asdict(pattern) for pattern in 
                sorted(self.patterns, key=lambda p: p.confidence, reverse=True)[:5]
            ],
            'recommendations': self._generate_recommendations()
        }
    
    def _get_recent_accuracy_trend(self) -> Dict:
        """Calculate recent accuracy trends."""
        effort_data = self.learning_data.get('effort_estimation', [])
        value_data = self.learning_data.get('value_prediction', [])
        
        trend = {}
        
        if len(effort_data) >= 10:
            recent_effort = [item['ratio'] for item in effort_data[-10:]]
            trend['effort_trend'] = 'improving' if recent_effort[-1] > recent_effort[0] else 'declining'
            trend['effort_variance'] = statistics.variance(recent_effort) if len(recent_effort) > 1 else 0
        
        if len(value_data) >= 10:
            recent_value = [item['accuracy'] for item in value_data[-10:]]
            trend['value_trend'] = 'improving' if recent_value[-1] > recent_value[0] else 'declining'
            trend['value_variance'] = statistics.variance(recent_value) if len(recent_value) > 1 else 0
        
        return trend
    
    def _generate_recommendations(self) -> List[str]:
        """Generate actionable recommendations based on learning."""
        recommendations = []
        
        # Accuracy-based recommendations
        if self.metrics.estimation_accuracy < 0.7:
            recommendations.append(
                "Consider breaking down large work items for better effort estimation"
            )
        
        if self.metrics.execution_success_rate < 0.8:
            recommendations.append(
                "Review quality gates and execution procedures to improve success rate"
            )
        
        # Pattern-based recommendations
        high_confidence_patterns = [p for p in self.patterns if p.confidence > 0.8]
        if len(high_confidence_patterns) > 5:
            recommendations.append(
                "Consider codifying high-confidence patterns into scoring rules"
            )
        
        # Category-specific recommendations
        cat_patterns = self.learning_data.get('category_patterns', {})
        for category, data in cat_patterns.items():
            if data['success_rate'] < 0.6 and data['count'] > 10:
                recommendations.append(
                    f"Review execution strategy for {category} items (low success rate)"
                )
        
        return recommendations


if __name__ == "__main__":
    learning_engine = ContinuousLearningEngine()
    
    # Generate learning report
    report = learning_engine.get_learning_report()
    print(json.dumps(report, indent=2))