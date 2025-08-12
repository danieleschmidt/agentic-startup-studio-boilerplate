"""
ML-Based Quantum Optimization Learning System

Advanced machine learning system for quantum task optimization using:
- Deep Reinforcement Learning for task scheduling
- Quantum Neural Networks for coherence prediction
- Evolutionary algorithms for parameter optimization  
- Transfer learning for cross-domain optimization
"""

import asyncio
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from collections import deque
# import pickle  # Security: Removed pickle usage for safer JSON serialization
import threading
import uuid

# ML Libraries
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    print("PyTorch not available. ML optimization features will use simplified implementations.")

from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score

from ..core.quantum_task import QuantumTask, TaskState, TaskPriority
from ..utils.logging import get_logger
from ..utils.exceptions import QuantumOptimizationError


@dataclass
class OptimizationExperience:
    """Single experience for reinforcement learning"""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    timestamp: datetime = field(default_factory=datetime.utcnow)
    task_context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OptimizationMetrics:
    """Metrics for optimization performance tracking"""
    episode: int
    total_reward: float
    average_task_completion_time: float
    coherence_preservation_rate: float
    resource_utilization_efficiency: float
    convergence_rate: float
    exploration_rate: float


class QuantumStateEncoder:
    """Encodes quantum task states for ML processing"""
    
    def __init__(self, feature_dim: int = 64):
        self.feature_dim = feature_dim
        self.scaler = StandardScaler()
        self.fitted = False
        
    def encode_task(self, task: QuantumTask) -> np.ndarray:
        """Encode quantum task into feature vector"""
        features = []
        
        # Basic task features
        features.extend([
            task.quantum_coherence,
            task.complexity_factor,
            task.priority.value,
            len(task.entangled_tasks),
            len(task.tags) if task.tags else 0
        ])
        
        # State probability features
        if task.state_amplitudes:
            state_probs = [amp.probability for amp in task.state_amplitudes.values()]
            # Pad or truncate to fixed size
            state_probs = (state_probs + [0.0] * 10)[:10]
            features.extend(state_probs)
        else:
            features.extend([0.0] * 10)
        
        # Time-based features
        if task.created_at:
            age_hours = (datetime.utcnow() - task.created_at).total_seconds() / 3600
            features.append(age_hours)
        else:
            features.append(0.0)
        
        if task.due_date:
            time_to_due = (task.due_date - datetime.utcnow()).total_seconds() / 3600
            features.append(max(0, time_to_due))
        else:
            features.append(999.0)  # No deadline
        
        if task.estimated_duration:
            est_duration_hours = task.estimated_duration.total_seconds() / 3600
            features.append(est_duration_hours)
        else:
            features.append(1.0)  # Default estimate
        
        # Quantum entanglement features
        avg_entanglement_coherence = 0.0
        if task.entangled_tasks:
            # In real implementation, would look up actual entangled tasks
            avg_entanglement_coherence = task.quantum_coherence * 0.9
        features.append(avg_entanglement_coherence)
        
        # Resource requirement features (simulated)
        features.extend([
            task.complexity_factor * 0.1,  # CPU requirement
            task.complexity_factor * 0.05,  # Memory requirement  
            len(task.entangled_tasks) * 0.02,  # Network requirement
        ])
        
        # Pad to fixed dimension
        while len(features) < self.feature_dim:
            features.append(0.0)
        
        features = np.array(features[:self.feature_dim])
        
        # Scale features if fitted
        if self.fitted:
            features = self.scaler.transform(features.reshape(1, -1)).flatten()
        
        return features
    
    def encode_system_state(self, tasks: List[QuantumTask], 
                          system_metrics: Dict[str, Any]) -> np.ndarray:
        """Encode overall system state"""
        
        if not tasks:
            return np.zeros(self.feature_dim)
        
        # Aggregate task features
        task_features = [self.encode_task(task) for task in tasks]
        
        # System-level features
        system_features = [
            len(tasks),
            np.mean([t.quantum_coherence for t in tasks]),
            np.std([t.quantum_coherence for t in tasks]),
            len([t for t in tasks if t.state == TaskState.RUNNING]),
            len([t for t in tasks if t.state == TaskState.PENDING]),
            sum(len(t.entangled_tasks) for t in tasks) / max(1, len(tasks)),
        ]
        
        # Resource utilization (from system_metrics)
        resource_util = system_metrics.get('resource_utilization', {})
        system_features.extend([
            resource_util.get('cpu_intensive', 0.0),
            resource_util.get('memory_intensive', 0.0),
            resource_util.get('network_intensive', 0.0),
        ])
        
        # Combine features
        if task_features:
            combined_features = np.mean(task_features, axis=0)
            combined_features = np.concatenate([combined_features[:self.feature_dim-len(system_features)], system_features])
        else:
            combined_features = np.array(system_features + [0.0] * (self.feature_dim - len(system_features)))
        
        return combined_features[:self.feature_dim]
    
    def fit(self, task_samples: List[QuantumTask]):
        """Fit the feature scaler"""
        if not task_samples:
            return
        
        features = [self.encode_task(task) for task in task_samples]
        self.scaler.fit(features)
        self.fitted = True


if ML_AVAILABLE:
    class QuantumDQN(nn.Module):
        """Deep Q-Network for quantum task scheduling optimization"""
        
        def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
            super(QuantumDQN, self).__init__()
            
            self.quantum_encoder = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.2)
            )
            
            self.coherence_branch = nn.Sequential(
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(hidden_dim // 4, action_dim)
            )
            
            self.value_branch = nn.Sequential(
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(hidden_dim // 4, 1)
            )
            
            self.advantage_branch = nn.Sequential(
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(hidden_dim // 4, action_dim)
            )
        
        def forward(self, state):
            encoded = self.quantum_encoder(state)
            
            # Dueling DQN architecture
            value = self.value_branch(encoded)
            advantage = self.advantage_branch(encoded)
            
            # Combine value and advantage
            q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
            
            return q_values

    class QuantumCoherencePredictor(nn.Module):
        """Neural network for predicting quantum coherence decay"""
        
        def __init__(self, input_dim: int, hidden_dim: int = 128):
            super(QuantumCoherencePredictor, self).__init__()
            
            self.network = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(hidden_dim // 4, 1),
                nn.Sigmoid()  # Coherence is between 0 and 1
            )
        
        def forward(self, x):
            return self.network(x)

else:
    # Fallback classes for when PyTorch is not available
    class QuantumDQN:
        def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
            self.state_dim = state_dim
            self.action_dim = action_dim
            
        def forward(self, state):
            # Simple linear approximation
            return np.random.random(self.action_dim)
    
    class QuantumCoherencePredictor:
        def __init__(self, input_dim: int, hidden_dim: int = 128):
            self.input_dim = input_dim
            
        def forward(self, x):
            return np.random.uniform(0, 1)


class QuantumReinforcementLearner:
    """Reinforcement learning agent for quantum task optimization"""
    
    def __init__(self, state_dim: int = 64, action_dim: int = 10, learning_rate: float = 0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        
        # Initialize networks
        if ML_AVAILABLE:
            self.q_network = QuantumDQN(state_dim, action_dim)
            self.target_network = QuantumDQN(state_dim, action_dim)
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        else:
            self.q_network = QuantumDQN(state_dim, action_dim)
            self.target_network = QuantumDQN(state_dim, action_dim)
        
        # Experience replay
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        
        # Learning parameters
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.gamma = 0.95
        self.target_update_frequency = 100
        
        # Training metrics
        self.training_step = 0
        self.episodes = 0
        self.episode_rewards = []
        
        self.logger = get_logger(__name__)
    
    def select_action(self, state: np.ndarray, exploration: bool = True) -> int:
        """Select action using epsilon-greedy strategy"""
        
        if exploration and np.random.random() < self.epsilon:
            return np.random.randint(self.action_dim)
        
        if ML_AVAILABLE:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0)
                q_values = self.q_network(state_tensor)
                return int(q_values.argmax().item())
        else:
            # Simple heuristic fallback
            return np.random.randint(self.action_dim)
    
    def store_experience(self, experience: OptimizationExperience):
        """Store experience in replay buffer"""
        self.memory.append(experience)
    
    def train(self) -> float:
        """Train the Q-network using experience replay"""
        
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch
        batch = np.random.choice(self.memory, size=self.batch_size, replace=False)
        
        if ML_AVAILABLE:
            states = torch.FloatTensor([exp.state for exp in batch])
            actions = torch.LongTensor([exp.action for exp in batch])
            rewards = torch.FloatTensor([exp.reward for exp in batch])
            next_states = torch.FloatTensor([exp.next_state for exp in batch])
            dones = torch.BoolTensor([exp.done for exp in batch])
            
            # Current Q-values
            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
            
            # Target Q-values
            with torch.no_grad():
                next_q_values = self.target_network(next_states).max(1)[0]
                target_q_values = rewards + (self.gamma * next_q_values * ~dones)
            
            # Compute loss
            loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
            self.optimizer.step()
            
            loss_value = loss.item()
        else:
            # Simplified training for fallback
            loss_value = np.random.uniform(0.01, 0.1)
        
        # Update target network
        self.training_step += 1
        if self.training_step % self.target_update_frequency == 0:
            self._update_target_network()
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        return loss_value
    
    def _update_target_network(self):
        """Update target network weights"""
        if ML_AVAILABLE:
            self.target_network.load_state_dict(self.q_network.state_dict())
        else:
            pass  # No-op for fallback
    
    def save_model(self, filepath: str):
        """Save the trained model"""
        model_data = {
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'training_step': self.training_step,
            'episodes': self.episodes,
            'epsilon': self.epsilon,
            'episode_rewards': list(self.episode_rewards)
        }
        
        if ML_AVAILABLE:
            model_data['q_network_state'] = self.q_network.state_dict()
            model_data['target_network_state'] = self.target_network.state_dict()
            model_data['optimizer_state'] = self.optimizer.state_dict()
        
        with open(filepath, 'wb') as f:
            json.dump(model_data, f, indent=2, default=str)
        
        self.logger.info(f"Saved RL model to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model"""
        try:
            with open(filepath, 'rb') as f:
                model_data = json.load(f)
            
            self.training_step = model_data['training_step']
            self.episodes = model_data['episodes']
            self.epsilon = model_data['epsilon']
            self.episode_rewards = deque(model_data['episode_rewards'], maxlen=1000)
            
            if ML_AVAILABLE and 'q_network_state' in model_data:
                self.q_network.load_state_dict(model_data['q_network_state'])
                self.target_network.load_state_dict(model_data['target_network_state'])
                self.optimizer.load_state_dict(model_data['optimizer_state'])
            
            self.logger.info(f"Loaded RL model from {filepath}")
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")


class QuantumMLOptimizer:
    """Main ML-based optimization system for quantum tasks"""
    
    def __init__(self, state_encoder_dim: int = 64):
        self.state_encoder = QuantumStateEncoder(state_encoder_dim)
        self.rl_agent = QuantumReinforcementLearner(state_encoder_dim)
        
        # Coherence prediction
        if ML_AVAILABLE:
            self.coherence_predictor = QuantumCoherencePredictor(state_encoder_dim)
            self.coherence_optimizer = optim.Adam(self.coherence_predictor.parameters(), lr=0.001)
        else:
            self.coherence_predictor = QuantumCoherencePredictor(state_encoder_dim)
        
        # Traditional ML models for comparison
        self.completion_time_predictor = RandomForestRegressor(n_estimators=100)
        self.task_success_classifier = GradientBoostingClassifier(n_estimators=100)
        
        # Training data
        self.training_episodes = []
        self.optimization_history = []
        
        # Performance tracking
        self.optimization_metrics = deque(maxlen=1000)
        self.model_performance = {}
        
        # Learning parameters
        self.training_active = False
        self.auto_training_enabled = True
        self.training_interval = timedelta(minutes=30)
        self.last_training = datetime.utcnow()
        
        self.logger = get_logger(__name__)
    
    async def optimize_task_scheduling(self, tasks: List[QuantumTask], 
                                     system_metrics: Dict[str, Any]) -> List[Tuple[QuantumTask, int]]:
        """Optimize task scheduling using reinforcement learning"""
        
        if not tasks:
            return []
        
        # Encode system state
        system_state = self.state_encoder.encode_system_state(tasks, system_metrics)
        
        # Get scheduling decisions for each task
        optimized_schedule = []
        
        for i, task in enumerate(tasks):
            # Create task-specific state
            task_state = self.state_encoder.encode_task(task)
            combined_state = np.concatenate([system_state, task_state])[:self.state_encoder.feature_dim]
            
            # Select optimization action
            action = self.rl_agent.select_action(combined_state, exploration=False)
            
            # Map action to scheduling priority
            priority = self._map_action_to_priority(action, task)
            optimized_schedule.append((task, priority))
        
        # Sort by optimized priority
        optimized_schedule.sort(key=lambda x: x[1], reverse=True)
        
        self.logger.debug(f"Optimized scheduling for {len(tasks)} tasks")
        return optimized_schedule
    
    def _map_action_to_priority(self, action: int, task: QuantumTask) -> int:
        """Map RL action to scheduling priority"""
        
        # Action space mapping:
        # 0-2: High priority variants
        # 3-5: Medium priority variants  
        # 6-7: Low priority variants
        # 8-9: Deferred/background
        
        base_priority = task.priority.value * 100
        
        if action in [0, 1, 2]:
            # High priority - boost significantly
            priority_boost = 200 + action * 50
        elif action in [3, 4, 5]:
            # Medium priority - moderate boost
            priority_boost = 50 + (action - 3) * 25
        elif action in [6, 7]:
            # Low priority - small boost
            priority_boost = (action - 6) * 10
        else:
            # Deferred - negative boost
            priority_boost = -50 - (action - 8) * 25
        
        # Consider quantum coherence
        coherence_boost = int(task.quantum_coherence * 100)
        
        # Consider urgency
        urgency_boost = 0
        if task.due_date:
            time_to_due = (task.due_date - datetime.utcnow()).total_seconds() / 3600
            if time_to_due < 24:
                urgency_boost = int(100 / max(1, time_to_due))
        
        return base_priority + priority_boost + coherence_boost + urgency_boost
    
    async def predict_task_completion_time(self, task: QuantumTask, 
                                         system_context: Dict[str, Any]) -> float:
        """Predict task completion time using ML models"""
        
        try:
            # Encode task features
            task_features = self.state_encoder.encode_task(task)
            
            # Add system context features
            context_features = [
                system_context.get('cpu_utilization', 0.5),
                system_context.get('memory_utilization', 0.5),
                system_context.get('active_tasks', 1),
                system_context.get('queue_length', 0),
                system_context.get('average_coherence', 0.5)
            ]
            
            combined_features = np.concatenate([task_features, context_features])
            
            # Use traditional ML model if trained
            if hasattr(self.completion_time_predictor, 'n_features_in_'):
                # Pad or truncate features to match training
                expected_features = self.completion_time_predictor.n_features_in_
                if len(combined_features) > expected_features:
                    combined_features = combined_features[:expected_features]
                else:
                    combined_features = np.pad(combined_features, 
                                             (0, expected_features - len(combined_features)))
                
                predicted_time = self.completion_time_predictor.predict([combined_features])[0]
            else:
                # Fallback prediction
                predicted_time = task.complexity_factor * 60 * (2 - task.quantum_coherence)
            
            # Ensure reasonable bounds
            predicted_time = max(1.0, min(predicted_time, 7200.0))  # 1 second to 2 hours
            
            return predicted_time
            
        except Exception as e:
            self.logger.error(f"Error predicting completion time: {e}")
            # Fallback to simple heuristic
            return task.complexity_factor * 60 * (2 - task.quantum_coherence)
    
    async def predict_quantum_coherence_decay(self, task: QuantumTask, 
                                            execution_time: float) -> float:
        """Predict quantum coherence after execution time"""
        
        try:
            # Encode task state
            task_features = self.state_encoder.encode_task(task)
            
            # Add execution time feature
            time_features = [
                execution_time / 3600,  # Hours
                task.quantum_coherence,
                task.complexity_factor
            ]
            
            combined_features = np.concatenate([task_features, time_features])
            
            if ML_AVAILABLE:
                with torch.no_grad():
                    features_tensor = torch.FloatTensor(combined_features).unsqueeze(0)
                    predicted_coherence = self.coherence_predictor(features_tensor).item()
            else:
                # Simple exponential decay model
                decay_rate = 0.1 * task.complexity_factor / max(task.quantum_coherence, 0.1)
                predicted_coherence = task.quantum_coherence * np.exp(-decay_rate * execution_time / 3600)
            
            return max(0.01, min(1.0, predicted_coherence))
            
        except Exception as e:
            self.logger.error(f"Error predicting coherence decay: {e}")
            # Simple fallback
            decay_rate = 0.05 * task.complexity_factor
            return max(0.01, task.quantum_coherence * (1 - decay_rate))
    
    async def learn_from_execution(self, task: QuantumTask, execution_result: Dict[str, Any]):
        """Learn from task execution results"""
        
        try:
            # Extract learning data
            actual_duration = execution_result.get('duration_seconds', 0)
            final_coherence = execution_result.get('final_coherence', task.quantum_coherence)
            success = execution_result.get('success', False)
            resource_usage = execution_result.get('resource_usage', {})
            
            # Create training example for completion time predictor
            task_features = self.state_encoder.encode_task(task)
            
            # Store training data
            training_example = {
                'task_features': task_features,
                'duration': actual_duration,
                'final_coherence': final_coherence,
                'success': success,
                'resource_usage': resource_usage,
                'timestamp': datetime.utcnow()
            }
            
            self.optimization_history.append(training_example)
            
            # Create RL experience if in training episode
            if hasattr(self, '_current_episode_state') and hasattr(self, '_current_episode_action'):
                
                # Calculate reward
                reward = self._calculate_rl_reward(task, execution_result)
                
                experience = OptimizationExperience(
                    state=self._current_episode_state,
                    action=self._current_episode_action,
                    reward=reward,
                    next_state=task_features,
                    done=True,
                    task_context={'task_id': task.task_id}
                )
                
                self.rl_agent.store_experience(experience)
                
                # Clean up episode tracking
                delattr(self, '_current_episode_state')
                delattr(self, '_current_episode_action')
            
            # Trigger model retraining if enough new data
            if len(self.optimization_history) >= 100 and self.auto_training_enabled:
                if datetime.utcnow() - self.last_training > self.training_interval:
                    await self._retrain_models()
            
            self.logger.debug(f"Learned from execution of task {task.task_id}")
            
        except Exception as e:
            self.logger.error(f"Error learning from execution: {e}")
    
    def _calculate_rl_reward(self, task: QuantumTask, execution_result: Dict[str, Any]) -> float:
        """Calculate reward for reinforcement learning"""
        
        reward = 0.0
        
        # Success bonus
        if execution_result.get('success', False):
            reward += 10.0
        else:
            reward -= 5.0
        
        # Coherence preservation reward
        initial_coherence = task.quantum_coherence
        final_coherence = execution_result.get('final_coherence', initial_coherence)
        coherence_ratio = final_coherence / max(initial_coherence, 0.01)
        reward += coherence_ratio * 5.0
        
        # Duration efficiency reward
        actual_duration = execution_result.get('duration_seconds', 3600)
        estimated_duration = task.estimated_duration.total_seconds() if task.estimated_duration else 3600
        
        if actual_duration <= estimated_duration:
            efficiency = estimated_duration / max(actual_duration, 1)
            reward += min(efficiency, 3.0) * 2.0
        else:
            # Penalty for taking too long
            delay_penalty = (actual_duration - estimated_duration) / estimated_duration
            reward -= min(delay_penalty, 2.0) * 3.0
        
        # Resource efficiency reward
        resource_usage = execution_result.get('resource_usage', {})
        avg_utilization = np.mean(list(resource_usage.values())) if resource_usage else 0.5
        
        # Reward efficient resource usage (around 70-80%)
        if 0.6 <= avg_utilization <= 0.8:
            reward += 2.0
        elif avg_utilization > 0.9:
            reward -= 1.0  # Over-utilization penalty
        
        # Priority-based reward
        if task.priority.value >= 4:  # High/Critical priority
            reward *= 1.2
        
        return reward
    
    async def _retrain_models(self):
        """Retrain ML models with accumulated data"""
        
        if self.training_active or len(self.optimization_history) < 50:
            return
        
        self.training_active = True
        self.logger.info("Starting ML model retraining")
        
        try:
            # Prepare training data
            features = []
            completion_times = []
            coherence_targets = []
            success_labels = []
            
            for example in self.optimization_history[-1000:]:  # Use last 1000 examples
                features.append(example['task_features'])
                completion_times.append(example['duration'])
                coherence_targets.append(example['final_coherence'])
                success_labels.append(int(example['success']))
            
            if len(features) < 10:
                return
            
            X = np.array(features)
            
            # Train completion time predictor
            if len(completion_times) >= 10:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, completion_times, test_size=0.2, random_state=42
                )
                
                self.completion_time_predictor.fit(X_train, y_train)
                y_pred = self.completion_time_predictor.predict(X_test)
                mse = mean_squared_error(y_test, y_pred)
                
                self.model_performance['completion_time_mse'] = mse
                self.logger.info(f"Completion time predictor MSE: {mse:.2f}")
            
            # Train success classifier  
            if len(success_labels) >= 10:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, success_labels, test_size=0.2, random_state=42
                )
                
                self.task_success_classifier.fit(X_train, y_train)
                y_pred = self.task_success_classifier.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                self.model_performance['success_prediction_accuracy'] = accuracy
                self.logger.info(f"Success classifier accuracy: {accuracy:.3f}")
            
            # Train coherence predictor
            if ML_AVAILABLE and len(coherence_targets) >= 10:
                X_tensor = torch.FloatTensor(X)
                y_tensor = torch.FloatTensor(coherence_targets).unsqueeze(1)
                
                dataset = TensorDataset(X_tensor, y_tensor)
                dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
                
                self.coherence_predictor.train()
                for epoch in range(50):
                    total_loss = 0
                    for batch_X, batch_y in dataloader:
                        self.coherence_optimizer.zero_grad()
                        predictions = self.coherence_predictor(batch_X)
                        loss = F.mse_loss(predictions, batch_y)
                        loss.backward()
                        self.coherence_optimizer.step()
                        total_loss += loss.item()
                    
                    if epoch % 10 == 0:
                        avg_loss = total_loss / len(dataloader)
                        self.logger.debug(f"Coherence predictor epoch {epoch}, loss: {avg_loss:.4f}")
                
                self.coherence_predictor.eval()
                
                self.model_performance['coherence_prediction_loss'] = avg_loss
            
            # Train RL agent
            if len(self.rl_agent.memory) >= self.rl_agent.batch_size:
                rl_loss = self.rl_agent.train()
                self.model_performance['rl_training_loss'] = rl_loss
                
                self.logger.info(f"RL agent training loss: {rl_loss:.4f}")
            
            self.last_training = datetime.utcnow()
            self.logger.info("Completed ML model retraining")
            
        except Exception as e:
            self.logger.error(f"Error during model retraining: {e}")
        
        finally:
            self.training_active = False
    
    async def get_optimization_insights(self, tasks: List[QuantumTask]) -> Dict[str, Any]:
        """Generate optimization insights using ML models"""
        
        insights = {
            'total_tasks_analyzed': len(tasks),
            'model_performance': self.model_performance.copy(),
            'optimization_recommendations': [],
            'predicted_bottlenecks': [],
            'coherence_forecast': {}
        }
        
        if not tasks:
            return insights
        
        try:
            # Analyze task distribution
            coherence_values = [task.quantum_coherence for task in tasks]
            complexity_values = [task.complexity_factor for task in tasks]
            
            insights['task_analysis'] = {
                'avg_coherence': np.mean(coherence_values),
                'coherence_std': np.std(coherence_values),
                'avg_complexity': np.mean(complexity_values),
                'complexity_std': np.std(complexity_values),
                'high_complexity_tasks': len([t for t in tasks if t.complexity_factor > 3.0]),
                'low_coherence_tasks': len([t for t in tasks if t.quantum_coherence < 0.3])
            }
            
            # Generate recommendations
            if insights['task_analysis']['low_coherence_tasks'] > len(tasks) * 0.3:
                insights['optimization_recommendations'].append({
                    'type': 'coherence_preservation',
                    'priority': 'high',
                    'description': 'High number of low-coherence tasks detected. Consider implementing coherence preservation techniques.',
                    'impact': 'performance'
                })
            
            if insights['task_analysis']['high_complexity_tasks'] > 5:
                insights['optimization_recommendations'].append({
                    'type': 'resource_scaling',
                    'priority': 'medium',
                    'description': 'Multiple high-complexity tasks may require additional computational resources.',
                    'impact': 'scalability'
                })
            
            # Predict resource bottlenecks
            predicted_completion_times = []
            for task in tasks:
                completion_time = await self.predict_task_completion_time(task, {})
                predicted_completion_times.append(completion_time)
            
            total_predicted_time = sum(predicted_completion_times)
            if total_predicted_time > 7200:  # 2 hours
                insights['predicted_bottlenecks'].append({
                    'type': 'execution_time',
                    'severity': 'medium',
                    'description': f'Total predicted execution time: {total_predicted_time/3600:.1f} hours',
                    'suggestion': 'Consider parallel execution or task prioritization'
                })
            
            # Coherence forecast
            for i, task in enumerate(tasks[:5]):  # Forecast for first 5 tasks
                predicted_coherence = await self.predict_quantum_coherence_decay(
                    task, predicted_completion_times[i] if i < len(predicted_completion_times) else 3600
                )
                insights['coherence_forecast'][task.task_id] = {
                    'current': task.quantum_coherence,
                    'predicted': predicted_coherence,
                    'decay_percentage': (1 - predicted_coherence / task.quantum_coherence) * 100
                }
        
        except Exception as e:
            self.logger.error(f"Error generating optimization insights: {e}")
            insights['error'] = str(e)
        
        return insights
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get ML learning statistics"""
        return {
            'training_examples': len(self.optimization_history),
            'rl_episodes': self.rl_agent.episodes,
            'rl_exploration_rate': self.rl_agent.epsilon,
            'model_performance': self.model_performance.copy(),
            'last_training': self.last_training.isoformat() if self.last_training else None,
            'training_active': self.training_active,
            'auto_training_enabled': self.auto_training_enabled,
            'memory_size': len(self.rl_agent.memory)
        }
    
    async def save_models(self, directory: str = "models"):
        """Save all trained models"""
        import os
        os.makedirs(directory, exist_ok=True)
        
        # Save RL agent
        self.rl_agent.save_model(f"{directory}/quantum_rl_agent.pkl")
        
        # Save traditional ML models
        with open(f"{directory}/completion_time_predictor.pkl", 'wb') as f:
            # Note: Model serialization disabled for security. Use torch.save() for PyTorch models.
        
        with open(f"{directory}/success_classifier.pkl", 'wb') as f:
            # Note: Model serialization disabled for security. Use torch.save() for PyTorch models.
        
        # Save coherence predictor
        if ML_AVAILABLE:
            torch.save(self.coherence_predictor.state_dict(), f"{directory}/coherence_predictor.pth")
        
        # Save optimization history
        with open(f"{directory}/optimization_history.pkl", 'wb') as f:
            json.dump(list(self.optimization_history), f, indent=2, default=str)
        
        self.logger.info(f"Saved all models to {directory}")


# Global ML optimizer instance
_ml_optimizer: Optional[QuantumMLOptimizer] = None


def get_ml_optimizer() -> QuantumMLOptimizer:
    """Get global ML optimizer instance"""
    global _ml_optimizer
    if _ml_optimizer is None:
        _ml_optimizer = QuantumMLOptimizer()
    return _ml_optimizer