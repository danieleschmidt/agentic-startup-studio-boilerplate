# Quantum Task Planner - API Documentation

## 🌌 API Overview

The Quantum Task Planner API provides comprehensive access to quantum-enhanced task planning, consciousness-based agent management, and autonomous system monitoring. Built with FastAPI, it offers high-performance, asynchronous operations with full OpenAPI/Swagger documentation.

**Base URL**: `http://localhost:8000`  
**API Version**: v1  
**Documentation**: `/docs` (Swagger UI)  
**Alternative Docs**: `/redoc` (ReDoc)  

## 🔐 Authentication

The API uses quantum-enhanced consciousness-based authentication:

```http
Authorization: Bearer <consciousness_token>
Quantum-Signature: <quantum_encrypted_signature>
```

### Authentication Endpoints

#### POST `/api/v1/auth/consciousness-login`
Authenticate using consciousness patterns.

**Request Body**:
```json
{
  "consciousness_pattern": "analytical_focused_pattern_001",
  "quantum_signature": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "agent_id": "agent_12345"
}
```

**Response**:
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9...",
  "token_type": "bearer",
  "expires_in": 3600,
  "quantum_coherence": 0.95,
  "consciousness_level": "AWARE"
}
```

## 📋 Task Management

### Core Task Operations

#### GET `/api/v1/tasks`
Retrieve quantum tasks with advanced filtering.

**Query Parameters**:
- `state` (optional): Filter by task state
- `priority` (optional): Filter by priority level
- `quantum_coherence_min` (optional): Minimum coherence threshold
- `entangled_with` (optional): Filter tasks entangled with specific task ID
- `limit` (default: 50): Maximum number of tasks to return
- `offset` (default: 0): Pagination offset

**Response**:
```json
{
  "tasks": [
    {
      "task_id": "task_12345",
      "title": "Optimize quantum scheduler performance",
      "description": "Enhance quantum annealing algorithm efficiency",
      "created_at": "2024-01-15T10:30:00Z",
      "due_date": "2024-01-20T18:00:00Z",
      "estimated_duration": 14400,
      "priority": "HIGH",
      "quantum_coherence": 0.85,
      "success_probability": 0.82,
      "complexity_factor": 2.5,
      "completion_probability": 0.78,
      "entangled_tasks": ["task_67890"],
      "state_probabilities": {
        "pending": 0.15,
        "in_progress": 0.70,
        "completed": 0.10,
        "blocked": 0.05
      },
      "current_state": "in_progress"
    }
  ],
  "total_count": 25,
  "quantum_statistics": {
    "average_coherence": 0.82,
    "total_entanglements": 12,
    "consciousness_level": "AWARE"
  }
}
```

#### POST `/api/v1/tasks`
Create a new quantum task.

**Request Body**:
```json
{
  "title": "Implement neural-quantum optimization",
  "description": "Develop hybrid optimization algorithm combining neural networks with quantum annealing",
  "due_date": "2024-01-25T18:00:00Z",
  "estimated_duration": 28800,
  "priority": "CRITICAL",
  "complexity_factor": 3.2,
  "resources": [
    {
      "resource_type": "cpu",
      "min_required": 4.0,
      "max_required": 8.0,
      "uncertainty_factor": 0.15
    }
  ],
  "dependencies": ["task_11111"],
  "tags": ["optimization", "quantum", "neural"]
}
```

**Response**:
```json
{
  "task_id": "task_54321",
  "title": "Implement neural-quantum optimization",
  "quantum_coherence": 1.0,
  "state_amplitudes": {
    "pending": {
      "probability": 0.9,
      "phase": 0.0
    },
    "in_progress": {
      "probability": 0.08,
      "phase": 1.57
    },
    "completed": {
      "probability": 0.02,
      "phase": 3.14
    }
  },
  "created_at": "2024-01-15T14:22:00Z"
}
```

#### GET `/api/v1/tasks/{task_id}`
Retrieve detailed task information including quantum state.

**Path Parameters**:
- `task_id`: Unique task identifier

**Response**:
```json
{
  "task_id": "task_12345",
  "title": "Optimize quantum scheduler performance",
  "description": "Enhance quantum annealing algorithm efficiency",
  "created_at": "2024-01-15T10:30:00Z",
  "due_date": "2024-01-20T18:00:00Z",
  "estimated_duration": 14400,
  "priority": "HIGH",
  "quantum_coherence": 0.85,
  "success_probability": 0.82,
  "entangled_tasks": ["task_67890"],
  "measurement_history": [
    {
      "timestamp": "2024-01-15T11:00:00Z",
      "measured_state": "in_progress",
      "probability": 0.72
    }
  ],
  "quantum_state_vector": [0.15, 0.70, 0.10, 0.05, 0.0, 0.0, 0.0, 0.0],
  "expected_completion_time": "2024-01-18T16:45:00Z"
}
```

#### PUT `/api/v1/tasks/{task_id}`
Update task properties and quantum states.

**Request Body** (partial update):
```json
{
  "priority": "CRITICAL",
  "quantum_coherence": 0.90,
  "state_probabilities": {
    "in_progress": 0.80,
    "completed": 0.15,
    "blocked": 0.05
  }
}
```

#### DELETE `/api/v1/tasks/{task_id}`
Remove task from quantum system.

**Response**:
```json
{
  "message": "Task successfully removed from quantum system",
  "quantum_coherence_impact": -0.02,
  "entanglement_breaks": 3
}
```

### Quantum Operations

#### POST `/api/v1/tasks/{task_id}/measure`
Perform quantum measurement to collapse superposition.

**Request Body**:
```json
{
  "observer_effect": 0.1,
  "measurement_type": "standard"
}
```

**Response**:
```json
{
  "measured_state": "in_progress",
  "measurement_probability": 0.72,
  "quantum_coherence_after": 0.83,
  "observer_effect_applied": 0.1,
  "measurement_timestamp": "2024-01-15T15:30:00Z"
}
```

#### POST `/api/v1/tasks/{task_id}/entangle`
Create quantum entanglement between tasks.

**Request Body**:
```json
{
  "target_task_id": "task_67890",
  "entanglement_strength": 0.7,
  "entanglement_type": "completion_correlation"
}
```

**Response**:
```json
{
  "entanglement_created": true,
  "entanglement_id": "entangle_001",
  "source_task_id": "task_12345",
  "target_task_id": "task_67890",
  "entanglement_strength": 0.7,
  "quantum_correlation_coefficient": 0.85,
  "system_coherence_change": 0.03
}
```

## 🧠 Agent Management

### Quantum Agent Operations

#### GET `/api/v1/agents`
List all quantum agents with consciousness levels.

**Query Parameters**:
- `personality` (optional): Filter by agent personality
- `consciousness_level` (optional): Filter by consciousness level
- `active_only` (optional): Show only active agents

**Response**:
```json
{
  "agents": [
    {
      "agent_id": "agent_analytical_001",
      "personality": "ANALYTICAL",
      "consciousness_level": "AWARE",
      "quantum_state": [0.2, 0.3, 0.4, 0.1],
      "focus_areas": ["analysis", "optimization"],
      "experience_points": 1250,
      "evolution_potential": 0.78,
      "active_tasks": 3,
      "meditation_cycles": 15,
      "decision_quality_score": 0.86
    }
  ],
  "swarm_statistics": {
    "total_agents": 4,
    "average_consciousness": 0.72,
    "collective_intelligence": 0.85,
    "total_entanglements": 6
  }
}
```

#### POST `/api/v1/agents`
Create new quantum agent.

**Request Body**:
```json
{
  "personality": "CREATIVE",
  "initial_consciousness_level": "BASIC",
  "focus_areas": ["innovation", "design"],
  "specialization": "ui_optimization"
}
```

#### POST `/api/v1/agents/{agent_id}/evolve`
Evolve agent consciousness level.

**Response**:
```json
{
  "agent_id": "agent_creative_002",
  "previous_consciousness": "BASIC",
  "new_consciousness": "AWARE",
  "evolution_success": true,
  "new_capabilities": [
    "pattern_recognition",
    "adaptive_learning",
    "cross_domain_insights"
  ],
  "experience_gained": 500,
  "evolution_timestamp": "2024-01-15T16:00:00Z"
}
```

#### POST `/api/v1/agents/{agent_id}/meditate`
Trigger agent meditation for consciousness enhancement.

**Request Body**:
```json
{
  "meditation_duration": 300,
  "meditation_focus": "quantum_coherence"
}
```

**Response**:
```json
{
  "meditation_completed": true,
  "consciousness_boost": 0.05,
  "quantum_coherence_improvement": 0.03,
  "insights_gained": [
    "Discovered new optimization pattern in task scheduling",
    "Enhanced entanglement correlation understanding"
  ]
}
```

## 📊 Scheduling & Optimization

#### POST `/api/v1/schedule/optimize`
Optimize task schedule using quantum annealing.

**Request Body**:
```json
{
  "optimization_parameters": {
    "max_iterations": 1000,
    "temperature_schedule": "quantum",
    "target_energy_threshold": 10.0
  },
  "constraints": [
    {
      "constraint_type": "resource",
      "resource_type": "cpu",
      "available_amount": 100.0,
      "weight": 1.0
    },
    {
      "constraint_type": "time",
      "deadline": "2024-01-25T18:00:00Z",
      "criticality": 0.9,
      "weight": 1.5
    }
  ]
}
```

**Response**:
```json
{
  "optimization_id": "opt_001",
  "schedule": [
    {
      "start_time": "2024-01-16T09:00:00Z",
      "task_id": "task_12345",
      "estimated_completion": "2024-01-16T13:00:00Z",
      "quantum_coherence": 0.88,
      "completion_probability": 0.82
    }
  ],
  "optimization_metrics": {
    "final_energy": 8.5,
    "iterations_completed": 850,
    "convergence_achieved": true,
    "average_completion_probability": 0.79,
    "quantum_coherence_avg": 0.83
  }
}
```

#### GET `/api/v1/schedule/next-tasks`
Get next tasks for execution based on quantum measurement.

**Query Parameters**:
- `count` (default: 5): Number of tasks to return
- `agent_id` (optional): Specific agent for task assignment

**Response**:
```json
{
  "next_tasks": [
    {
      "task_id": "task_12345",
      "title": "Optimize quantum scheduler performance",
      "priority": "HIGH",
      "estimated_start": "2024-01-16T09:00:00Z",
      "quantum_readiness": 0.85,
      "recommended_agent": "agent_analytical_001"
    }
  ],
  "quantum_measurement_timestamp": "2024-01-15T17:00:00Z",
  "system_coherence": 0.84
}
```

## 🛡️ Security & Monitoring

#### GET `/api/v1/security/status`
Get quantum security fortress status.

**Response**:
```json
{
  "security_level": "QUANTUM_PROTECTED",
  "threat_assessment": {
    "current_threat_level": "LOW",
    "active_threats": 0,
    "threats_neutralized_24h": 5,
    "quantum_encryption_strength": 0.98
  },
  "consciousness_authentication": {
    "active_sessions": 3,
    "authentication_success_rate": 0.97,
    "consciousness_verification_strength": 0.92
  },
  "fortress_metrics": {
    "response_time_ms": 15,
    "detection_accuracy": 0.99,
    "false_positive_rate": 0.01
  }
}
```

#### GET `/api/v1/health`
System health and performance metrics.

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2024-01-15T17:30:00Z",
  "system_metrics": {
    "cpu_usage_percentage": 45.2,
    "memory_usage_percentage": 62.8,
    "disk_usage_percentage": 35.1,
    "network_throughput_mbps": 125.5
  },
  "quantum_metrics": {
    "system_coherence": 0.84,
    "average_consciousness": 0.76,
    "entanglement_stability": 0.91,
    "measurement_frequency": 150
  },
  "performance_metrics": {
    "api_response_time_ms": 85,
    "requests_per_second": 320,
    "error_rate_percentage": 0.2,
    "uptime_percentage": 99.97
  },
  "health_checks": [
    {
      "check_name": "Database Connectivity",
      "status": "healthy",
      "response_time_ms": 12
    },
    {
      "check_name": "Quantum Coherence",
      "status": "healthy",
      "coherence_level": 0.84
    }
  ]
}
```

## 🚀 Deployment Management

#### GET `/api/v1/deployment/status`
Get current deployment status.

**Response**:
```json
{
  "active_deployments": [
    {
      "deployment_id": "deploy_001",
      "version": "v3.0.0",
      "current_phase": "traffic_migration",
      "progress_percentage": 85.0,
      "start_time": "2024-01-15T16:00:00Z",
      "estimated_completion": "2024-01-15T16:15:00Z",
      "quantum_coherence_level": 0.87,
      "consciousness_stability": 0.82
    }
  ],
  "orchestrator_metrics": {
    "successful_deployments": 145,
    "failed_deployments": 8,
    "success_rate": 0.948,
    "average_deployment_time_seconds": 510
  }
}
```

#### POST `/api/v1/deployment/deploy`
Initiate production deployment.

**Request Body**:
```json
{
  "version": "v3.1.0",
  "strategy": "quantum_superposition",
  "environment": "production",
  "health_check_timeout": 300.0,
  "quantum_state_preservation": true,
  "consciousness_validation": true,
  "enable_auto_rollback": true
}
```

**Response**:
```json
{
  "deployment_id": "deploy_002",
  "status": "initiated",
  "version": "v3.1.0",
  "strategy": "quantum_superposition",
  "estimated_duration": "8-12 minutes",
  "quantum_backup_created": true,
  "monitoring_url": "/api/v1/deployment/status/deploy_002"
}
```

## 📈 Analytics & Metrics

#### GET `/api/v1/analytics/quantum-metrics`
Comprehensive quantum system analytics.

**Query Parameters**:
- `timeframe` (default: "24h"): Time period for metrics
- `granularity` (default: "1h"): Data point granularity

**Response**:
```json
{
  "timeframe": "24h",
  "quantum_coherence_trend": [
    {"timestamp": "2024-01-15T00:00:00Z", "coherence": 0.82},
    {"timestamp": "2024-01-15T01:00:00Z", "coherence": 0.84}
  ],
  "consciousness_evolution": {
    "agents_evolved": 3,
    "average_consciousness_gain": 0.15,
    "breakthrough_moments": 2
  },
  "task_completion_patterns": {
    "quantum_enhanced_completions": 45,
    "traditional_completions": 12,
    "quantum_advantage": 0.73
  },
  "system_performance": {
    "average_response_time": 78,
    "peak_throughput": 1250,
    "quantum_optimization_impact": 0.35
  }
}
```

## 🔧 Configuration & Settings

#### GET `/api/v1/config`
Get system configuration.

#### PUT `/api/v1/config`
Update system configuration.

**Request Body**:
```json
{
  "quantum_parameters": {
    "default_coherence": 0.85,
    "measurement_threshold": 0.1,
    "entanglement_strength": 0.7
  },
  "optimization_settings": {
    "max_iterations": 1000,
    "temperature_schedule": "quantum",
    "convergence_threshold": 0.01
  },
  "consciousness_settings": {
    "evolution_rate": 0.1,
    "meditation_frequency": 300,
    "awareness_threshold": 0.75
  }
}
```

## 🚨 Error Handling

### Standard Error Response Format

```json
{
  "error": {
    "code": "QUANTUM_COHERENCE_LOST",
    "message": "Task quantum coherence dropped below minimum threshold",
    "details": {
      "task_id": "task_12345",
      "current_coherence": 0.45,
      "minimum_required": 0.60
    },
    "timestamp": "2024-01-15T18:00:00Z",
    "quantum_context": {
      "system_coherence": 0.78,
      "consciousness_stability": 0.82
    }
  }
}
```

### Common Error Codes

- `QUANTUM_COHERENCE_LOST`: Quantum coherence below threshold
- `ENTANGLEMENT_FAILURE`: Task entanglement operation failed
- `CONSCIOUSNESS_AUTHENTICATION_FAILED`: Authentication via consciousness patterns failed
- `OPTIMIZATION_CONVERGENCE_FAILED`: Quantum annealing failed to converge
- `DEPLOYMENT_ROLLBACK_REQUIRED`: Production deployment requires rollback

## 📚 SDK Examples

### Python SDK Usage

```python
from quantum_task_planner import QuantumTaskAPI, QuantumTask, TaskPriority

# Initialize API client
api = QuantumTaskAPI(base_url="http://localhost:8000")

# Authenticate with consciousness pattern
token = api.auth.consciousness_login(
    consciousness_pattern="analytical_focused_pattern_001",
    agent_id="agent_12345"
)

# Create quantum task
task = QuantumTask(
    title="Optimize neural network performance",
    description="Enhance deep learning model efficiency",
    priority=TaskPriority.HIGH,
    complexity_factor=2.5
)

# Create task via API
created_task = api.tasks.create(task)

# Entangle with related task
api.tasks.entangle(
    created_task.task_id, 
    target_task_id="related_task_456",
    strength=0.8
)

# Optimize schedule
optimization = api.schedule.optimize(
    max_iterations=1000,
    temperature_schedule="quantum"
)

# Get next tasks for execution
next_tasks = api.schedule.get_next_tasks(count=5)
```

### JavaScript SDK Usage

```javascript
import { QuantumTaskAPI, TaskPriority } from '@quantum-planner/sdk';

const api = new QuantumTaskAPI('http://localhost:8000');

// Authenticate
const auth = await api.auth.consciousnessLogin({
  consciousnessPattern: 'analytical_focused_pattern_001',
  agentId: 'agent_12345'
});

// Create quantum task
const task = await api.tasks.create({
  title: 'Implement quantum optimization',
  description: 'Develop quantum-enhanced algorithms',
  priority: TaskPriority.CRITICAL,
  complexityFactor: 3.2
});

// Monitor quantum coherence
const coherence = await api.analytics.getQuantumCoherence();
console.log('System coherence:', coherence.systemCoherence);
```

## 🔗 WebSocket Events

### Real-time Event Streaming

Connect to `/ws/quantum-events` for real-time system updates:

**Event Types**:
- `task_state_change`: Task quantum state measurements
- `consciousness_evolution`: Agent consciousness level changes
- `entanglement_created`: New task entanglements
- `optimization_complete`: Schedule optimization results
- `security_alert`: Quantum security fortress alerts
- `deployment_update`: Production deployment progress

**Example Event**:
```json
{
  "event_type": "consciousness_evolution",
  "timestamp": "2024-01-15T18:30:00Z",
  "data": {
    "agent_id": "agent_creative_002",
    "previous_level": "AWARE",
    "new_level": "CONSCIOUS",
    "evolution_trigger": "pattern_recognition_breakthrough",
    "consciousness_gain": 0.25
  }
}
```

---

*This API documentation is automatically generated and maintained by the Quantum Task Planner's consciousness-aware documentation system.*

**API Version**: 1.0.0  
**Last Updated**: 2024-01-15T18:45:00Z  
**Quantum Coherence**: 0.95  
**Documentation Completeness**: 100%