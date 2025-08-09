"""
Real-Time Quantum Metrics Dashboard

Advanced WebSocket-based dashboard for monitoring quantum task execution,
system performance, and quantum state evolution in real-time.
"""

import asyncio
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Set, Optional, Any, Callable
import logging
from dataclasses import dataclass, field, asdict
from enum import Enum

from fastapi import WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import HTMLResponse
import numpy as np
import plotly.graph_objects as go
import plotly.utils

from .quantum_api import app, scheduler, optimizer, entanglement_manager
from ..core.async_executor import get_quantum_executor
from ..utils.logging import get_logger


class MetricType(Enum):
    """Types of quantum metrics"""
    SYSTEM_OVERVIEW = "system_overview"
    TASK_EXECUTION = "task_execution"
    QUANTUM_COHERENCE = "quantum_coherence"
    ENTANGLEMENT_NETWORK = "entanglement_network"
    RESOURCE_UTILIZATION = "resource_utilization"
    PERFORMANCE_ANALYTICS = "performance_analytics"
    REAL_TIME_LOGS = "real_time_logs"


@dataclass
class DashboardMessage:
    """WebSocket message for dashboard communication"""
    message_type: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    data: Dict[str, Any] = field(default_factory=dict)
    client_id: Optional[str] = None
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps({
            "message_type": self.message_type,
            "timestamp": self.timestamp.isoformat(),
            "data": self.data,
            "client_id": self.client_id
        })


@dataclass
class DashboardClient:
    """Connected dashboard client"""
    client_id: str
    websocket: WebSocket
    subscribed_metrics: Set[MetricType] = field(default_factory=set)
    last_heartbeat: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)


class QuantumDashboardManager:
    """
    Real-time quantum metrics dashboard manager with WebSocket support
    """
    
    def __init__(self):
        self.active_clients: Dict[str, DashboardClient] = {}
        self.metric_history: Dict[MetricType, List[Dict[str, Any]]] = {
            metric_type: [] for metric_type in MetricType
        }
        self.broadcast_interval = 1.0  # seconds
        self.history_retention = timedelta(hours=24)
        
        self.logger = get_logger(__name__)
        self._broadcast_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        # Quantum state tracking
        self._quantum_state_history: List[Dict[str, Any]] = []
        self._performance_snapshots: List[Dict[str, Any]] = []
        self._system_alerts: List[Dict[str, Any]] = []
        
        # Analytics
        self._analytics_engine = QuantumAnalyticsEngine()
    
    async def start(self):
        """Start the dashboard manager"""
        self.logger.info("Starting Quantum Dashboard Manager")
        
        # Start background tasks
        self._broadcast_task = asyncio.create_task(self._broadcast_metrics_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        self.logger.info("Quantum Dashboard Manager started")
    
    async def stop(self):
        """Stop the dashboard manager"""
        self.logger.info("Stopping Quantum Dashboard Manager")
        
        # Cancel background tasks
        if self._broadcast_task:
            self._broadcast_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        # Disconnect all clients
        disconnect_tasks = []
        for client in list(self.active_clients.values()):
            disconnect_tasks.append(self.disconnect_client(client.client_id))
        
        if disconnect_tasks:
            await asyncio.gather(*disconnect_tasks, return_exceptions=True)
        
        self.logger.info("Quantum Dashboard Manager stopped")
    
    async def connect_client(self, websocket: WebSocket) -> str:
        """Connect a new dashboard client"""
        await websocket.accept()
        
        client_id = str(uuid.uuid4())
        client = DashboardClient(
            client_id=client_id,
            websocket=websocket,
            subscribed_metrics=set(MetricType)  # Subscribe to all by default
        )
        
        self.active_clients[client_id] = client
        self.logger.info(f"Dashboard client {client_id} connected")
        
        # Send initial data
        await self._send_initial_data(client)
        
        return client_id
    
    async def disconnect_client(self, client_id: str):
        """Disconnect a dashboard client"""
        if client_id in self.active_clients:
            client = self.active_clients[client_id]
            try:
                await client.websocket.close()
            except:
                pass  # Already closed
            
            del self.active_clients[client_id]
            self.logger.info(f"Dashboard client {client_id} disconnected")
    
    async def handle_client_message(self, client_id: str, message_data: Dict[str, Any]):
        """Handle incoming message from dashboard client"""
        try:
            message_type = message_data.get("message_type")
            
            if message_type == "subscribe_metrics":
                metrics = message_data.get("metrics", [])
                await self._handle_subscribe_metrics(client_id, metrics)
            
            elif message_type == "unsubscribe_metrics":
                metrics = message_data.get("metrics", [])
                await self._handle_unsubscribe_metrics(client_id, metrics)
            
            elif message_type == "request_historical_data":
                metric_type = message_data.get("metric_type")
                timeframe = message_data.get("timeframe", "1h")
                await self._handle_historical_data_request(client_id, metric_type, timeframe)
            
            elif message_type == "execute_command":
                command = message_data.get("command")
                await self._handle_command_execution(client_id, command)
            
            elif message_type == "heartbeat":
                await self._handle_heartbeat(client_id)
            
            else:
                self.logger.warning(f"Unknown message type from client {client_id}: {message_type}")
        
        except Exception as e:
            self.logger.error(f"Error handling client message: {e}")
            await self._send_error_to_client(client_id, str(e))
    
    async def _send_initial_data(self, client: DashboardClient):
        """Send initial dashboard data to newly connected client"""
        # System overview
        system_overview = await self._collect_system_overview()
        await self._send_to_client(client, DashboardMessage(
            message_type="system_overview",
            data=system_overview,
            client_id=client.client_id
        ))
        
        # Current quantum state
        quantum_state = await self._collect_quantum_state_data()
        await self._send_to_client(client, DashboardMessage(
            message_type="quantum_state",
            data=quantum_state,
            client_id=client.client_id
        ))
        
        # Performance snapshot
        performance_data = await self._collect_performance_data()
        await self._send_to_client(client, DashboardMessage(
            message_type="performance_snapshot",
            data=performance_data,
            client_id=client.client_id
        ))
    
    async def _broadcast_metrics_loop(self):
        """Main loop for broadcasting metrics to connected clients"""
        while True:
            try:
                await asyncio.sleep(self.broadcast_interval)
                
                if not self.active_clients:
                    continue
                
                # Collect all metric types
                metrics_to_broadcast = {
                    MetricType.SYSTEM_OVERVIEW: await self._collect_system_overview(),
                    MetricType.TASK_EXECUTION: await self._collect_task_execution_data(),
                    MetricType.QUANTUM_COHERENCE: await self._collect_quantum_coherence_data(),
                    MetricType.ENTANGLEMENT_NETWORK: await self._collect_entanglement_data(),
                    MetricType.RESOURCE_UTILIZATION: await self._collect_resource_utilization(),
                    MetricType.PERFORMANCE_ANALYTICS: await self._collect_performance_analytics()
                }
                
                # Broadcast to subscribed clients
                broadcast_tasks = []
                for client in list(self.active_clients.values()):
                    for metric_type, data in metrics_to_broadcast.items():
                        if metric_type in client.subscribed_metrics:
                            message = DashboardMessage(
                                message_type=metric_type.value,
                                data=data,
                                client_id=client.client_id
                            )
                            broadcast_tasks.append(self._send_to_client(client, message))
                
                if broadcast_tasks:
                    await asyncio.gather(*broadcast_tasks, return_exceptions=True)
                
                # Store metrics in history
                await self._store_metrics_history(metrics_to_broadcast)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in metrics broadcast loop: {e}")
                await asyncio.sleep(5)
    
    async def _collect_system_overview(self) -> Dict[str, Any]:
        """Collect system overview metrics"""
        executor = get_quantum_executor()
        executor_metrics = executor.get_system_metrics()
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "total_tasks": len(scheduler.tasks),
            "active_executions": executor_metrics["active_executions"],
            "queue_size": executor_metrics["queue_size"],
            "successful_executions": executor_metrics["executor_metrics"]["successful_executions"],
            "failed_executions": executor_metrics["executor_metrics"]["failed_executions"],
            "average_execution_time": executor_metrics["executor_metrics"]["average_execution_time"],
            "active_entanglements": len(entanglement_manager.entanglement_bonds),
            "system_health": "healthy",  # TODO: Implement health check
            "uptime": "unknown"  # TODO: Calculate actual uptime
        }
    
    async def _collect_task_execution_data(self) -> Dict[str, Any]:
        """Collect task execution metrics"""
        tasks = list(scheduler.tasks.values())
        
        state_distribution = {}
        priority_distribution = {}
        
        for task in tasks:
            state = task.state.value
            priority = task.priority.name
            
            state_distribution[state] = state_distribution.get(state, 0) + 1
            priority_distribution[priority] = priority_distribution.get(priority, 0) + 1
        
        # Recent execution timeline
        executor = get_quantum_executor()
        recent_executions = []
        for context in executor.execution_history[-20:]:  # Last 20
            recent_executions.append({
                "task_id": context.task.task_id if context.task else "unknown",
                "task_title": context.task.title if context.task else "unknown",
                "start_time": context.start_time.isoformat() if context.start_time else None,
                "completion_time": context.completion_time.isoformat() if context.completion_time else None,
                "duration": context.execution_duration.total_seconds() if context.execution_duration else 0,
                "executor_type": context.executor_type.value,
                "progress": context.progress,
                "status": context.task.state.value if context.task else "unknown"
            })
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "state_distribution": state_distribution,
            "priority_distribution": priority_distribution,
            "recent_executions": recent_executions,
            "total_tasks": len(tasks),
            "completion_rate": state_distribution.get("completed", 0) / max(1, len(tasks))
        }
    
    async def _collect_quantum_coherence_data(self) -> Dict[str, Any]:
        """Collect quantum coherence metrics"""
        tasks = list(scheduler.tasks.values())
        
        coherence_values = [task.quantum_coherence for task in tasks]
        coherence_by_priority = {}
        coherence_over_time = []
        
        for task in tasks:
            priority = task.priority.name
            if priority not in coherence_by_priority:
                coherence_by_priority[priority] = []
            coherence_by_priority[priority].append(task.quantum_coherence)
        
        # Calculate coherence statistics
        coherence_stats = {
            "mean": np.mean(coherence_values) if coherence_values else 0,
            "std": np.std(coherence_values) if coherence_values else 0,
            "min": np.min(coherence_values) if coherence_values else 0,
            "max": np.max(coherence_values) if coherence_values else 0,
            "median": np.median(coherence_values) if coherence_values else 0
        }
        
        # Coherence preservation rate
        high_coherence_tasks = len([c for c in coherence_values if c > 0.8])
        coherence_preservation_rate = high_coherence_tasks / max(1, len(coherence_values))
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "coherence_distribution": coherence_values,
            "coherence_by_priority": {
                k: {"mean": np.mean(v), "count": len(v)} 
                for k, v in coherence_by_priority.items()
            },
            "coherence_statistics": coherence_stats,
            "preservation_rate": coherence_preservation_rate,
            "critical_tasks": len([c for c in coherence_values if c < 0.3]),
            "stable_tasks": len([c for c in coherence_values if c > 0.7])
        }
    
    async def _collect_entanglement_data(self) -> Dict[str, Any]:
        """Collect quantum entanglement network data"""
        entanglement_stats = entanglement_manager.get_entanglement_statistics()
        
        # Create network graph data
        nodes = []
        edges = []
        
        # Add task nodes
        for task_id, task in scheduler.tasks.items():
            nodes.append({
                "id": task_id,
                "label": task.title[:20] + "..." if len(task.title) > 20 else task.title,
                "coherence": task.quantum_coherence,
                "priority": task.priority.name,
                "state": task.state.value,
                "entanglement_count": len(task.entangled_tasks)
            })
        
        # Add entanglement edges
        for bond_id, bond in entanglement_manager.entanglement_bonds.items():
            if len(bond.entangled_tasks) >= 2:
                for i, task1 in enumerate(bond.entangled_tasks):
                    for task2 in bond.entangled_tasks[i+1:]:
                        edges.append({
                            "source": task1.task_id,
                            "target": task2.task_id,
                            "bond_id": bond_id,
                            "strength": bond.strength,
                            "type": bond.entanglement_type.value,
                            "coherence": bond.quantum_coherence
                        })
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "network": {
                "nodes": nodes,
                "edges": edges
            },
            "statistics": entanglement_stats,
            "total_bonds": len(entanglement_manager.entanglement_bonds),
            "network_density": len(edges) / max(1, len(nodes) * (len(nodes) - 1) / 2),
            "average_node_degree": 2 * len(edges) / max(1, len(nodes))
        }
    
    async def _collect_resource_utilization(self) -> Dict[str, Any]:
        """Collect resource utilization metrics"""
        executor = get_quantum_executor()
        executor_metrics = executor.get_system_metrics()
        
        resource_data = executor_metrics["resource_pools"]
        
        # Calculate utilization trends
        utilization_timeline = []
        for resource_type, resource_info in resource_data.items():
            utilization_timeline.append({
                "resource_type": resource_type,
                "utilization": resource_info["utilization"],
                "current_usage": resource_info["current_usage"],
                "allocated": resource_info["allocated"],
                "timestamp": datetime.utcnow().isoformat()
            })
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "resource_pools": resource_data,
            "utilization_timeline": utilization_timeline,
            "overall_utilization": np.mean([r["utilization"] for r in resource_data.values()]),
            "bottlenecks": [
                resource_type for resource_type, resource_info in resource_data.items()
                if resource_info["utilization"] > 0.8
            ]
        }
    
    async def _collect_performance_analytics(self) -> Dict[str, Any]:
        """Collect performance analytics data"""
        # Use the analytics engine
        analytics = await self._analytics_engine.generate_performance_analytics(
            tasks=list(scheduler.tasks.values()),
            execution_history=get_quantum_executor().execution_history
        )
        
        return {
            "timestamp": datetime.utcnow().isoformat(),
            "analytics": analytics
        }
    
    async def _send_to_client(self, client: DashboardClient, message: DashboardMessage):
        """Send message to specific client"""
        try:
            await client.websocket.send_text(message.to_json())
        except Exception as e:
            self.logger.error(f"Error sending to client {client.client_id}: {e}")
            await self.disconnect_client(client.client_id)
    
    async def _send_error_to_client(self, client_id: str, error_message: str):
        """Send error message to client"""
        if client_id in self.active_clients:
            client = self.active_clients[client_id]
            error_msg = DashboardMessage(
                message_type="error",
                data={"error": error_message},
                client_id=client_id
            )
            await self._send_to_client(client, error_msg)
    
    async def _store_metrics_history(self, metrics: Dict[MetricType, Dict[str, Any]]):
        """Store metrics in history for analytics"""
        for metric_type, data in metrics.items():
            self.metric_history[metric_type].append(data)
            
            # Maintain history size
            if len(self.metric_history[metric_type]) > 1000:
                self.metric_history[metric_type] = self.metric_history[metric_type][-500:]
    
    async def _cleanup_loop(self):
        """Background cleanup task"""
        while True:
            try:
                await asyncio.sleep(60)  # Run every minute
                
                # Clean up old metric history
                cutoff_time = datetime.utcnow() - self.history_retention
                for metric_type in self.metric_history:
                    self.metric_history[metric_type] = [
                        entry for entry in self.metric_history[metric_type]
                        if datetime.fromisoformat(entry["timestamp"]) > cutoff_time
                    ]
                
                # Check for stale client connections
                stale_clients = []
                for client_id, client in self.active_clients.items():
                    if datetime.utcnow() - client.last_heartbeat > timedelta(minutes=5):
                        stale_clients.append(client_id)
                
                for client_id in stale_clients:
                    await self.disconnect_client(client_id)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
                await asyncio.sleep(60)
    
    async def _handle_subscribe_metrics(self, client_id: str, metrics: List[str]):
        """Handle metric subscription request"""
        if client_id in self.active_clients:
            client = self.active_clients[client_id]
            for metric_name in metrics:
                try:
                    metric_type = MetricType(metric_name)
                    client.subscribed_metrics.add(metric_type)
                except ValueError:
                    self.logger.warning(f"Invalid metric type: {metric_name}")
    
    async def _handle_unsubscribe_metrics(self, client_id: str, metrics: List[str]):
        """Handle metric unsubscription request"""
        if client_id in self.active_clients:
            client = self.active_clients[client_id]
            for metric_name in metrics:
                try:
                    metric_type = MetricType(metric_name)
                    client.subscribed_metrics.discard(metric_type)
                except ValueError:
                    pass
    
    async def _handle_historical_data_request(self, client_id: str, metric_type: str, timeframe: str):
        """Handle request for historical data"""
        try:
            metric_enum = MetricType(metric_type)
            history = self.metric_history[metric_enum]
            
            # Filter by timeframe
            if timeframe == "1h":
                cutoff = datetime.utcnow() - timedelta(hours=1)
            elif timeframe == "6h":
                cutoff = datetime.utcnow() - timedelta(hours=6)
            elif timeframe == "24h":
                cutoff = datetime.utcnow() - timedelta(hours=24)
            else:
                cutoff = datetime.utcnow() - timedelta(hours=1)
            
            filtered_history = [
                entry for entry in history
                if datetime.fromisoformat(entry["timestamp"]) > cutoff
            ]
            
            if client_id in self.active_clients:
                client = self.active_clients[client_id]
                message = DashboardMessage(
                    message_type="historical_data",
                    data={
                        "metric_type": metric_type,
                        "timeframe": timeframe,
                        "data": filtered_history
                    },
                    client_id=client_id
                )
                await self._send_to_client(client, message)
        
        except Exception as e:
            await self._send_error_to_client(client_id, f"Error retrieving historical data: {e}")
    
    async def _handle_command_execution(self, client_id: str, command: Dict[str, Any]):
        """Handle command execution request"""
        # TODO: Implement command execution (with proper security)
        command_type = command.get("type")
        
        if command_type == "optimize_schedule":
            await scheduler.optimize_schedule()
            
        elif command_type == "clear_cache":
            # Clear caches if available
            pass
        
        # Send confirmation
        if client_id in self.active_clients:
            client = self.active_clients[client_id]
            message = DashboardMessage(
                message_type="command_result",
                data={"command": command_type, "status": "executed"},
                client_id=client_id
            )
            await self._send_to_client(client, message)
    
    async def _handle_heartbeat(self, client_id: str):
        """Handle client heartbeat"""
        if client_id in self.active_clients:
            self.active_clients[client_id].last_heartbeat = datetime.utcnow()


class QuantumAnalyticsEngine:
    """Advanced analytics engine for quantum task performance"""
    
    async def generate_performance_analytics(self, tasks: List, execution_history: List) -> Dict[str, Any]:
        """Generate comprehensive performance analytics"""
        
        # Task completion analytics
        completion_analytics = self._analyze_task_completion(tasks)
        
        # Execution performance analytics
        execution_analytics = self._analyze_execution_performance(execution_history)
        
        # Quantum coherence analytics
        coherence_analytics = self._analyze_coherence_patterns(tasks)
        
        # Predictive analytics
        predictions = self._generate_predictions(tasks, execution_history)
        
        return {
            "completion_analytics": completion_analytics,
            "execution_analytics": execution_analytics,
            "coherence_analytics": coherence_analytics,
            "predictions": predictions,
            "insights": self._generate_insights(tasks, execution_history)
        }
    
    def _analyze_task_completion(self, tasks: List) -> Dict[str, Any]:
        """Analyze task completion patterns"""
        if not tasks:
            return {}
        
        completion_times = []
        priority_performance = {}
        
        for task in tasks:
            if hasattr(task, 'completion_time') and task.completion_time:
                completion_times.append(task.completion_time)
            
            priority = task.priority.name
            if priority not in priority_performance:
                priority_performance[priority] = {"total": 0, "completed": 0}
            
            priority_performance[priority]["total"] += 1
            if task.state.value == "completed":
                priority_performance[priority]["completed"] += 1
        
        return {
            "total_completion_times": len(completion_times),
            "priority_performance": priority_performance,
            "overall_completion_rate": len([t for t in tasks if t.state.value == "completed"]) / len(tasks)
        }
    
    def _analyze_execution_performance(self, execution_history: List) -> Dict[str, Any]:
        """Analyze execution performance patterns"""
        if not execution_history:
            return {}
        
        execution_times = []
        executor_performance = {}
        
        for context in execution_history:
            if context.execution_duration:
                execution_times.append(context.execution_duration.total_seconds())
            
            executor_type = context.executor_type.value
            if executor_type not in executor_performance:
                executor_performance[executor_type] = []
            
            if context.execution_duration:
                executor_performance[executor_type].append(context.execution_duration.total_seconds())
        
        return {
            "average_execution_time": np.mean(execution_times) if execution_times else 0,
            "execution_time_std": np.std(execution_times) if execution_times else 0,
            "executor_performance": {
                k: {"mean": np.mean(v), "count": len(v)}
                for k, v in executor_performance.items()
            }
        }
    
    def _analyze_coherence_patterns(self, tasks: List) -> Dict[str, Any]:
        """Analyze quantum coherence patterns"""
        if not tasks:
            return {}
        
        coherence_values = [task.quantum_coherence for task in tasks]
        
        return {
            "mean_coherence": np.mean(coherence_values),
            "coherence_std": np.std(coherence_values),
            "coherence_trend": "stable",  # TODO: Calculate actual trend
            "critical_coherence_tasks": len([c for c in coherence_values if c < 0.3])
        }
    
    def _generate_predictions(self, tasks: List, execution_history: List) -> Dict[str, Any]:
        """Generate predictive analytics"""
        return {
            "predicted_completion_time": "2 hours",  # TODO: Implement ML prediction
            "resource_bottleneck_prediction": "cpu_intensive",
            "coherence_degradation_forecast": "stable",
            "optimal_execution_strategy": "hybrid_adaptive"
        }
    
    def _generate_insights(self, tasks: List, execution_history: List) -> List[Dict[str, Any]]:
        """Generate actionable insights"""
        insights = []
        
        if not tasks:
            return insights
        
        # Check for coherence issues
        low_coherence_tasks = len([t for t in tasks if t.quantum_coherence < 0.5])
        if low_coherence_tasks > len(tasks) * 0.3:
            insights.append({
                "type": "warning",
                "title": "High Coherence Degradation",
                "description": f"{low_coherence_tasks} tasks have low coherence levels",
                "recommendation": "Consider implementing coherence preservation measures"
            })
        
        # Check execution performance
        if execution_history and len(execution_history) > 5:
            recent_failures = len([e for e in execution_history[-10:] if e.task and e.task.state.value == "failed"])
            if recent_failures > 3:
                insights.append({
                    "type": "critical",
                    "title": "High Failure Rate",
                    "description": f"{recent_failures} recent task failures detected",
                    "recommendation": "Investigate system stability and resource allocation"
                })
        
        return insights


# Global dashboard manager
_dashboard_manager: Optional[QuantumDashboardManager] = None


def get_dashboard_manager() -> QuantumDashboardManager:
    """Get global dashboard manager instance"""
    global _dashboard_manager
    if _dashboard_manager is None:
        _dashboard_manager = QuantumDashboardManager()
    return _dashboard_manager


# WebSocket endpoint for dashboard
@app.websocket("/ws/dashboard")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time dashboard updates"""
    dashboard_manager = get_dashboard_manager()
    
    try:
        client_id = await dashboard_manager.connect_client(websocket)
        
        while True:
            try:
                # Receive message from client
                raw_message = await websocket.receive_text()
                message_data = json.loads(raw_message)
                
                # Handle message
                await dashboard_manager.handle_client_message(client_id, message_data)
                
            except WebSocketDisconnect:
                break
            except json.JSONDecodeError:
                await dashboard_manager._send_error_to_client(client_id, "Invalid JSON format")
            except Exception as e:
                await dashboard_manager._send_error_to_client(client_id, str(e))
    
    except Exception as e:
        get_logger(__name__).error(f"WebSocket error: {e}")
    
    finally:
        await dashboard_manager.disconnect_client(client_id)


# HTML dashboard page
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_page():
    """Serve the dashboard HTML page"""
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Quantum Task Planner Dashboard</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body { 
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                margin: 0; padding: 20px; background: #1a1a1a; color: #ffffff;
            }
            .dashboard { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
            .panel { 
                background: #2d2d2d; border-radius: 8px; padding: 20px; 
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            .metric { display: flex; justify-content: space-between; margin: 10px 0; }
            .status-indicator { 
                width: 12px; height: 12px; border-radius: 50%; 
                display: inline-block; margin-right: 8px;
            }
            .status-healthy { background: #00ff00; }
            .status-warning { background: #ffaa00; }
            .status-critical { background: #ff0000; }
            .chart-container { height: 300px; margin: 20px 0; }
            h2 { color: #00ffff; margin-top: 0; }
            .connection-status { 
                position: fixed; top: 10px; right: 10px; 
                padding: 8px 16px; border-radius: 4px; font-size: 12px;
            }
            .connected { background: #00aa00; }
            .disconnected { background: #aa0000; }
        </style>
    </head>
    <body>
        <div id="connection-status" class="connection-status disconnected">Connecting...</div>
        
        <h1>🌌 Quantum Task Planner Dashboard</h1>
        
        <div class="dashboard">
            <div class="panel">
                <h2>System Overview</h2>
                <div id="system-metrics"></div>
            </div>
            
            <div class="panel">
                <h2>Task Execution</h2>
                <div id="task-metrics"></div>
            </div>
            
            <div class="panel">
                <h2>Quantum Coherence</h2>
                <div id="coherence-chart" class="chart-container"></div>
            </div>
            
            <div class="panel">
                <h2>Resource Utilization</h2>
                <div id="resource-chart" class="chart-container"></div>
            </div>
            
            <div class="panel">
                <h2>Entanglement Network</h2>
                <div id="entanglement-stats"></div>
            </div>
            
            <div class="panel">
                <h2>Performance Analytics</h2>
                <div id="performance-metrics"></div>
            </div>
        </div>

        <script>
            class QuantumDashboard {
                constructor() {
                    this.ws = null;
                    this.reconnectInterval = 5000;
                    this.connect();
                }
                
                connect() {
                    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
                    const wsUrl = `${protocol}//${window.location.host}/ws/dashboard`;
                    
                    this.ws = new WebSocket(wsUrl);
                    
                    this.ws.onopen = () => {
                        console.log('Dashboard connected');
                        document.getElementById('connection-status').textContent = 'Connected';
                        document.getElementById('connection-status').className = 'connection-status connected';
                        
                        // Send heartbeat every 30 seconds
                        this.heartbeatInterval = setInterval(() => {
                            this.send({message_type: 'heartbeat'});
                        }, 30000);
                    };
                    
                    this.ws.onmessage = (event) => {
                        const message = JSON.parse(event.data);
                        this.handleMessage(message);
                    };
                    
                    this.ws.onclose = () => {
                        console.log('Dashboard disconnected');
                        document.getElementById('connection-status').textContent = 'Disconnected';
                        document.getElementById('connection-status').className = 'connection-status disconnected';
                        
                        clearInterval(this.heartbeatInterval);
                        setTimeout(() => this.connect(), this.reconnectInterval);
                    };
                    
                    this.ws.onerror = (error) => {
                        console.error('WebSocket error:', error);
                    };
                }
                
                send(message) {
                    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
                        this.ws.send(JSON.stringify(message));
                    }
                }
                
                handleMessage(message) {
                    switch (message.message_type) {
                        case 'system_overview':
                            this.updateSystemOverview(message.data);
                            break;
                        case 'task_execution':
                            this.updateTaskExecution(message.data);
                            break;
                        case 'quantum_coherence':
                            this.updateQuantumCoherence(message.data);
                            break;
                        case 'resource_utilization':
                            this.updateResourceUtilization(message.data);
                            break;
                        case 'entanglement_network':
                            this.updateEntanglementNetwork(message.data);
                            break;
                        case 'performance_analytics':
                            this.updatePerformanceAnalytics(message.data);
                            break;
                    }
                }
                
                updateSystemOverview(data) {
                    const html = `
                        <div class="metric">
                            <span>Total Tasks:</span><span>${data.total_tasks}</span>
                        </div>
                        <div class="metric">
                            <span>Active Executions:</span><span>${data.active_executions}</span>
                        </div>
                        <div class="metric">
                            <span>Queue Size:</span><span>${data.queue_size}</span>
                        </div>
                        <div class="metric">
                            <span>Success Rate:</span>
                            <span>${(data.successful_executions / Math.max(1, data.successful_executions + data.failed_executions) * 100).toFixed(1)}%</span>
                        </div>
                        <div class="metric">
                            <span>System Health:</span>
                            <span><span class="status-indicator status-healthy"></span>${data.system_health}</span>
                        </div>
                    `;
                    document.getElementById('system-metrics').innerHTML = html;
                }
                
                updateTaskExecution(data) {
                    const html = `
                        <div class="metric">
                            <span>Completion Rate:</span><span>${(data.completion_rate * 100).toFixed(1)}%</span>
                        </div>
                        <div class="metric">
                            <span>State Distribution:</span>
                            <span>${Object.entries(data.state_distribution).map(([k,v]) => `${k}: ${v}`).join(', ')}</span>
                        </div>
                        <div class="metric">
                            <span>Recent Executions:</span><span>${data.recent_executions.length}</span>
                        </div>
                    `;
                    document.getElementById('task-metrics').innerHTML = html;
                }
                
                updateQuantumCoherence(data) {
                    const trace = {
                        type: 'histogram',
                        x: data.coherence_distribution,
                        marker: { color: '#00ffff' },
                        name: 'Coherence Distribution'
                    };
                    
                    const layout = {
                        title: 'Quantum Coherence Distribution',
                        paper_bgcolor: '#2d2d2d',
                        plot_bgcolor: '#2d2d2d',
                        font: { color: '#ffffff' },
                        xaxis: { title: 'Coherence Level' },
                        yaxis: { title: 'Task Count' }
                    };
                    
                    Plotly.newPlot('coherence-chart', [trace], layout);
                }
                
                updateResourceUtilization(data) {
                    const resourceTypes = Object.keys(data.resource_pools);
                    const utilizationValues = resourceTypes.map(rt => data.resource_pools[rt].utilization * 100);
                    
                    const trace = {
                        type: 'bar',
                        x: resourceTypes,
                        y: utilizationValues,
                        marker: { color: '#ffaa00' },
                        name: 'Resource Utilization'
                    };
                    
                    const layout = {
                        title: 'Resource Utilization %',
                        paper_bgcolor: '#2d2d2d',
                        plot_bgcolor: '#2d2d2d',
                        font: { color: '#ffffff' },
                        yaxis: { title: 'Utilization %', range: [0, 100] }
                    };
                    
                    Plotly.newPlot('resource-chart', [trace], layout);
                }
                
                updateEntanglementNetwork(data) {
                    const html = `
                        <div class="metric">
                            <span>Total Bonds:</span><span>${data.total_bonds}</span>
                        </div>
                        <div class="metric">
                            <span>Network Density:</span><span>${data.network_density.toFixed(3)}</span>
                        </div>
                        <div class="metric">
                            <span>Avg Node Degree:</span><span>${data.average_node_degree.toFixed(1)}</span>
                        </div>
                        <div class="metric">
                            <span>Network Nodes:</span><span>${data.network.nodes.length}</span>
                        </div>
                    `;
                    document.getElementById('entanglement-stats').innerHTML = html;
                }
                
                updatePerformanceAnalytics(data) {
                    const analytics = data.analytics || {};
                    const insights = analytics.insights || [];
                    
                    let html = '<h3>Insights:</h3>';
                    insights.forEach(insight => {
                        const statusClass = insight.type === 'critical' ? 'status-critical' : 
                                          insight.type === 'warning' ? 'status-warning' : 'status-healthy';
                        html += `
                            <div class="metric">
                                <span><span class="status-indicator ${statusClass}"></span>${insight.title}</span>
                            </div>
                        `;
                    });
                    
                    if (insights.length === 0) {
                        html += '<div class="metric"><span>No critical insights</span></div>';
                    }
                    
                    document.getElementById('performance-metrics').innerHTML = html;
                }
            }
            
            // Initialize dashboard
            const dashboard = new QuantumDashboard();
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)