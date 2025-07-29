#!/usr/bin/env python3
"""
Performance benchmarking script for the Agentic Startup Studio Boilerplate.

This script provides comprehensive performance testing and benchmarking capabilities
for all components of the system including API endpoints, database operations,
AI agent performance, and frontend metrics.
"""

import asyncio
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any
import sys
import os

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

import aiohttp
import psutil
import pytest
from dataclasses import dataclass, asdict
from statistics import mean, median, stdev


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    test_name: str
    duration_ms: float
    memory_usage_mb: float
    cpu_percent: float
    status: str
    timestamp: datetime
    metadata: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result


class PerformanceBenchmark:
    """Main benchmarking class."""
    
    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.results: List[BenchmarkResult] = []
        self.session = None

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    def measure_resources(self):
        """Measure current resource usage."""
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        cpu_percent = process.cpu_percent()
        return memory_mb, cpu_percent

    async def benchmark_api_endpoint(self, endpoint: str, method: str = "GET", 
                                   payload: Dict = None, iterations: int = 100):
        """Benchmark a specific API endpoint."""
        print(f"Benchmarking {method} {endpoint} ({iterations} iterations)...")
        
        durations = []
        memory_usage = []
        cpu_usage = []
        
        for i in range(iterations):
            start_time = time.perf_counter()
            start_memory, start_cpu = self.measure_resources()
            
            try:
                async with self.session.request(
                    method, 
                    f"{self.base_url}{endpoint}",
                    json=payload
                ) as response:
                    await response.text()
                    status = "success" if response.status < 400 else "error"
                    
            except Exception as e:
                status = f"error: {str(e)}"
            
            end_time = time.perf_counter()
            end_memory, end_cpu = self.measure_resources()
            
            duration_ms = (end_time - start_time) * 1000
            durations.append(duration_ms)
            memory_usage.append(end_memory)
            cpu_usage.append(end_cpu)
            
            # Brief pause to allow system to settle
            await asyncio.sleep(0.01)
        
        # Calculate statistics
        avg_duration = mean(durations)
        median_duration = median(durations)
        p95_duration = sorted(durations)[int(0.95 * len(durations))]
        p99_duration = sorted(durations)[int(0.99 * len(durations))]
        
        result = BenchmarkResult(
            test_name=f"{method} {endpoint}",
            duration_ms=avg_duration,
            memory_usage_mb=mean(memory_usage),
            cpu_percent=mean(cpu_usage),
            status=status,
            timestamp=datetime.now(),
            metadata={
                "iterations": iterations,
                "median_duration_ms": median_duration,
                "p95_duration_ms": p95_duration,
                "p99_duration_ms": p99_duration,
                "min_duration_ms": min(durations),
                "max_duration_ms": max(durations),
                "std_dev_ms": stdev(durations) if len(durations) > 1 else 0
            }
        )
        
        self.results.append(result)
        print(f"  Avg: {avg_duration:.2f}ms, P95: {p95_duration:.2f}ms, P99: {p99_duration:.2f}ms")
        return result

    async def benchmark_ai_agent(self, agent_endpoint: str = "/api/v1/agents/execute",
                                 complexity: str = "medium", iterations: int = 10):
        """Benchmark AI agent performance."""
        print(f"Benchmarking AI agent ({complexity} complexity, {iterations} iterations)...")
        
        test_prompts = {
            "simple": "What is 2+2?",
            "medium": "Analyze the current market trends for AI startups and provide 3 key insights.",
            "complex": """
            Create a comprehensive business plan for an AI-powered fintech startup 
            that helps small businesses optimize their cash flow. Include market analysis, 
            competitive landscape, revenue model, and 3-year financial projections.
            """
        }
        
        payload = {
            "prompt": test_prompts.get(complexity, test_prompts["medium"]),
            "agent_type": "business_analyst",
            "max_tokens": 2000
        }
        
        return await self.benchmark_api_endpoint(
            agent_endpoint, 
            method="POST", 
            payload=payload, 
            iterations=iterations
        )

    async def benchmark_database_operations(self, iterations: int = 50):
        """Benchmark database operations."""
        print(f"Benchmarking database operations ({iterations} iterations)...")
        
        # Test different database operations
        operations = [
            ("/api/v1/users", "GET"),  # Read operation
            ("/api/v1/projects", "GET"),  # List operation
            ("/api/v1/agents", "POST", {"name": "test_agent", "type": "analyzer"}),  # Create
        ]
        
        for endpoint, method, *payload in operations:
            data = payload[0] if payload else None
            await self.benchmark_api_endpoint(endpoint, method, data, iterations)

    async def run_comprehensive_benchmark(self):
        """Run comprehensive performance benchmark."""
        print("Starting comprehensive performance benchmark...")
        print("=" * 60)
        
        # Health check first
        try:
            async with self.session.get(f"{self.base_url}/health") as response:
                if response.status != 200:
                    print(f"❌ Health check failed: {response.status}")
                    return
                print("✅ Health check passed")
        except Exception as e:
            print(f"❌ Cannot connect to server: {e}")
            return
        
        # API endpoint benchmarks
        print("\n📡 API Endpoint Benchmarks")
        print("-" * 30)
        await self.benchmark_api_endpoint("/", "GET", iterations=100)
        await self.benchmark_api_endpoint("/health", "GET", iterations=100)
        await self.benchmark_api_endpoint("/api/v1/agents", "GET", iterations=50)
        
        # Database benchmarks
        print("\n🗄️ Database Benchmarks")
        print("-" * 30)
        await self.benchmark_database_operations(iterations=25)
        
        # AI agent benchmarks
        print("\n🤖 AI Agent Benchmarks") 
        print("-" * 30)
        await self.benchmark_ai_agent(complexity="simple", iterations=5)
        await self.benchmark_ai_agent(complexity="medium", iterations=3)
        
        print("\n✅ Benchmark completed!")
        return self.results

    def generate_report(self, output_file: str = None):
        """Generate performance report."""
        if not self.results:
            print("No benchmark results to report")
            return
        
        # Calculate summary statistics
        total_tests = len(self.results)
        avg_duration = mean([r.duration_ms for r in self.results])
        avg_memory = mean([r.memory_usage_mb for r in self.results])
        avg_cpu = mean([r.cpu_percent for r in self.results])
        
        # Find slowest tests
        slowest_tests = sorted(self.results, key=lambda x: x.duration_ms, reverse=True)[:5]
        
        report = {
            "summary": {
                "total_tests": total_tests,
                "avg_duration_ms": round(avg_duration, 2),
                "avg_memory_usage_mb": round(avg_memory, 2),
                "avg_cpu_percent": round(avg_cpu, 2),
                "timestamp": datetime.now().isoformat()
            },
            "slowest_tests": [r.to_dict() for r in slowest_tests],
            "all_results": [r.to_dict() for r in self.results],
            "performance_targets": {
                "api_response_time_ms": 200,
                "ai_agent_response_time_ms": 30000,
                "memory_usage_mb": 512,
                "cpu_utilization_percent": 70
            },
            "analysis": self._analyze_results()
        }
        
        # Output report
        if output_file:
            Path(output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(output_file, 'w') as f:
                json.dump(report, f, indent=2)
            print(f"📊 Report saved to {output_file}")
        
        # Print summary to console
        self._print_summary(report)
        
        return report

    def _analyze_results(self) -> Dict[str, Any]:
        """Analyze benchmark results and provide insights."""
        analysis = {
            "performance_status": "good",
            "bottlenecks": [],
            "recommendations": []
        }
        
        # Check against performance targets
        api_results = [r for r in self.results if not r.test_name.startswith("POST /api/v1/agents/execute")]
        
        if api_results:
            avg_api_time = mean([r.duration_ms for r in api_results])
            if avg_api_time > 200:
                analysis["performance_status"] = "needs_improvement"
                analysis["bottlenecks"].append("API response times exceed 200ms target")
                analysis["recommendations"].append("Optimize database queries and implement caching")
        
        # Check AI agent performance
        ai_results = [r for r in self.results if r.test_name.startswith("POST /api/v1/agents/execute")]
        if ai_results:
            avg_ai_time = mean([r.duration_ms for r in ai_results])
            if avg_ai_time > 30000:
                analysis["bottlenecks"].append("AI agent responses exceed 30s target")
                analysis["recommendations"].append("Implement response streaming and prompt optimization")
        
        # Check memory usage
        avg_memory = mean([r.memory_usage_mb for r in self.results])
        if avg_memory > 512:
            analysis["bottlenecks"].append("Memory usage exceeds 512MB target")
            analysis["recommendations"].append("Profile memory usage and optimize data structures")
        
        return analysis

    def _print_summary(self, report: Dict[str, Any]):
        """Print benchmark summary to console."""
        print("\n" + "=" * 60)
        print("📊 PERFORMANCE BENCHMARK SUMMARY")
        print("=" * 60)
        
        summary = report["summary"]
        print(f"Total Tests: {summary['total_tests']}")
        print(f"Average Response Time: {summary['avg_duration_ms']}ms")
        print(f"Average Memory Usage: {summary['avg_memory_usage_mb']:.1f}MB")
        print(f"Average CPU Usage: {summary['avg_cpu_percent']:.1f}%")
        
        print(f"\nPerformance Status: {report['analysis']['performance_status'].upper()}")
        
        if report['analysis']['bottlenecks']:
            print("\n🔍 Identified Bottlenecks:")
            for bottleneck in report['analysis']['bottlenecks']:
                print(f"  • {bottleneck}")
        
        if report['analysis']['recommendations']:
            print("\n💡 Recommendations:")
            for rec in report['analysis']['recommendations']:
                print(f"  • {rec}")
        
        print("\n🐌 Slowest Tests:")
        for test in report['slowest_tests']:
            print(f"  • {test['test_name']}: {test['duration_ms']:.2f}ms")


async def main():
    """Main function to run benchmarks."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Performance benchmark for Agentic Startup Studio")
    parser.add_argument("--url", default="http://localhost:8000", help="Base URL for API")
    parser.add_argument("--output", default="reports/benchmark_results.json", help="Output file for results")
    parser.add_argument("--quick", action="store_true", help="Run quick benchmark with fewer iterations")
    
    args = parser.parse_args()
    
    async with PerformanceBenchmark(args.url) as benchmark:
        if args.quick:
            # Quick benchmark for development
            print("Running quick benchmark...")
            await benchmark.benchmark_api_endpoint("/health", iterations=10)
            await benchmark.benchmark_api_endpoint("/", iterations=10)
        else:
            # Full comprehensive benchmark
            await benchmark.run_comprehensive_benchmark()
        
        benchmark.generate_report(args.output)


if __name__ == "__main__":
    asyncio.run(main())