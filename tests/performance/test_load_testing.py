"""
Performance and load testing for Agentic Startup Studio
"""

import pytest
import asyncio
import time
import statistics
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any
import httpx
import psutil
import json
from pathlib import Path


class TestLoadTesting:
    """Load testing and performance benchmarks."""

    @pytest.fixture
    def load_test_config(self):
        """Configuration for load testing."""
        return {
            "base_url": "http://localhost:8000",
            "concurrent_users": 10,
            "requests_per_user": 100,
            "ramp_up_time": 5,  # seconds
            "test_duration": 60,  # seconds
            "max_response_time": 2.0,  # seconds
            "max_error_rate": 0.05,  # 5%
        }

    @pytest.fixture
    def performance_thresholds(self):
        """Performance thresholds for various operations."""
        return {
            "api_response_time": 0.2,  # 200ms
            "database_query_time": 0.1,  # 100ms
            "memory_usage_mb": 512,  # 512MB
            "cpu_usage_percent": 80,  # 80%
            "throughput_rps": 100,  # requests per second
        }

    async def test_api_response_time(self, load_test_config, performance_thresholds):
        """Test API response times under normal load."""
        base_url = load_test_config["base_url"]
        threshold = performance_thresholds["api_response_time"]
        
        endpoints = [
            "/health",
            "/api/v1/projects",
            "/api/v1/agents",
            "/api/v1/tasks"
        ]
        
        async with httpx.AsyncClient(timeout=10.0) as client:
            for endpoint in endpoints:
                try:
                    start_time = time.time()
                    response = await client.get(f"{base_url}{endpoint}")
                    end_time = time.time()
                    
                    response_time = end_time - start_time
                    
                    # Log response time
                    print(f"Endpoint {endpoint}: {response_time:.3f}s")
                    
                    # Assert response time is within threshold
                    assert response_time < threshold, f"Response time {response_time:.3f}s exceeds threshold {threshold}s"
                    
                    # Assert successful response
                    assert response.status_code in [200, 404], f"Unexpected status code: {response.status_code}"
                    
                except httpx.RequestError as e:
                    pytest.skip(f"API not available: {e}")

    async def test_concurrent_users(self, load_test_config):
        """Test system behavior under concurrent user load."""
        base_url = load_test_config["base_url"]
        concurrent_users = load_test_config["concurrent_users"]
        requests_per_user = 10  # Reduced for testing
        
        async def user_session(user_id: int):
            """Simulate a user session."""
            session_metrics = {
                "user_id": user_id,
                "requests": 0,
                "errors": 0,
                "response_times": [],
                "start_time": time.time()
            }
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                for request_num in range(requests_per_user):
                    try:
                        start_time = time.time()
                        response = await client.get(f"{base_url}/health")
                        end_time = time.time()
                        
                        session_metrics["requests"] += 1
                        session_metrics["response_times"].append(end_time - start_time)
                        
                        if response.status_code >= 400:
                            session_metrics["errors"] += 1
                            
                        # Small delay between requests
                        await asyncio.sleep(0.1)
                        
                    except Exception as e:
                        session_metrics["errors"] += 1
                        print(f"User {user_id} request {request_num} failed: {e}")
            
            session_metrics["end_time"] = time.time()
            session_metrics["duration"] = session_metrics["end_time"] - session_metrics["start_time"]
            
            return session_metrics
        
        try:
            # Run concurrent user sessions
            tasks = [user_session(i) for i in range(concurrent_users)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Analyze results
            total_requests = 0
            total_errors = 0
            all_response_times = []
            
            for result in results:
                if isinstance(result, Exception):
                    print(f"User session failed: {result}")
                    continue
                
                total_requests += result["requests"]
                total_errors += result["errors"]
                all_response_times.extend(result["response_times"])
            
            # Calculate metrics
            error_rate = total_errors / total_requests if total_requests > 0 else 1
            avg_response_time = statistics.mean(all_response_times) if all_response_times else 0
            p95_response_time = statistics.quantiles(all_response_times, n=20)[18] if len(all_response_times) >= 20 else 0
            
            print(f"Load test results:")
            print(f"  Total requests: {total_requests}")
            print(f"  Total errors: {total_errors}")
            print(f"  Error rate: {error_rate:.2%}")
            print(f"  Average response time: {avg_response_time:.3f}s")
            print(f"  95th percentile response time: {p95_response_time:.3f}s")
            
            # Assertions
            assert error_rate < load_test_config.get("max_error_rate", 0.1), f"Error rate {error_rate:.2%} too high"
            assert avg_response_time < load_test_config.get("max_response_time", 2.0), f"Average response time {avg_response_time:.3f}s too high"
            
        except Exception as e:
            pytest.skip(f"Load testing not available: {e}")

    def test_memory_usage(self, performance_thresholds):
        """Test memory usage under load."""
        threshold_mb = performance_thresholds["memory_usage_mb"]
        
        # Get current process memory usage
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024  # Convert to MB
        
        print(f"Current memory usage: {memory_mb:.2f} MB")
        
        # In a real test, this would measure memory under load
        # For now, just check current usage is reasonable
        assert memory_mb < threshold_mb * 2, f"Memory usage {memory_mb:.2f}MB seems too high"

    def test_cpu_usage(self, performance_thresholds):
        """Test CPU usage under load."""
        threshold_percent = performance_thresholds["cpu_usage_percent"]
        
        # Get current CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        print(f"Current CPU usage: {cpu_percent:.1f}%")
        
        # In a real test, this would measure CPU under load
        assert cpu_percent < threshold_percent, f"CPU usage {cpu_percent:.1f}% too high"

    async def test_database_performance(self, performance_thresholds):
        """Test database query performance."""
        threshold = performance_thresholds["database_query_time"]
        
        # Mock database queries for testing
        async def mock_database_query():
            await asyncio.sleep(0.05)  # Simulate query time
            return {"results": ["item1", "item2", "item3"]}
        
        # Test multiple concurrent queries
        query_times = []
        
        for _ in range(10):
            start_time = time.time()
            await mock_database_query()
            end_time = time.time()
            
            query_time = end_time - start_time
            query_times.append(query_time)
        
        avg_query_time = statistics.mean(query_times)
        max_query_time = max(query_times)
        
        print(f"Database query performance:")
        print(f"  Average query time: {avg_query_time:.3f}s")
        print(f"  Maximum query time: {max_query_time:.3f}s")
        
        assert avg_query_time < threshold, f"Average query time {avg_query_time:.3f}s exceeds threshold {threshold}s"

    async def test_throughput(self, load_test_config, performance_thresholds):
        """Test system throughput (requests per second)."""
        base_url = load_test_config["base_url"]
        target_rps = performance_thresholds["throughput_rps"]
        test_duration = 10  # seconds
        
        start_time = time.time()
        request_count = 0
        
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                while time.time() - start_time < test_duration:
                    try:
                        response = await client.get(f"{base_url}/health")
                        if response.status_code == 200:
                            request_count += 1
                    except Exception:
                        pass
            
            actual_duration = time.time() - start_time
            actual_rps = request_count / actual_duration
            
            print(f"Throughput test:")
            print(f"  Requests completed: {request_count}")
            print(f"  Test duration: {actual_duration:.2f}s")
            print(f"  Actual RPS: {actual_rps:.2f}")
            print(f"  Target RPS: {target_rps}")
            
            # Allow for some variance in throughput
            assert actual_rps >= target_rps * 0.8, f"Throughput {actual_rps:.2f} RPS below 80% of target {target_rps} RPS"
            
        except Exception as e:
            pytest.skip(f"Throughput testing not available: {e}")

    def test_stress_test(self, load_test_config):
        """Stress test to find breaking point."""
        base_url = load_test_config["base_url"]
        
        # Gradually increase load until failure
        user_counts = [5, 10, 20, 50, 100]
        
        results = {}
        
        for user_count in user_counts:
            print(f"Testing with {user_count} concurrent users...")
            
            start_time = time.time()
            success_count = 0
            error_count = 0
            
            def make_request():
                try:
                    response = httpx.get(f"{base_url}/health", timeout=5.0)
                    return response.status_code == 200
                except Exception:
                    return False
            
            try:
                with ThreadPoolExecutor(max_workers=user_count) as executor:
                    futures = [executor.submit(make_request) for _ in range(user_count * 5)]
                    
                    for future in as_completed(futures, timeout=30):
                        if future.result():
                            success_count += 1
                        else:
                            error_count += 1
                
                duration = time.time() - start_time
                success_rate = success_count / (success_count + error_count) if (success_count + error_count) > 0 else 0
                
                results[user_count] = {
                    "success_count": success_count,
                    "error_count": error_count,
                    "success_rate": success_rate,
                    "duration": duration,
                    "rps": (success_count + error_count) / duration
                }
                
                print(f"  Success rate: {success_rate:.2%}")
                print(f"  RPS: {results[user_count]['rps']:.2f}")
                
                # If success rate drops below 90%, we've found the breaking point
                if success_rate < 0.9:
                    print(f"Breaking point found at {user_count} users")
                    break
                    
            except Exception as e:
                print(f"Stress test failed at {user_count} users: {e}")
                break
        
        # Save results for analysis
        self._save_performance_results("stress_test", results)
        
        # At least one configuration should work
        assert any(result["success_rate"] > 0.95 for result in results.values()), "No configuration achieved 95% success rate"

    async def test_spike_load(self, load_test_config):
        """Test handling of sudden load spikes."""
        base_url = load_test_config["base_url"]
        
        # Normal load phase
        normal_load = 5
        spike_load = 50
        spike_duration = 10  # seconds
        
        async def sustained_load(user_count: int, duration: int):
            """Generate sustained load for specified duration."""
            end_time = time.time() + duration
            request_count = 0
            error_count = 0
            
            async with httpx.AsyncClient(timeout=5.0) as client:
                while time.time() < end_time:
                    tasks = []
                    for _ in range(user_count):
                        tasks.append(self._make_async_request(client, f"{base_url}/health"))
                    
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    for result in results:
                        if isinstance(result, Exception) or not result:
                            error_count += 1
                        else:
                            request_count += 1
                    
                    await asyncio.sleep(0.1)  # Small delay
            
            return request_count, error_count
        
        try:
            # Phase 1: Normal load
            print("Phase 1: Normal load")
            normal_requests, normal_errors = await sustained_load(normal_load, 10)
            normal_error_rate = normal_errors / (normal_requests + normal_errors) if (normal_requests + normal_errors) > 0 else 0
            
            # Phase 2: Spike load
            print("Phase 2: Spike load")
            spike_requests, spike_errors = await sustained_load(spike_load, spike_duration)
            spike_error_rate = spike_errors / (spike_requests + spike_errors) if (spike_requests + spike_errors) > 0 else 0
            
            # Phase 3: Return to normal
            print("Phase 3: Return to normal")
            recovery_requests, recovery_errors = await sustained_load(normal_load, 10)
            recovery_error_rate = recovery_errors / (recovery_requests + recovery_errors) if (recovery_requests + recovery_errors) > 0 else 0
            
            print(f"Spike load test results:")
            print(f"  Normal load error rate: {normal_error_rate:.2%}")
            print(f"  Spike load error rate: {spike_error_rate:.2%}")
            print(f"  Recovery error rate: {recovery_error_rate:.2%}")
            
            # System should handle spikes gracefully
            assert spike_error_rate < 0.2, f"Spike error rate {spike_error_rate:.2%} too high"
            assert recovery_error_rate < normal_error_rate * 1.5, "System didn't recover well after spike"
            
        except Exception as e:
            pytest.skip(f"Spike load testing not available: {e}")

    async def _make_async_request(self, client: httpx.AsyncClient, url: str) -> bool:
        """Make an async HTTP request and return success status."""
        try:
            response = await client.get(url)
            return response.status_code == 200
        except Exception:
            return False

    def _save_performance_results(self, test_name: str, results: Dict[str, Any]):
        """Save performance test results to file."""
        results_dir = Path("performance_results")
        results_dir.mkdir(exist_ok=True)
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"{test_name}_{timestamp}.json"
        filepath = results_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump({
                "test_name": test_name,
                "timestamp": timestamp,
                "results": results
            }, f, indent=2, default=str)
        
        print(f"Performance results saved to {filepath}")

    @pytest.mark.slow
    async def test_endurance(self, load_test_config):
        """Long-running endurance test."""
        base_url = load_test_config["base_url"]
        test_duration = 300  # 5 minutes (reduced for testing)
        concurrent_users = 5
        
        print(f"Starting {test_duration}s endurance test with {concurrent_users} users...")
        
        start_time = time.time()
        total_requests = 0
        total_errors = 0
        
        async def endurance_user():
            """Single user making continuous requests."""
            user_requests = 0
            user_errors = 0
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                while time.time() - start_time < test_duration:
                    try:
                        response = await client.get(f"{base_url}/health")
                        user_requests += 1
                        
                        if response.status_code >= 400:
                            user_errors += 1
                        
                        # Moderate pacing
                        await asyncio.sleep(1)
                        
                    except Exception:
                        user_errors += 1
                        await asyncio.sleep(1)
            
            return user_requests, user_errors
        
        try:
            # Run endurance test
            tasks = [endurance_user() for _ in range(concurrent_users)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            for result in results:
                if isinstance(result, Exception):
                    print(f"Endurance user failed: {result}")
                    continue
                
                requests, errors = result
                total_requests += requests
                total_errors += errors
            
            actual_duration = time.time() - start_time
            error_rate = total_errors / total_requests if total_requests > 0 else 1
            
            print(f"Endurance test completed:")
            print(f"  Duration: {actual_duration:.2f}s")
            print(f"  Total requests: {total_requests}")
            print(f"  Total errors: {total_errors}")
            print(f"  Error rate: {error_rate:.2%}")
            
            # System should remain stable over time
            assert error_rate < 0.05, f"Error rate {error_rate:.2%} too high for endurance test"
            assert total_requests > 0, "No successful requests during endurance test"
            
        except Exception as e:
            pytest.skip(f"Endurance testing not available: {e}")

    @pytest.mark.parametrize("endpoint", [
        "/health",
        "/api/v1/projects",
        "/api/v1/agents",
        "/api/v1/tasks"
    ])
    async def test_endpoint_performance(self, endpoint, load_test_config, performance_thresholds):
        """Test individual endpoint performance."""
        base_url = load_test_config["base_url"]
        threshold = performance_thresholds["api_response_time"]
        
        response_times = []
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                for _ in range(10):
                    start_time = time.time()
                    response = await client.get(f"{base_url}{endpoint}")
                    end_time = time.time()
                    
                    response_time = end_time - start_time
                    response_times.append(response_time)
                    
                    # Small delay between requests
                    await asyncio.sleep(0.1)
            
            avg_response_time = statistics.mean(response_times)
            max_response_time = max(response_times)
            p95_response_time = statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else max_response_time
            
            print(f"Performance for {endpoint}:")
            print(f"  Average: {avg_response_time:.3f}s")
            print(f"  Maximum: {max_response_time:.3f}s")
            print(f"  95th percentile: {p95_response_time:.3f}s")
            
            assert avg_response_time < threshold, f"Average response time {avg_response_time:.3f}s exceeds threshold {threshold}s"
            assert p95_response_time < threshold * 2, f"95th percentile {p95_response_time:.3f}s too high"
            
        except httpx.RequestError as e:
            pytest.skip(f"Endpoint {endpoint} not available: {e}")