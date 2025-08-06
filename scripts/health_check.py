#!/usr/bin/env python3
"""
Health check script for Docker container
"""

import sys
import requests
import json
from urllib.parse import urljoin


def check_health():
    """Check application health"""
    try:
        # Check main health endpoint
        response = requests.get("http://localhost:8000/api/v1/health", timeout=5)
        
        if response.status_code != 200:
            print(f"Health check failed: HTTP {response.status_code}")
            return False
        
        health_data = response.json()
        
        # Check overall status
        if health_data.get("status") not in ["healthy", "degraded"]:
            print(f"Health check failed: status is {health_data.get('status')}")
            return False
        
        # Check critical health checks
        checks = health_data.get("checks", {})
        for check_name, check_result in checks.items():
            if check_result.get("critical", False) and check_result.get("status") in ["unhealthy", "critical"]:
                print(f"Critical health check failed: {check_name} - {check_result.get('message', 'Unknown error')}")
                return False
        
        print("Health check passed")
        return True
        
    except requests.exceptions.RequestException as e:
        print(f"Health check failed: {e}")
        return False
    except Exception as e:
        print(f"Health check error: {e}")
        return False


if __name__ == "__main__":
    if check_health():
        sys.exit(0)
    else:
        sys.exit(1)