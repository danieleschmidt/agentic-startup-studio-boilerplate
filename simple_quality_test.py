#!/usr/bin/env python3
"""
Simple Quality Test Script
Tests core functionality for quality gate validation
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

def test_imports():
    """Test core imports"""
    try:
        from quantum_task_planner import cli
        print("✅ CLI import successful")
        return True
    except Exception as e:
        print(f"❌ CLI import failed: {e}")
        return False

def test_main_help():
    """Test main.py help"""
    try:
        import subprocess
        result = subprocess.run([sys.executable, 'main.py', '--help'], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✅ main.py --help successful")
            return True
        else:
            print(f"❌ main.py --help failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ main.py test failed: {e}")
        return False

def test_generation_9():
    """Test Generation 9 enhancement"""
    try:
        import subprocess
        result = subprocess.run([sys.executable, 'generation_9_progressive_enhancement.py'], 
                              capture_output=True, text=True, timeout=30)
        if result.returncode == 0:
            print("✅ Generation 9 enhancement successful")
            return True
        else:
            print(f"❌ Generation 9 enhancement failed: {result.stderr}")
            return False
    except Exception as e:
        print(f"❌ Generation 9 test failed: {e}")
        return False

if __name__ == "__main__":
    print("🧪 SIMPLE QUALITY TEST")
    print("=" * 30)
    
    tests = [test_imports, test_main_help, test_generation_9]
    passed = 0
    
    for test in tests:
        if test():
            passed += 1
    
    print(f"\nResults: {passed}/{len(tests)} tests passed")
    print("✅ All core functionality working!" if passed == len(tests) else "⚠️ Some tests failed")