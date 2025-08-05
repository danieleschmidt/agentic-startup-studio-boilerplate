#!/usr/bin/env python3
"""
Quantum Task Planner - Main Entry Point

A revolutionary task planning system that applies quantum computing principles
to optimize task scheduling, resource allocation, and execution strategies.
"""

import sys
import asyncio
from pathlib import Path

# Add quantum_task_planner to Python path
sys.path.insert(0, str(Path(__file__).parent))

from quantum_task_planner.cli import cli
from quantum_task_planner.api.quantum_api import quantum_api


def main():
    """Main entry point for the Quantum Task Planner"""
    
    # Check if running as API server
    if len(sys.argv) > 1 and sys.argv[1] == 'serve':
        print("🚀 Starting Quantum Task Planner API Server...")
        quantum_api.run()
        return
    
    # Otherwise run CLI
    cli()


if __name__ == '__main__':
    main()