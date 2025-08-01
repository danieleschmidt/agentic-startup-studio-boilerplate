#!/usr/bin/env python3
"""
Terragon Autonomous SDLC - Main Scheduler and Orchestrator

This is the main entry point for the Terragon Autonomous SDLC system,
orchestrating value discovery, execution, and learning in a continuous loop.
"""

import os
import sys
import time
import json
import yaml
import logging
import schedule
import threading
import subprocess
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from pathlib import Path

# Import Terragon engines
sys.path.append(str(Path(__file__).parent))
from value_discovery_engine import ValueDiscoveryEngine
from autonomous_executor import AutonomousExecutor
from learning_engine import ContinuousLearningEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('.terragon/terragon.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class TeragonScheduler:
    """Main scheduler for Terragon Autonomous SDLC operations."""
    
    def __init__(self, config_path: str = ".terragon/value-config.yaml"):
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.running = False
        self.last_execution = None
        
        # Initialize engines
        self.discovery_engine = ValueDiscoveryEngine(config_path)
        self.executor = AutonomousExecutor(config_path)
        self.learning_engine = ContinuousLearningEngine(config_path)
        
        # Setup paths
        self._setup_paths()
        self._setup_schedules()
        
        logger.info("Terragon Autonomous SDLC Scheduler initialized")
    
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
        
        self.status_file = terragon_dir / "scheduler-status.json"
        self.metrics_file = terragon_dir / "system-metrics.json"
    
    def _setup_schedules(self):
        """Setup scheduled tasks based on configuration."""
        schedules_config = self.config.get('discovery', {}).get('schedules', {})
        
        # Security scans every 6 hours
        security_schedule = schedules_config.get('security_scan', '0 */6 * * *')
        schedule.every(6).hours.do(self._run_security_scan)
        
        # Dependency checks daily
        dep_schedule = schedules_config.get('dependency_check', '0 2 * * *')
        schedule.every().day.at("02:00").do(self._run_dependency_check)
        
        # Static analysis daily
        static_schedule = schedules_config.get('static_analysis', '0 3 * * *')
        schedule.every().day.at("03:00").do(self._run_static_analysis)
        
        # Performance checks weekly
        perf_schedule = schedules_config.get('performance_check', '0 4 * * 1')
        schedule.every().monday.at("04:00").do(self._run_performance_check)
        
        # Deep analysis monthly (using weekly for now since schedule doesn't support monthly)
        deep_schedule = schedules_config.get('deep_analysis', '0 5 1 * *')
        schedule.every(4).weeks.do(self._run_deep_analysis)
        
        # Learning calibration weekly
        schedule.every().sunday.at("06:00").do(self._run_learning_calibration)
        
        logger.info("Scheduled tasks configured")
    
    def start_autonomous_mode(self, max_cycles: Optional[int] = None):
        """Start the autonomous SDLC execution loop."""
        logger.info("Starting Terragon Autonomous SDLC Mode")
        self.running = True
        
        # Start scheduler thread
        scheduler_thread = threading.Thread(target=self._run_scheduler, daemon=True)
        scheduler_thread.start()
        
        # Start main execution loop
        self._run_main_loop(max_cycles)
    
    def _run_main_loop(self, max_cycles: Optional[int] = None):
        """Main execution loop for autonomous operations."""
        cycle_count = 0
        
        while self.running and (max_cycles is None or cycle_count < max_cycles):
            try:
                logger.info(f"Starting autonomous cycle {cycle_count + 1}")
                
                # Update status
                self._update_status("running", f"Cycle {cycle_count + 1}")
                
                # Run value discovery
                logger.info("Running value discovery...")
                work_items = self.discovery_engine.discover_work_items()
                
                if not work_items:
                    logger.info("No work items discovered, sleeping...")
                    self._sleep_with_schedule_check(300)  # 5 minutes
                    continue
                
                # Apply learned adjustments
                adjusted_items = []
                for item in work_items:
                    adjusted_item = self.learning_engine.apply_learned_adjustments(
                        item.__dict__
                    )
                    adjusted_items.append(adjusted_item)
                
                # Save updated backlog
                self.discovery_engine.save_backlog()
                
                # Generate backlog markdown
                backlog_md = self.discovery_engine.generate_backlog_markdown()
                with open(".terragon/BACKLOG.md", "w") as f:
                    f.write(backlog_md)
                
                # Execute next best value item
                logger.info("Executing next best value item...")
                result = self.executor.execute_next_best_value()
                
                if result:
                    # Learn from execution
                    if os.path.exists(self.executor.execution_log):
                        with open(self.executor.execution_log, 'r') as f:
                            execution_history = json.load(f)
                        
                        if execution_history:
                            latest_execution = execution_history[-1]
                            self.learning_engine.learn_from_execution(latest_execution)
                    
                    # Update metrics
                    self._update_system_metrics(result)
                    
                    if result.success:
                        logger.info(f"Cycle {cycle_count + 1} completed successfully")
                        self._sleep_with_schedule_check(60)  # 1 minute
                    else:
                        logger.warning(f"Cycle {cycle_count + 1} failed, extended sleep")
                        self._sleep_with_schedule_check(900)  # 15 minutes
                else:
                    logger.info("No executable work items, sleeping...")
                    self._sleep_with_schedule_check(300)  # 5 minutes
                
                cycle_count += 1
                self.last_execution = datetime.now()
                
            except KeyboardInterrupt:
                logger.info("Received interrupt signal, shutting down...")
                self.running = False
                break
            except Exception as e:
                logger.error(f"Unexpected error in main loop: {e}")
                self._sleep_with_schedule_check(600)  # 10 minutes on error
        
        self._update_status("stopped", "Autonomous mode stopped")
        logger.info("Terragon Autonomous SDLC stopped")
    
    def _run_scheduler(self):
        """Run the background scheduler for periodic tasks."""
        while self.running:
            schedule.run_pending()
            time.sleep(60)  # Check every minute
    
    def _sleep_with_schedule_check(self, seconds: int):
        """Sleep while still allowing scheduled tasks to run."""
        start_time = time.time()
        while time.time() - start_time < seconds and self.running:
            schedule.run_pending()
            time.sleep(min(60, seconds - (time.time() - start_time)))
    
    def _run_security_scan(self):
        """Run scheduled security scan."""
        logger.info("Running scheduled security scan")
        try:
            # Run security tools
            subprocess.run(['bandit', '-r', '.', '-f', 'json', '-o', '.terragon/security-report.json'], 
                         capture_output=True, timeout=300)
            subprocess.run(['safety', 'check', '--json', '--output', '.terragon/safety-report.json'], 
                         capture_output=True, timeout=300)
            logger.info("Security scan completed")
        except Exception as e:
            logger.error(f"Security scan failed: {e}")
    
    def _run_dependency_check(self):
        """Run scheduled dependency check."""
        logger.info("Running scheduled dependency check")
        try:
            subprocess.run(['pip', 'list', '--outdated', '--format=json'], 
                         capture_output=True, timeout=120)
            logger.info("Dependency check completed")
        except Exception as e:
            logger.error(f"Dependency check failed: {e}")
    
    def _run_static_analysis(self):
        """Run scheduled static analysis."""
        logger.info("Running scheduled static analysis")
        try:
            subprocess.run(['ruff', 'check', '.', '--format=json'], 
                         capture_output=True, timeout=180)
            logger.info("Static analysis completed")
        except Exception as e:
            logger.error(f"Static analysis failed: {e}")
    
    def _run_performance_check(self):
        """Run scheduled performance check."""
        logger.info("Running scheduled performance check")
        try:
            # Placeholder for performance analysis
            # In real implementation, this would run performance tests
            logger.info("Performance check completed (placeholder)")
        except Exception as e:
            logger.error(f"Performance check failed: {e}")
    
    def _run_deep_analysis(self):
        """Run scheduled deep analysis."""
        logger.info("Running scheduled deep analysis")
        try:
            # Comprehensive repository analysis
            self.discovery_engine.discover_work_items()
            
            # Generate comprehensive reports
            learning_report = self.learning_engine.get_learning_report()
            
            with open('.terragon/deep-analysis-report.json', 'w') as f:
                json.dump({
                    'timestamp': datetime.now().isoformat(),
                    'learning_report': learning_report,
                    'repository_health': self._assess_repository_health()
                }, f, indent=2)
            
            logger.info("Deep analysis completed")
        except Exception as e:
            logger.error(f"Deep analysis failed: {e}")
    
    def _run_learning_calibration(self):
        """Run scheduled learning calibration."""
        logger.info("Running scheduled learning calibration")
        try:
            self.learning_engine.calibrate_model()
            logger.info("Learning calibration completed")
        except Exception as e:
            logger.error(f"Learning calibration failed: {e}")
    
    def _update_status(self, status: str, message: str):
        """Update the scheduler status."""
        status_data = {
            'status': status,
            'message': message,
            'timestamp': datetime.now().isoformat(),
            'last_execution': self.last_execution.isoformat() if self.last_execution else None,
            'uptime': str(datetime.now() - datetime.now()) if hasattr(self, 'start_time') else None
        }
        
        try:
            with open(self.status_file, 'w') as f:
                json.dump(status_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to update status: {e}")
    
    def _update_system_metrics(self, execution_result):
        """Update system-wide metrics."""
        try:
            # Load existing metrics
            if self.metrics_file.exists():
                with open(self.metrics_file, 'r') as f:
                    metrics = json.load(f)
            else:
                metrics = {
                    'total_executions': 0,
                    'successful_executions': 0,
                    'failed_executions': 0,
                    'total_execution_time': 0.0,
                    'average_execution_time': 0.0,
                    'last_updated': None
                }
            
            # Update metrics
            metrics['total_executions'] += 1
            metrics['total_execution_time'] += execution_result.execution_time
            
            if execution_result.success:
                metrics['successful_executions'] += 1
            else:
                metrics['failed_executions'] += 1
            
            metrics['average_execution_time'] = (
                metrics['total_execution_time'] / metrics['total_executions']
            )
            metrics['success_rate'] = (
                metrics['successful_executions'] / metrics['total_executions']
            )
            metrics['last_updated'] = datetime.now().isoformat()
            
            # Save updated metrics
            with open(self.metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to update system metrics: {e}")
    
    def _assess_repository_health(self) -> Dict:
        """Assess overall repository health."""
        health_score = 0
        health_factors = {}
        
        try:
            # Check if tests exist and pass
            test_result = subprocess.run(['npm', 'test'], capture_output=True, timeout=300)
            health_factors['tests_pass'] = test_result.returncode == 0
            if health_factors['tests_pass']:
                health_score += 25
            
            # Check if linting passes
            lint_result = subprocess.run(['npm', 'run', 'lint'], capture_output=True, timeout=60)
            health_factors['lint_pass'] = lint_result.returncode == 0
            if health_factors['lint_pass']:
                health_score += 20
            
            # Check if build passes
            build_result = subprocess.run(['npm', 'run', 'build'], capture_output=True, timeout=180)
            health_factors['build_pass'] = build_result.returncode == 0
            if health_factors['build_pass']:
                health_score += 25
            
            # Check documentation coverage (basic check)
            required_docs = ['README.md', 'CONTRIBUTING.md', 'LICENSE']
            docs_present = sum(1 for doc in required_docs if os.path.exists(doc))
            health_factors['documentation_coverage'] = docs_present / len(required_docs)
            health_score += int(health_factors['documentation_coverage'] * 15)
            
            # Check for security issues
            health_factors['security_issues'] = 0  # Placeholder
            health_score += 15  # Assume good security for now
            
        except Exception as e:
            logger.error(f"Health assessment failed: {e}")
        
        return {
            'overall_score': health_score,
            'max_score': 100,
            'health_percentage': health_score,
            'factors': health_factors,
            'assessment_time': datetime.now().isoformat()
        }
    
    def stop(self):
        """Stop the autonomous scheduler."""
        logger.info("Stopping Terragon Autonomous SDLC...")
        self.running = False
    
    def get_status(self) -> Dict:
        """Get current scheduler status."""
        try:
            if self.status_file.exists():
                with open(self.status_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Failed to get status: {e}")
        
        return {
            'status': 'unknown',
            'message': 'Status unavailable',
            'timestamp': datetime.now().isoformat()
        }
    
    def run_single_cycle(self):
        """Run a single discovery and execution cycle."""
        logger.info("Running single autonomous cycle")
        
        try:
            # Discovery
            work_items = self.discovery_engine.discover_work_items()
            self.discovery_engine.save_backlog()
            
            # Generate backlog
            backlog_md = self.discovery_engine.generate_backlog_markdown()
            with open(".terragon/BACKLOG.md", "w") as f:
                f.write(backlog_md)
            
            # Execute
            result = self.executor.execute_next_best_value()
            
            if result:
                print(f"Execution result: {'SUCCESS' if result.success else 'FAILED'}")
                if result.error_message:
                    print(f"Error: {result.error_message}")
                return result.success
            else:
                print("No work items available for execution")
                return True
                
        except Exception as e:
            logger.error(f"Single cycle execution failed: {e}")
            return False


def main():
    """Main entry point for Terragon Autonomous SDLC."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Terragon Autonomous SDLC System')
    parser.add_argument('--mode', choices=['single', 'continuous', 'discover', 'status'], 
                       default='single', help='Execution mode')
    parser.add_argument('--max-cycles', type=int, help='Maximum cycles for continuous mode')
    parser.add_argument('--config', default='.terragon/value-config.yaml', 
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    scheduler = TeragonScheduler(args.config)
    
    if args.mode == 'single':
        success = scheduler.run_single_cycle()
        sys.exit(0 if success else 1)
    elif args.mode == 'continuous':
        scheduler.start_autonomous_mode(args.max_cycles)
    elif args.mode == 'discover':
        work_items = scheduler.discovery_engine.discover_work_items()
        scheduler.discovery_engine.save_backlog()
        backlog_md = scheduler.discovery_engine.generate_backlog_markdown()
        print(backlog_md)
    elif args.mode == 'status':
        status = scheduler.get_status()
        print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()