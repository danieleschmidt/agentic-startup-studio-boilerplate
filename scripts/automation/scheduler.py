#!/usr/bin/env python3
"""
Automation scheduler for repository maintenance tasks
Manages scheduled execution of various automation scripts
"""

import os
import sys
import time
import json
import schedule
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, List, Callable
import subprocess
import threading

# Add the scripts directory to path for imports
sys.path.append(str(Path(__file__).parent))

try:
    from metrics_collector import MetricsCollector
    from repo_maintenance import RepositoryMaintenance
except ImportError as e:
    logging.error(f"Failed to import automation modules: {e}")
    sys.exit(1)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('automation-scheduler.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AutomationScheduler:
    def __init__(self, repo_path: str = "."):
        self.repo_path = Path(repo_path)
        self.config_file = self.repo_path / ".github" / "automation-config.json"
        self.status_file = self.repo_path / ".github" / "automation-status.json"
        self.running = False
        self.config = self.load_configuration()
        
        # Initialize automation modules
        self.metrics_collector = MetricsCollector(repo_path)
        self.repo_maintenance = RepositoryMaintenance(repo_path)
        
    def load_configuration(self) -> Dict[str, Any]:
        """Load automation configuration"""
        default_config = {
            "schedules": {
                "metrics_collection": {
                    "enabled": True,
                    "cron": "0 6 * * *",  # Daily at 6 AM
                    "description": "Collect repository metrics"
                },
                "dependency_check": {
                    "enabled": True,
                    "cron": "0 2 * * 1",  # Weekly on Monday at 2 AM
                    "description": "Check for dependency updates"
                },
                "security_scan": {
                    "enabled": True,
                    "cron": "0 1 * * *",  # Daily at 1 AM
                    "description": "Run security scans"
                },
                "repository_cleanup": {
                    "enabled": True,
                    "cron": "0 3 * * 0",  # Weekly on Sunday at 3 AM
                    "description": "Clean up repository artifacts"
                },
                "health_check": {
                    "enabled": True,
                    "cron": "0 */6 * * *",  # Every 6 hours
                    "description": "Repository health check"
                },
                "backup_verification": {
                    "enabled": True,
                    "cron": "0 4 * * 0",  # Weekly on Sunday at 4 AM
                    "description": "Verify backup integrity"
                },
                "performance_monitoring": {
                    "enabled": True,
                    "cron": "0 */12 * * *",  # Every 12 hours
                    "description": "Monitor performance metrics"
                }
            },
            "notifications": {
                "slack_webhook": os.getenv("SLACK_WEBHOOK_URL"),
                "email_notifications": os.getenv("EMAIL_NOTIFICATIONS", "false").lower() == "true",
                "failure_only": True
            },
            "settings": {
                "max_concurrent_tasks": 2,
                "task_timeout_minutes": 60,
                "retry_failed_tasks": True,
                "retain_logs_days": 30
            }
        }
        
        if self.config_file.exists():
            try:
                with open(self.config_file, 'r') as f:
                    loaded_config = json.load(f)
                    # Merge with defaults
                    default_config.update(loaded_config)
            except Exception as e:
                logger.warning(f"Error loading config file: {e}, using defaults")
        
        return default_config
    
    def save_configuration(self) -> None:
        """Save current configuration"""
        self.config_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.config_file, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def update_status(self, task_name: str, status: str, details: Dict[str, Any] = None) -> None:
        """Update automation status"""
        current_status = {}
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r') as f:
                    current_status = json.load(f)
            except Exception:
                pass
        
        current_status[task_name] = {
            "status": status,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "details": details or {}
        }
        
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.status_file, 'w') as f:
            json.dump(current_status, f, indent=2)
    
    def setup_schedules(self) -> None:
        """Setup scheduled tasks"""
        logger.info("Setting up automation schedules...")
        
        schedules_config = self.config.get("schedules", {})
        
        for task_name, task_config in schedules_config.items():
            if not task_config.get("enabled", False):
                logger.info(f"Skipping disabled task: {task_name}")
                continue
            
            cron_expr = task_config.get("cron")
            if not cron_expr:
                logger.warning(f"No cron expression for task: {task_name}")
                continue
            
            # Convert cron to schedule calls
            try:
                self.schedule_task(task_name, cron_expr)
                logger.info(f"Scheduled {task_name}: {task_config.get('description', 'No description')}")
            except Exception as e:
                logger.error(f"Failed to schedule {task_name}: {e}")
    
    def schedule_task(self, task_name: str, cron_expr: str) -> None:
        """Schedule a task based on cron expression"""
        # Parse basic cron expressions (minute hour day_of_month month day_of_week)
        parts = cron_expr.strip().split()
        if len(parts) != 5:
            raise ValueError(f"Invalid cron expression: {cron_expr}")
        
        minute, hour, day_month, month, day_week = parts
        
        # Create the scheduled job
        if day_week != "*" and day_month == "*":
            # Weekly schedule
            day_names = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
            if day_week.isdigit() and 0 <= int(day_week) <= 6:
                day_name = day_names[int(day_week)]
                if hour.isdigit() and minute.isdigit():
                    schedule.every().week.at(f"{hour.zfill(2)}:{minute.zfill(2)}").do(
                        self.run_task_wrapper, task_name
                    ).tag(task_name)
        elif hour.isdigit() and minute.isdigit():
            if hour == "*/6":
                # Every 6 hours
                for h in [0, 6, 12, 18]:
                    schedule.every().day.at(f"{h:02d}:{minute.zfill(2)}").do(
                        self.run_task_wrapper, task_name
                    ).tag(task_name)
            elif hour == "*/12":
                # Every 12 hours
                for h in [0, 12]:
                    schedule.every().day.at(f"{h:02d}:{minute.zfill(2)}").do(
                        self.run_task_wrapper, task_name
                    ).tag(task_name)
            else:
                # Daily schedule
                schedule.every().day.at(f"{hour.zfill(2)}:{minute.zfill(2)}").do(
                    self.run_task_wrapper, task_name
                ).tag(task_name)
    
    def run_task_wrapper(self, task_name: str) -> None:
        """Wrapper to run tasks with error handling and logging"""
        logger.info(f"Starting scheduled task: {task_name}")
        self.update_status(task_name, "running")
        
        try:
            start_time = time.time()
            result = self.run_task(task_name)
            duration = time.time() - start_time
            
            self.update_status(task_name, "completed", {
                "duration_seconds": round(duration, 2),
                "result": result
            })
            
            logger.info(f"Task {task_name} completed successfully in {duration:.2f} seconds")
            
            # Send success notification if configured
            if not self.config.get("notifications", {}).get("failure_only", True):
                self.send_notification(f"✅ Task {task_name} completed successfully", result)
                
        except Exception as e:
            error_msg = str(e)
            logger.error(f"Task {task_name} failed: {error_msg}")
            
            self.update_status(task_name, "failed", {"error": error_msg})
            
            # Send failure notification
            self.send_notification(f"❌ Task {task_name} failed", {"error": error_msg})
            
            # Retry if configured
            if self.config.get("settings", {}).get("retry_failed_tasks", True):
                logger.info(f"Scheduling retry for {task_name} in 30 minutes")
                schedule.every(30).minutes.do(self.run_task_wrapper, task_name).tag(f"{task_name}_retry")
    
    def run_task(self, task_name: str) -> Dict[str, Any]:
        """Run a specific automation task"""
        task_methods = {
            "metrics_collection": self.run_metrics_collection,
            "dependency_check": self.run_dependency_check,
            "security_scan": self.run_security_scan,
            "repository_cleanup": self.run_repository_cleanup,
            "health_check": self.run_health_check,
            "backup_verification": self.run_backup_verification,
            "performance_monitoring": self.run_performance_monitoring
        }
        
        task_method = task_methods.get(task_name)
        if not task_method:
            raise ValueError(f"Unknown task: {task_name}")
        
        return task_method()
    
    def run_metrics_collection(self) -> Dict[str, Any]:
        """Run metrics collection"""
        logger.info("Running metrics collection...")
        metrics = self.metrics_collector.collect_all_metrics()
        self.metrics_collector.update_metrics_file(metrics)
        
        return {
            "metrics_collected": len(metrics),
            "status": "success"
        }
    
    def run_dependency_check(self) -> Dict[str, Any]:
        """Run dependency update check"""
        logger.info("Running dependency check...")
        return self.repo_maintenance.check_dependency_updates()
    
    def run_security_scan(self) -> Dict[str, Any]:
        """Run security scan"""
        logger.info("Running security scan...")
        return self.repo_maintenance.run_security_scan()
    
    def run_repository_cleanup(self) -> Dict[str, Any]:
        """Run repository cleanup"""
        logger.info("Running repository cleanup...")
        return self.repo_maintenance.cleanup_repository()
    
    def run_health_check(self) -> Dict[str, Any]:
        """Run repository health check"""
        logger.info("Running health check...")
        return self.repo_maintenance.repository_health_check()
    
    def run_backup_verification(self) -> Dict[str, Any]:
        """Run backup verification"""
        logger.info("Running backup verification...")
        return self.repo_maintenance.verify_backups()
    
    def run_performance_monitoring(self) -> Dict[str, Any]:
        """Run performance monitoring"""
        logger.info("Running performance monitoring...")
        return self.repo_maintenance.check_performance_metrics()
    
    def send_notification(self, message: str, details: Dict[str, Any] = None) -> None:
        """Send notification via configured channels"""
        notifications_config = self.config.get("notifications", {})
        
        # Slack notification
        slack_webhook = notifications_config.get("slack_webhook")
        if slack_webhook:
            try:
                import requests
                payload = {
                    "text": message,
                    "attachments": [
                        {
                            "color": "good" if "✅" in message else "danger",
                            "fields": [
                                {
                                    "title": "Details",
                                    "value": json.dumps(details, indent=2) if details else "No additional details",
                                    "short": False
                                }
                            ]
                        }
                    ]
                }
                
                response = requests.post(slack_webhook, json=payload, timeout=10)
                if response.status_code == 200:
                    logger.info("Slack notification sent successfully")
                else:
                    logger.warning(f"Failed to send Slack notification: {response.status_code}")
                    
            except Exception as e:
                logger.error(f"Error sending Slack notification: {e}")
        
        # Email notification (basic implementation)
        if notifications_config.get("email_notifications", False):
            try:
                self.send_email_notification(message, details)
            except Exception as e:
                logger.error(f"Error sending email notification: {e}")
    
    def send_email_notification(self, message: str, details: Dict[str, Any] = None) -> None:
        """Send email notification (placeholder implementation)"""
        # This would require email server configuration
        logger.info(f"Email notification would be sent: {message}")
    
    def start_scheduler(self) -> None:
        """Start the automation scheduler"""
        logger.info("Starting automation scheduler...")
        self.running = True
        
        # Setup schedules
        self.setup_schedules()
        
        logger.info(f"Scheduler started with {len(schedule.jobs)} jobs")
        
        # Main scheduler loop
        while self.running:
            try:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
            except KeyboardInterrupt:
                logger.info("Received interrupt signal, stopping scheduler...")
                self.stop_scheduler()
            except Exception as e:
                logger.error(f"Scheduler error: {e}")
                time.sleep(60)  # Continue after error
    
    def stop_scheduler(self) -> None:
        """Stop the automation scheduler"""
        logger.info("Stopping automation scheduler...")
        self.running = False
        schedule.clear()
        logger.info("Scheduler stopped")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current automation status"""
        status = {}
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r') as f:
                    status = json.load(f)
            except Exception:
                pass
        
        return {
            "running": self.running,
            "scheduled_jobs": len(schedule.jobs),
            "task_status": status,
            "next_run": str(schedule.next_run()) if schedule.jobs else None
        }
    
    def run_task_now(self, task_name: str) -> Dict[str, Any]:
        """Run a task immediately (for manual execution)"""
        logger.info(f"Running task immediately: {task_name}")
        
        try:
            result = self.run_task(task_name)
            self.update_status(task_name, "completed_manual", result)
            return {"status": "success", "result": result}
        except Exception as e:
            error_msg = str(e)
            self.update_status(task_name, "failed_manual", {"error": error_msg})
            return {"status": "failed", "error": error_msg}
    
    def list_available_tasks(self) -> List[str]:
        """List all available tasks"""
        return list(self.config.get("schedules", {}).keys())
    
    def enable_task(self, task_name: str) -> bool:
        """Enable a scheduled task"""
        if task_name in self.config.get("schedules", {}):
            self.config["schedules"][task_name]["enabled"] = True
            self.save_configuration()
            logger.info(f"Enabled task: {task_name}")
            return True
        return False
    
    def disable_task(self, task_name: str) -> bool:
        """Disable a scheduled task"""
        if task_name in self.config.get("schedules", {}):
            self.config["schedules"][task_name]["enabled"] = False
            schedule.clear(task_name)  # Remove from scheduler
            self.save_configuration()
            logger.info(f"Disabled task: {task_name}")
            return True
        return False

def main():
    """Main function to run the automation scheduler"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Repository Automation Scheduler")
    parser.add_argument("--start", action="store_true", help="Start the scheduler daemon")
    parser.add_argument("--status", action="store_true", help="Show scheduler status")
    parser.add_argument("--run-now", help="Run a specific task immediately")
    parser.add_argument("--list-tasks", action="store_true", help="List available tasks")
    parser.add_argument("--enable", help="Enable a specific task")
    parser.add_argument("--disable", help="Disable a specific task")
    parser.add_argument("--repo-path", default=".", help="Repository path (default: current directory)")
    
    args = parser.parse_args()
    
    scheduler = AutomationScheduler(args.repo_path)
    
    if args.start:
        try:
            scheduler.start_scheduler()
        except KeyboardInterrupt:
            logger.info("Scheduler interrupted by user")
    elif args.status:
        status = scheduler.get_status()
        print(json.dumps(status, indent=2))
    elif args.run_now:
        result = scheduler.run_task_now(args.run_now)
        print(json.dumps(result, indent=2))
    elif args.list_tasks:
        tasks = scheduler.list_available_tasks()
        print("Available tasks:")
        for task in tasks:
            print(f"  - {task}")
    elif args.enable:
        success = scheduler.enable_task(args.enable)
        print(f"Task {args.enable}: {'enabled' if success else 'not found'}")
    elif args.disable:
        success = scheduler.disable_task(args.disable)
        print(f"Task {args.disable}: {'disabled' if success else 'not found'}")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()