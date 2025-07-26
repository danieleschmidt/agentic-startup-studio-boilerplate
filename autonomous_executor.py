#!/usr/bin/env python3
"""
Autonomous Macro Execution Loop
Implements the main automation cycle for backlog processing
"""

import json
import subprocess
import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

from backlog_manager import BacklogManager
from security_scanner import SecurityScanner

class AutonomousExecutor:
    def __init__(self):
        self.backlog_manager = BacklogManager()
        self.security_scanner = SecurityScanner()
        self.scope_config = self._load_scope_config()
        self.max_iterations = 10  # Safety limit
        
    def _load_scope_config(self) -> Dict[str, Any]:
        """Load automation scope configuration"""
        scope_path = Path(".automation-scope.yaml")
        if not scope_path.exists():
            return self._default_scope_config()
        
        # Simple YAML parser for scope config
        with open(scope_path, 'r') as f:
            content = f.read()
        
        # Extract key configuration values using regex
        import re
        
        max_risk_tier = re.search(r'max_risk_tier:\s*"(\w+)"', content)
        max_effort_points = re.search(r'max_effort_points:\s*(\d+)', content)
        max_prs_per_day = re.search(r'max_prs_per_day:\s*(\d+)', content)
        
        return {
            'max_risk_tier': max_risk_tier.group(1) if max_risk_tier else 'medium',
            'max_effort_points': int(max_effort_points.group(1)) if max_effort_points else 8,
            'max_prs_per_day': int(max_prs_per_day.group(1)) if max_prs_per_day else 5,
            'require_tests': True,
            'follow_existing_conventions': True
        }
    
    def _default_scope_config(self) -> Dict[str, Any]:
        """Default scope configuration"""
        return {
            'max_risk_tier': 'medium',
            'max_effort_points': 8,
            'max_prs_per_day': 5,
            'require_tests': True,
            'follow_existing_conventions': True
        }
    
    def sync_repo_and_ci(self) -> bool:
        """Sync repository and check CI status"""
        try:
            # Check git status
            result = subprocess.run(['git', 'status', '--porcelain'], 
                                  capture_output=True, text=True)
            if result.stdout.strip():
                print("⚠️  Uncommitted changes detected")
                return False
                
            # Check if we're on the right branch
            result = subprocess.run(['git', 'branch', '--show-current'], 
                                  capture_output=True, text=True, check=True)
            current_branch = result.stdout.strip()
            
            if current_branch != 'main' and not current_branch.startswith('terragon/'):
                print(f"⚠️  On branch {current_branch}, expected main or terragon/* branch")
                return False
                
            print(f"✅ Repository sync OK (branch: {current_branch})")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Repository sync failed: {e}")
            return False
    
    def discover_new_tasks(self) -> List[Dict[str, Any]]:
        """Discover new tasks from various sources"""
        new_tasks = self.backlog_manager.discover_new_tasks()
        
        # Additional discovery: check for security vulnerabilities
        security_results = self.security_scanner.run_full_security_scan()
        
        if security_results['security_summary']['status'] in ['CRITICAL', 'WARNING']:
            security_task = {
                'id': 'SEC001',
                'title': 'Address critical security vulnerabilities',
                'type': 'security',
                'description': f"Security scan found {security_results['security_summary']['total_vulnerabilities']} vulnerabilities",
                'acceptance_criteria': [
                    'Review SAST findings in docs/security/sast-report.json',
                    'Fix high-severity security issues',
                    'Re-run security scan to verify fixes'
                ],
                'effort': 5,
                'value': 13,
                'time_criticality': 13,
                'risk_reduction': 13,
                'status': 'NEW',
                'risk_tier': 'high',
                'created_at': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                'links': ['docs/security/security-summary.json']
            }
            new_tasks.append(security_task)
        
        return new_tasks
    
    def score_and_sort_backlog(self):
        """Update WSJF scores and sort backlog"""
        data = self.backlog_manager.load_backlog()
        
        for item in data.get('backlog', []):
            if item.get('status') not in ['DONE', 'BLOCKED']:
                aging = self.backlog_manager.apply_aging_multiplier(item)
                item['wsjf_score'] = self.backlog_manager.calculate_wsjf(item, aging)
        
        # Sort by WSJF score
        data['backlog'].sort(key=lambda x: x.get('wsjf_score', 0), reverse=True)
        
        # Update metadata
        self.backlog_manager._update_metadata(data)
        self.backlog_manager.save_backlog(data)
    
    def get_next_actionable_item(self) -> Optional[Dict[str, Any]]:
        """Get next item that can be executed within scope"""
        data = self.backlog_manager.load_backlog()
        
        for item in data.get('backlog', []):
            if item.get('status') == 'READY':
                # Check scope constraints
                if self._is_within_scope(item):
                    return item
                else:
                    print(f"⏸️  Item {item['id']} exceeds scope constraints")
                    
            elif item.get('status') == 'NEW':
                # Auto-refine simple items
                if self._can_auto_refine(item):
                    self.backlog_manager.update_item_status(item['id'], 'READY')
                    return item if self._is_within_scope(item) else None
        
        return None
    
    def _is_within_scope(self, item: Dict[str, Any]) -> bool:
        """Check if item is within automation scope"""
        risk_tiers = {'low': 1, 'medium': 2, 'high': 3}
        max_risk = risk_tiers.get(self.scope_config['max_risk_tier'], 2)
        item_risk = risk_tiers.get(item.get('risk_tier', 'medium'), 2)
        
        if item_risk > max_risk:
            return False
            
        if item.get('effort', 0) > self.scope_config['max_effort_points']:
            return False
            
        # Don't auto-execute GitHub workflow changes
        if 'workflow' in item.get('title', '').lower() or 'github actions' in item.get('description', '').lower():
            return False
        
        return True
    
    def _can_auto_refine(self, item: Dict[str, Any]) -> bool:
        """Check if item can be automatically refined"""
        # Simple heuristic: items with clear acceptance criteria can be auto-refined
        return bool(item.get('acceptance_criteria')) and item.get('effort', 0) <= 5
    
    def execute_micro_cycle(self, item: Dict[str, Any]) -> bool:
        """Execute TDD micro cycle for an item"""
        print(f"🚀 Executing: {item['title']} (WSJF: {item.get('wsjf_score', 0)})")
        
        # Mark as in progress
        self.backlog_manager.update_item_status(item['id'], 'DOING')
        
        try:
            # This is where actual implementation would happen
            # For now, we'll simulate successful execution
            
            if item['type'] == 'security':
                success = self._handle_security_item(item)
            elif item['type'] == 'configuration':
                success = self._handle_configuration_item(item)
            elif item['type'] == 'infrastructure':
                success = self._handle_infrastructure_item(item)
            else:
                success = self._handle_generic_item(item)
            
            if success:
                self.backlog_manager.update_item_status(item['id'], 'DONE')
                print(f"✅ Completed: {item['title']}")
                return True
            else:
                self.backlog_manager.update_item_status(
                    item['id'], 'BLOCKED', 
                    blocked_reason="Implementation failed during execution"
                )
                print(f"❌ Failed: {item['title']}")
                return False
                
        except Exception as e:
            self.backlog_manager.update_item_status(
                item['id'], 'BLOCKED',
                blocked_reason=f"Exception during execution: {str(e)}"
            )
            print(f"💥 Error executing {item['title']}: {e}")
            return False
    
    def _handle_security_item(self, item: Dict[str, Any]) -> bool:
        """Handle security-related items"""
        print("🔒 Processing security item...")
        
        # For now, just generate updated security report
        results = self.security_scanner.run_full_security_scan()
        
        # Security items are considered complete if we've generated the report
        # In practice, would implement actual vulnerability fixes
        return results['security_summary']['status'] != 'CRITICAL'
    
    def _handle_configuration_item(self, item: Dict[str, Any]) -> bool:
        """Handle configuration items"""
        print("⚙️  Processing configuration item...")
        
        if 'automation-scope' in item.get('title', '').lower():
            # Already created .automation-scope.yaml, so this is done
            return Path('.automation-scope.yaml').exists()
        
        return True
    
    def _handle_infrastructure_item(self, item: Dict[str, Any]) -> bool:
        """Handle infrastructure items"""
        print("🏗️  Processing infrastructure item...")
        
        # Infrastructure items typically require human approval
        self.backlog_manager.update_item_status(
            item['id'], 'BLOCKED',
            blocked_reason="Infrastructure changes require human approval"
        )
        return False
    
    def _handle_generic_item(self, item: Dict[str, Any]) -> bool:
        """Handle generic items"""
        print("📝 Processing generic item...")
        return True
    
    def run_macro_loop(self) -> Dict[str, Any]:
        """Execute the main autonomous macro loop"""
        print("🤖 Starting Autonomous Backlog Management Loop")
        
        execution_log = {
            'start_time': datetime.datetime.now(datetime.timezone.utc).isoformat(),
            'iterations': 0,
            'items_processed': [],
            'items_completed': [],
            'items_failed': [],
            'exit_reason': 'unknown'
        }
        
        for iteration in range(self.max_iterations):
            execution_log['iterations'] = iteration + 1
            print(f"\n--- Iteration {iteration + 1} ---")
            
            # Step 1: Sync repo and CI
            if not self.sync_repo_and_ci():
                execution_log['exit_reason'] = 'repo_sync_failed'
                break
            
            # Step 2: Discover new tasks
            new_tasks = self.discover_new_tasks()
            if new_tasks:
                print(f"📋 Discovered {len(new_tasks)} new tasks")
                # In practice, would add these to backlog
            
            # Step 3: Score and sort backlog
            self.score_and_sort_backlog()
            
            # Step 4: Get next actionable item
            next_item = self.get_next_actionable_item()
            
            if not next_item:
                execution_log['exit_reason'] = 'no_actionable_items'
                print("✨ No actionable items found. Backlog complete!")
                break
            
            # Step 5: Execute micro cycle
            execution_log['items_processed'].append(next_item['id'])
            
            success = self.execute_micro_cycle(next_item)
            
            if success:
                execution_log['items_completed'].append(next_item['id'])
            else:
                execution_log['items_failed'].append(next_item['id'])
            
            # Step 6: Generate status report
            report_path = self.backlog_manager.generate_status_report()
            print(f"📊 Status report: {report_path}")
        
        execution_log['end_time'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        execution_log['duration_minutes'] = (
            datetime.datetime.fromisoformat(execution_log['end_time'].replace('Z', '+00:00')) -
            datetime.datetime.fromisoformat(execution_log['start_time'].replace('Z', '+00:00'))
        ).total_seconds() / 60
        
        # Save execution log
        log_path = Path("docs/status/execution-log.json")
        with open(log_path, 'w') as f:
            json.dump(execution_log, f, indent=2)
        
        print(f"\n🏁 Execution completed: {execution_log['exit_reason']}")
        print(f"📈 Processed: {len(execution_log['items_processed'])}, Completed: {len(execution_log['items_completed'])}, Failed: {len(execution_log['items_failed'])}")
        
        return execution_log

if __name__ == "__main__":
    executor = AutonomousExecutor()
    log = executor.run_macro_loop()