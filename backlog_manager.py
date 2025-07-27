#!/usr/bin/env python3
"""
Autonomous Backlog Management System
Implements WSJF prioritization and macro execution loop
"""

import json
# Note: PyYAML not available, using JSON for now
import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path
import subprocess
import os

class BacklogManager:
    def __init__(self, backlog_path: str = "backlog.yml"):
        self.backlog_path = Path(backlog_path)
        self.status_dir = Path("docs/status")
        self.status_dir.mkdir(parents=True, exist_ok=True)
        
    def load_backlog(self) -> Dict[str, Any]:
        """Load and parse backlog.yml - simplified YAML parser"""
        # Simple YAML to dict parser for basic backlog.yml structure
        with open(self.backlog_path, 'r') as f:
            content = f.read()
        
        # Parse the basic YAML structure manually
        import re
        
        # Extract backlog items
        backlog_items = []
        item_blocks = re.findall(r'  - id: (BL\d+)\s+title: "(.*?)"\s+type: (\w+)\s+description: "(.*?)"\s+acceptance_criteria:(.*?)effort: (\d+)\s+value: (\d+)\s+time_criticality: (\d+)\s+risk_reduction: (\d+)\s+status: (\w+)\s+risk_tier: (\w+)\s+created_at: "(.*?)"(?:\s+completed_at: "(.*?)")?(?:\s+blocked_reason: "(.*?)")?(?:\s+wsjf_score: ([\d.]+))?', content, re.DOTALL)
        
        for match in item_blocks:
            item = {
                'id': match[0],
                'title': match[1],
                'type': match[2],
                'description': match[3],
                'effort': int(match[5]),
                'value': int(match[6]),
                'time_criticality': int(match[7]),
                'risk_reduction': int(match[8]),
                'status': match[9],
                'risk_tier': match[10],
                'created_at': match[11],
                'links': []
            }
            if match[12]:  # completed_at
                item['completed_at'] = match[12]
            if match[13]:  # blocked_reason
                item['blocked_reason'] = match[13]
            if match[14]:  # wsjf_score
                item['wsjf_score'] = float(match[14])
            
            backlog_items.append(item)
        
        return {
            'backlog': backlog_items,
            'meta': {
                'last_updated': '2025-07-26T05:42:00Z',
                'total_items': len(backlog_items)
            }
        }
            
    def save_backlog(self, data: Dict[str, Any]):
        """Save backlog.yml with updated metadata"""
        # For now, we'll just save as JSON to preserve the data
        # In production, would use proper YAML library
        data['meta']['last_updated'] = datetime.datetime.utcnow().isoformat() + "Z"
        json_path = str(self.backlog_path).replace('.yml', '.json')
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def calculate_wsjf(self, item: Dict[str, Any], aging_multiplier: float = 1.0) -> float:
        """Calculate WSJF score: (value + time_criticality + risk_reduction) / effort"""
        value = item.get('value', 1)
        time_criticality = item.get('time_criticality', 1) 
        risk_reduction = item.get('risk_reduction', 1)
        effort = item.get('effort', 1)
        
        score = (value + time_criticality + risk_reduction) / effort
        return round(score * aging_multiplier, 2)
    
    def apply_aging_multiplier(self, item: Dict[str, Any], max_multiplier: float = 2.0) -> float:
        """Apply aging multiplier based on item age"""
        created_at = datetime.datetime.fromisoformat(item['created_at'].replace('Z', '+00:00'))
        age_days = (datetime.datetime.now(datetime.timezone.utc) - created_at).days
        
        # Apply linear aging: 1.0 + (age_days / 30) * max_multiplier, capped at max_multiplier
        multiplier = min(1.0 + (age_days / 30) * (max_multiplier - 1.0), max_multiplier)
        return multiplier
    
    def get_actionable_items(self) -> List[Dict[str, Any]]:
        """Get items that can be worked on (status: READY)"""
        data = self.load_backlog()
        actionable = []
        
        for item in data.get('backlog', []):
            if item.get('status') == 'READY':
                # Recalculate WSJF with aging
                aging = self.apply_aging_multiplier(item)
                item['wsjf_score'] = self.calculate_wsjf(item, aging)
                actionable.append(item)
                
        # Sort by WSJF score descending
        return sorted(actionable, key=lambda x: x['wsjf_score'], reverse=True)
    
    def get_next_ready_item(self) -> Optional[Dict[str, Any]]:
        """Get highest priority actionable item"""
        actionable = self.get_actionable_items()
        return actionable[0] if actionable else None
    
    def update_item_status(self, item_id: str, status: str, **kwargs):
        """Update item status and metadata"""
        data = self.load_backlog()
        
        for item in data.get('backlog', []):
            if item['id'] == item_id:
                item['status'] = status
                
                if status == 'DOING':
                    item['started_at'] = datetime.datetime.utcnow().isoformat() + "Z"
                elif status == 'DONE':
                    item['completed_at'] = datetime.datetime.utcnow().isoformat() + "Z"
                elif status == 'BLOCKED':
                    if 'blocked_reason' in kwargs:
                        item['blocked_reason'] = kwargs['blocked_reason']
                
                # Update other fields
                for key, value in kwargs.items():
                    if key != 'blocked_reason':
                        item[key] = value
                break
        
        # Update metadata
        self._update_metadata(data)
        self.save_backlog(data)
    
    def calculate_actual_cycle_time(self, data: Dict[str, Any]) -> float:
        """Calculate actual average cycle time from completed items"""
        completed_items = [item for item in data.get('backlog', []) 
                          if item.get('status') == 'DONE' and 
                             item.get('created_at') and 
                             item.get('completed_at')]
        
        if not completed_items:
            return 0
        
        total_minutes = 0
        for item in completed_items:
            created = datetime.datetime.fromisoformat(item['created_at'].replace('Z', '+00:00'))
            completed = datetime.datetime.fromisoformat(item['completed_at'].replace('Z', '+00:00'))
            cycle_time_minutes = (completed - created).total_seconds() / 60
            total_minutes += cycle_time_minutes
        
        return round(total_minutes / len(completed_items), 1)

    def _update_metadata(self, data: Dict[str, Any]):
        """Update backlog metadata"""
        items = data.get('backlog', [])
        total_items = len(items)
        done_items = len([i for i in items if i.get('status') == 'DONE'])
        blocked_items = len([i for i in items if i.get('status') == 'BLOCKED'])
        ready_items = len([i for i in items if i.get('status') == 'READY'])
        
        # Calculate actual cycle time
        actual_cycle_time = self.calculate_actual_cycle_time(data)
        cycle_time_display = f"{actual_cycle_time}_minutes" if actual_cycle_time > 0 else "no_completed_items"
        
        data['meta'] = {
            'last_updated': datetime.datetime.utcnow().isoformat() + "Z",
            'total_items': total_items,
            'done_items': done_items,
            'blocked_items': blocked_items,
            'ready_items': ready_items,
            'completion_rate': round(done_items / total_items, 2) if total_items > 0 else 0,
            'avg_cycle_time': cycle_time_display,
            'wsjf_weights': {
                'aging_multiplier_max': 2.0,
                'score_scale': [1, 2, 3, 5, 8, 13]
            },
            'status_flow': ["NEW", "REFINED", "READY", "DOING", "PR", "DONE", "BLOCKED"]
        }
    
    def discover_new_tasks(self) -> List[Dict[str, Any]]:
        """Discover new tasks from various sources"""
        discovered = []
        
        # Check for missing CI workflows
        if not Path('.github/workflows').exists():
            discovered.append({
                'id': f'BL{len(self.load_backlog().get("backlog", [])) + 1:03d}',
                'title': 'Set up basic GitHub Actions workflows',
                'type': 'infrastructure',
                'description': 'Missing CI/CD workflows for automated testing and deployment',
                'acceptance_criteria': [
                    'Create .github/workflows/ci.yml for testing',
                    'Add dependabot.yml for dependency updates',
                    'Include basic security scanning'
                ],
                'effort': 5,
                'value': 8,
                'time_criticality': 5,
                'risk_reduction': 6,
                'status': 'NEW',
                'risk_tier': 'medium',
                'created_at': datetime.datetime.utcnow().isoformat() + "Z",
                'links': []
            })
        
        # Check for missing automation scope
        if not Path('.automation-scope.yaml').exists():
            discovered.append({
                'id': f'BL{len(self.load_backlog().get("backlog", [])) + len(discovered) + 1:03d}',
                'title': 'Create automation scope configuration',
                'type': 'configuration',
                'description': 'Define scope boundaries for autonomous operations',
                'acceptance_criteria': [
                    'Create .automation-scope.yaml',
                    'Define permitted external operations',
                    'Set safety boundaries for automation'
                ],
                'effort': 2,
                'value': 3,
                'time_criticality': 4,
                'risk_reduction': 8,
                'status': 'NEW',
                'risk_tier': 'high',
                'created_at': datetime.datetime.utcnow().isoformat() + "Z",
                'links': []
            })
        
        return discovered
    
    def sync_repo_and_ci(self) -> bool:
        """Sync repository and check CI status"""
        try:
            # Check git status
            result = subprocess.run(['git', 'status', '--porcelain'], 
                                  capture_output=True, text=True)
            if result.stdout.strip():
                return False  # Uncommitted changes
                
            # Pull latest changes
            subprocess.run(['git', 'pull', '--rebase'], check=True)
            return True
        except subprocess.CalledProcessError:
            return False
    
    def generate_status_report(self) -> str:
        """Generate comprehensive status report"""
        data = self.load_backlog()
        timestamp = datetime.datetime.utcnow().isoformat() + "Z"
        date_str = datetime.datetime.utcnow().strftime("%Y-%m-%d")
        
        # Generate JSON metrics
        metrics = {
            "timestamp": timestamp,
            "completed_ids": [item['id'] for item in data.get('backlog', []) 
                            if item.get('status') == 'DONE'],
            "coverage_delta": "N/A",  # Would need test coverage tracking
            "flaky_tests": [],
            "ci_summary": "No CI configured",
            "open_prs": 0,  # Would need GitHub API
            "risks_or_blocks": [item.get('blocked_reason', 'Unknown') 
                               for item in data.get('backlog', []) 
                               if item.get('status') == 'BLOCKED'],
            "backlog_size_by_status": self._count_by_status(data),
            "avg_cycle_time": data.get('meta', {}).get('avg_cycle_time', 'unknown'),
            "dora": {
                "deploy_freq": "0_per_day",
                "lead_time": "unknown",
                "change_fail_rate": "0%",
                "mttr": "unknown"
            },
            "rerere_auto_resolved_total": 0,
            "merge_driver_hits_total": 0,
            "ci_failure_rate": "0%",
            "pr_backoff_state": "inactive",
            "wsjf_snapshot": [item['title'] for item in self.get_actionable_items()[:3]]
        }
        
        # Save JSON report
        json_path = self.status_dir / f"{date_str}.json"
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        return json_path
    
    def _count_by_status(self, data: Dict[str, Any]) -> Dict[str, int]:
        """Count items by status"""
        counts = {}
        for item in data.get('backlog', []):
            status = item.get('status', 'UNKNOWN')
            counts[status] = counts.get(status, 0) + 1
        return counts

if __name__ == "__main__":
    manager = BacklogManager()
    
    # Discover new tasks
    new_tasks = manager.discover_new_tasks()
    if new_tasks:
        print(f"Discovered {len(new_tasks)} new tasks")
        
    # Check for actionable items
    next_item = manager.get_next_ready_item()
    if next_item:
        print(f"Next actionable item: {next_item['title']} (WSJF: {next_item['wsjf_score']})")
    else:
        print("No actionable items in backlog")
    
    # Generate status report
    report_path = manager.generate_status_report()
    print(f"Status report generated: {report_path}")