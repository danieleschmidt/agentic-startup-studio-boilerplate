"""
Unit tests for BacklogManager class
"""

import pytest
import json
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import patch, mock_open

from backlog_manager import BacklogManager


class TestBacklogManager:
    """Test cases for BacklogManager class."""

    @pytest.fixture
    def temp_backlog_file(self):
        """Create a temporary backlog file for testing."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write("""backlog:
  - id: BL001
    title: "Test item 1"
    type: feature
    description: "Test description 1"
    effort: 5
    value: 8
    time_criticality: 3
    risk_reduction: 2
    status: READY
    risk_tier: medium
    created_at: "2025-07-27T10:00:00Z"
    
  - id: BL002
    title: "Test item 2"
    type: bug
    description: "Test description 2"
    effort: 3
    value: 5
    time_criticality: 8
    risk_reduction: 1
    status: DONE
    risk_tier: low
    created_at: "2025-07-26T10:00:00Z"
    completed_at: "2025-07-27T11:00:00Z"

meta:
  last_updated: "2025-07-27T12:00:00Z"
  total_items: 2
""")
            temp_file = Path(f.name)
        
        yield temp_file
        
        # Cleanup
        temp_file.unlink(missing_ok=True)

    @pytest.fixture
    def backlog_manager(self, temp_backlog_file):
        """Create BacklogManager with temporary file."""
        return BacklogManager(str(temp_backlog_file))

    def test_load_backlog(self, backlog_manager):
        """Test loading backlog from YAML file."""
        data = backlog_manager.load_backlog()
        
        assert 'backlog' in data
        assert 'meta' in data
        assert len(data['backlog']) == 2
        assert data['backlog'][0]['id'] == 'BL001'
        assert data['backlog'][1]['id'] == 'BL002'

    def test_calculate_wsjf(self, backlog_manager):
        """Test WSJF score calculation."""
        item = {
            'value': 8,
            'time_criticality': 3,
            'risk_reduction': 2,
            'effort': 5
        }
        
        score = backlog_manager.calculate_wsjf(item)
        expected_score = (8 + 3 + 2) / 5  # 2.6
        
        assert score == 2.6

    def test_calculate_wsjf_with_aging(self, backlog_manager):
        """Test WSJF calculation with aging multiplier."""
        item = {
            'value': 8,
            'time_criticality': 3,
            'risk_reduction': 2,
            'effort': 5
        }
        
        aging_multiplier = 1.5
        score = backlog_manager.calculate_wsjf(item, aging_multiplier)
        expected_score = ((8 + 3 + 2) / 5) * 1.5  # 3.9
        
        assert score == 3.9

    def test_apply_aging_multiplier(self, backlog_manager):
        """Test aging multiplier calculation."""
        # Item created 30 days ago
        created_at = datetime.now(timezone.utc).replace(day=1)
        item = {
            'created_at': created_at.isoformat()
        }
        
        with patch('backlog_manager.datetime') as mock_datetime:
            mock_datetime.datetime.now.return_value = created_at.replace(day=31)
            mock_datetime.timezone = timezone
            
            multiplier = backlog_manager.apply_aging_multiplier(item)
            
            # Should be close to 2.0 for 30-day-old item
            assert 1.8 <= multiplier <= 2.0

    def test_get_actionable_items(self, backlog_manager):
        """Test getting actionable items (READY status)."""
        actionable = backlog_manager.get_actionable_items()
        
        assert len(actionable) == 1
        assert actionable[0]['id'] == 'BL001'
        assert actionable[0]['status'] == 'READY'
        assert 'wsjf_score' in actionable[0]

    def test_get_next_ready_item(self, backlog_manager):
        """Test getting highest priority ready item."""
        next_item = backlog_manager.get_next_ready_item()
        
        assert next_item is not None
        assert next_item['id'] == 'BL001'
        assert next_item['status'] == 'READY'

    def test_update_item_status(self, backlog_manager):
        """Test updating item status."""
        # Update status to DOING
        backlog_manager.update_item_status('BL001', 'DOING')
        
        data = backlog_manager.load_backlog()
        item = next((item for item in data['backlog'] if item['id'] == 'BL001'), None)
        
        assert item is not None
        assert item['status'] == 'DOING'
        assert 'started_at' in item

    def test_update_item_status_to_done(self, backlog_manager):
        """Test updating item status to DONE."""
        backlog_manager.update_item_status('BL001', 'DONE')
        
        data = backlog_manager.load_backlog()
        item = next((item for item in data['backlog'] if item['id'] == 'BL001'), None)
        
        assert item is not None
        assert item['status'] == 'DONE'
        assert 'completed_at' in item

    def test_update_item_status_to_blocked(self, backlog_manager):
        """Test updating item status to BLOCKED with reason."""
        blocked_reason = "Waiting for external dependency"
        backlog_manager.update_item_status('BL001', 'BLOCKED', blocked_reason=blocked_reason)
        
        data = backlog_manager.load_backlog()
        item = next((item for item in data['backlog'] if item['id'] == 'BL001'), None)
        
        assert item is not None
        assert item['status'] == 'BLOCKED'
        assert item['blocked_reason'] == blocked_reason

    def test_calculate_actual_cycle_time(self, backlog_manager):
        """Test actual cycle time calculation."""
        data = backlog_manager.load_backlog()
        cycle_time = backlog_manager.calculate_actual_cycle_time(data)
        
        # Should calculate cycle time from completed items
        assert cycle_time > 0
        assert isinstance(cycle_time, float)

    def test_calculate_actual_cycle_time_no_completed_items(self, backlog_manager):
        """Test cycle time calculation with no completed items."""
        data = {
            'backlog': [
                {
                    'id': 'BL003',
                    'status': 'READY',
                    'created_at': '2025-07-27T10:00:00Z'
                }
            ]
        }
        
        cycle_time = backlog_manager.calculate_actual_cycle_time(data)
        assert cycle_time == 0

    def test_discover_new_tasks(self, backlog_manager):
        """Test discovering new tasks."""
        # Mock missing .github/workflows directory
        with patch('pathlib.Path.exists', return_value=False):
            new_tasks = backlog_manager.discover_new_tasks()
            
            assert len(new_tasks) >= 1
            assert any('GitHub Actions' in task['title'] for task in new_tasks)

    def test_discover_new_tasks_automation_scope(self, backlog_manager):
        """Test discovering automation scope configuration task."""
        with patch('pathlib.Path.exists', side_effect=lambda path: str(path) != '.automation-scope.yaml'):
            new_tasks = backlog_manager.discover_new_tasks()
            
            assert any('automation scope' in task['title'].lower() for task in new_tasks)

    @patch('subprocess.run')
    def test_sync_repo_and_ci_clean(self, mock_run, backlog_manager):
        """Test repository sync with clean working directory."""
        # Mock git status returning clean
        mock_run.side_effect = [
            type('Result', (), {'stdout': '', 'returncode': 0})(),  # git status
            type('Result', (), {'stdout': '', 'returncode': 0})()   # git pull
        ]
        
        result = backlog_manager.sync_repo_and_ci()
        assert result is True

    @patch('subprocess.run')
    def test_sync_repo_and_ci_dirty(self, mock_run, backlog_manager):
        """Test repository sync with uncommitted changes."""
        # Mock git status returning uncommitted changes
        mock_run.return_value = type('Result', (), {'stdout': 'M file.py', 'returncode': 0})()
        
        result = backlog_manager.sync_repo_and_ci()
        assert result is False

    def test_generate_status_report(self, backlog_manager):
        """Test status report generation."""
        report_path = backlog_manager.generate_status_report()
        
        assert report_path.exists()
        assert report_path.suffix == '.json'
        
        # Check report content
        with open(report_path) as f:
            report_data = json.load(f)
        
        assert 'timestamp' in report_data
        assert 'completed_ids' in report_data
        assert 'backlog_size_by_status' in report_data
        assert 'wsjf_snapshot' in report_data

    def test_count_by_status(self, backlog_manager):
        """Test counting items by status."""
        data = backlog_manager.load_backlog()
        counts = backlog_manager._count_by_status(data)
        
        assert counts['READY'] == 1
        assert counts['DONE'] == 1
        assert counts.get('BLOCKED', 0) == 0

    def test_update_metadata(self, backlog_manager):
        """Test metadata update."""
        data = backlog_manager.load_backlog()
        backlog_manager._update_metadata(data)
        
        meta = data['meta']
        assert 'last_updated' in meta
        assert 'total_items' in meta
        assert 'done_items' in meta
        assert 'completion_rate' in meta
        assert 'avg_cycle_time' in meta

    def test_empty_backlog_handling(self):
        """Test handling of empty backlog file."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write("backlog: []\nmeta: {}")
            temp_file = Path(f.name)
        
        try:
            manager = BacklogManager(str(temp_file))
            data = manager.load_backlog()
            
            assert data['backlog'] == []
            assert isinstance(data['meta'], dict)
            
            # Should handle empty backlog gracefully
            actionable = manager.get_actionable_items()
            assert actionable == []
            
            next_item = manager.get_next_ready_item()
            assert next_item is None
            
        finally:
            temp_file.unlink(missing_ok=True)

    def test_invalid_backlog_format(self):
        """Test handling of invalid backlog format."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write("invalid yaml content: [")
            temp_file = Path(f.name)
        
        try:
            manager = BacklogManager(str(temp_file))
            
            # Should handle invalid format gracefully
            with pytest.raises(Exception):
                manager.load_backlog()
                
        finally:
            temp_file.unlink(missing_ok=True)

    @pytest.mark.parametrize("status,expected_count", [
        ("READY", 1),
        ("DONE", 1),
        ("BLOCKED", 0),
        ("NEW", 0),
    ])
    def test_status_filtering(self, backlog_manager, status, expected_count):
        """Test filtering items by status."""
        data = backlog_manager.load_backlog()
        items_with_status = [item for item in data['backlog'] if item.get('status') == status]
        
        assert len(items_with_status) == expected_count

    def test_wsjf_score_ordering(self, backlog_manager):
        """Test that actionable items are ordered by WSJF score."""
        # Add more items to test ordering
        data = backlog_manager.load_backlog()
        data['backlog'].append({
            'id': 'BL003',
            'title': 'High priority item',
            'status': 'READY',
            'value': 13,
            'time_criticality': 13,
            'risk_reduction': 8,
            'effort': 2,
            'created_at': '2025-07-27T10:00:00Z'
        })
        
        backlog_manager.save_backlog(data)
        
        actionable = backlog_manager.get_actionable_items()
        
        # Should be sorted by WSJF score descending
        if len(actionable) > 1:
            for i in range(len(actionable) - 1):
                assert actionable[i]['wsjf_score'] >= actionable[i + 1]['wsjf_score']