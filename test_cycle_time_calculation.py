#!/usr/bin/env python3
"""
Test for cycle time calculation functionality
"""

import json
import datetime
from backlog_manager import BacklogManager

def test_cycle_time_calculation():
    """Test that cycle time is calculated from actual completion data"""
    
    # Create test data with realistic timestamps
    test_data = {
        'backlog': [
            {
                'id': 'BL001',
                'title': 'Test item 1',
                'status': 'DONE',
                'created_at': '2025-07-26T05:24:00Z',
                'completed_at': '2025-07-26T05:32:00Z'  # 8 minutes
            },
            {
                'id': 'BL002',
                'title': 'Test item 2', 
                'status': 'DONE',
                'created_at': '2025-07-26T05:30:00Z',
                'completed_at': '2025-07-26T05:40:00Z'  # 10 minutes
            },
            {
                'id': 'BL003',
                'title': 'Test item 3',
                'status': 'DONE', 
                'created_at': '2025-07-26T05:24:00Z',
                'completed_at': '2025-07-26T05:30:00Z'  # 6 minutes
            },
            {
                'id': 'BL004',
                'title': 'Test item 4',
                'status': 'BLOCKED',
                'created_at': '2025-07-26T05:24:00Z'
                # No completion - should be excluded
            }
        ]
    }
    
    manager = BacklogManager()
    
    # Test calculate_actual_cycle_time method
    cycle_time = manager.calculate_actual_cycle_time(test_data)
    
    # Expected: (8 + 10 + 6) / 3 = 8.0 minutes average
    expected_avg = 8.0
    
    assert abs(cycle_time - expected_avg) < 0.1, f"Expected ~{expected_avg} minutes, got {cycle_time}"
    
    print("✓ Cycle time calculation works correctly")
    print(f"✓ Average cycle time: {cycle_time} minutes")
    
    # Test edge case: no completed items
    empty_data = {'backlog': [{'id': 'BL001', 'status': 'NEW'}]}
    empty_cycle_time = manager.calculate_actual_cycle_time(empty_data)
    assert empty_cycle_time == 0, f"Expected 0 for no completed items, got {empty_cycle_time}"
    
    print("✓ Edge case (no completed items) handled correctly")

def test_metadata_update_uses_real_cycle_time():
    """Test that _update_metadata uses calculated cycle time instead of hardcoded value"""
    
    manager = BacklogManager()
    
    # Load actual backlog data
    data = manager.load_backlog()
    
    # Update metadata (this should now calculate real cycle time)
    manager._update_metadata(data)
    
    # Check that avg_cycle_time is no longer the hardcoded string
    avg_cycle_time = data['meta']['avg_cycle_time']
    
    assert avg_cycle_time != "~8_minutes", "Cycle time should not be hardcoded anymore"
    assert isinstance(avg_cycle_time, (int, float)) or avg_cycle_time.endswith('_minutes'), \
           f"Expected numeric cycle time or formatted string, got: {avg_cycle_time}"
    
    print(f"✓ Metadata now uses calculated cycle time: {avg_cycle_time}")

if __name__ == "__main__":
    test_cycle_time_calculation()
    test_metadata_update_uses_real_cycle_time()
    print("\n🎉 All cycle time calculation tests passed!")