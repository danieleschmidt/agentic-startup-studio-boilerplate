"""
Tests for agent functionality.
"""

import pytest
from unittest.mock import Mock, patch

from app.agents.tools import WebSearchTool, DataAnalysisTool, EmailTool, DatabaseTool


def test_web_search_tool():
    """Test web search tool."""
    tool = WebSearchTool()
    
    # Mock the HTTP request to avoid external dependencies
    with patch('httpx.Client') as mock_client:
        mock_response = Mock()
        mock_response.text = '''
        <div class="result">
            <a class="result__a" href="https://example.com">Test Result</a>
            <a class="result__snippet">Test snippet about the query</a>
        </div>
        '''
        mock_response.raise_for_status.return_value = None
        mock_client.return_value.__enter__.return_value.get.return_value = mock_response
        
        result = tool._run("test query")
        
        assert "Web Search Results" in result
        assert "test query" in result
        assert "Test Result" in result or "Information about test query" in result


def test_data_analysis_tool():
    """Test data analysis tool."""
    tool = DataAnalysisTool()
    
    result = tool._run("test dataset")
    
    assert "Data Analysis Results" in result
    assert "test dataset" in result
    assert "Sample Size" in result
    assert "Key Metrics" in result
    assert "Key Insights" in result
    assert "Recommendations" in result


def test_email_tool():
    """Test email tool."""
    tool = EmailTool()
    
    with patch('app.agents.tools.get_settings') as mock_settings:
        # Mock settings for console backend
        mock_settings.return_value.email_backend = "console"
        
        result = tool._run("test@example.com", "Test Subject", "Test body")
        
        assert "Email logged to console" in result
        assert "test@example.com" in result
        assert "Test Subject" in result


def test_database_tool():
    """Test database tool."""
    tool = DatabaseTool()
    
    with patch('app.agents.tools.get_session') as mock_session:
        # Mock database session and query results
        mock_db = Mock()
        mock_session.return_value = mock_db
        mock_db.query.return_value.scalar.return_value = 5
        mock_db.query.return_value.filter.return_value.scalar.return_value = 3
        
        result = tool._run("user statistics")
        
        assert "Database Query Results" in result
        assert "user statistics" in result
        assert "Execution Time" in result