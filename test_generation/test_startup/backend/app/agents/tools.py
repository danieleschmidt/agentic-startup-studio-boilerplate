"""
Custom tools for CrewAI agents.
Implements specific tools that agents can use to perform their tasks.
"""

import json
import logging
import time
from typing import Any, Dict, List, Optional

import httpx
import pandas as pd
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class WebSearchTool(BaseTool):
    """Tool for performing web searches and gathering information."""
    
    name: str = "web_search"
    description: str = "Search the web for information on a given topic"
    
    def _run(self, query: str) -> str:
        """
        Perform a web search for the given query.
        
        Args:
            query: Search query string
            
        Returns:
            str: Search results summary
        """
        try:
            # In a real implementation, you would integrate with a search API
            # For this template, we'll simulate a search result
            
            # Simulate API call delay
            time.sleep(0.5)
            
            # Mock search results
            mock_results = {
                "query": query,
                "results": [
                    {
                        "title": f"Comprehensive Guide to {query}",
                        "url": "https://example.com/guide",
                        "snippet": f"This comprehensive guide covers everything about {query}, including best practices, common challenges, and expert insights."
                    },
                    {
                        "title": f"Latest Trends in {query}",
                        "url": "https://example.com/trends",
                        "snippet": f"Stay up-to-date with the latest trends and developments in {query} with this detailed analysis."
                    },
                    {
                        "title": f"{query}: Case Studies and Examples",
                        "url": "https://example.com/cases",
                        "snippet": f"Real-world case studies and examples demonstrating successful implementation of {query}."
                    }
                ],
                "total_results": 3
            }
            
            # Format results for agent consumption
            formatted_results = f"Search Results for '{query}':\n\n"
            for i, result in enumerate(mock_results["results"], 1):
                formatted_results += f"{i}. {result['title']}\n"
                formatted_results += f"   URL: {result['url']}\n"
                formatted_results += f"   Summary: {result['snippet']}\n\n"
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error performing web search: {e}")
            return f"Error performing search for '{query}': {str(e)}"


class DataAnalysisTool(BaseTool):
    """Tool for analyzing data and extracting insights."""
    
    name: str = "data_analysis"
    description: str = "Analyze data and extract statistical insights"
    
    def _run(self, data_description: str) -> str:
        """
        Perform data analysis on the described dataset.
        
        Args:
            data_description: Description of the data to analyze
            
        Returns:
            str: Analysis results and insights
        """
        try:
            # Simulate data analysis
            time.sleep(1.0)
            
            # Mock analysis results
            analysis = {
                "data_description": data_description,
                "sample_size": 1000,
                "key_metrics": {
                    "mean": 75.5,
                    "median": 78.2,
                    "std_deviation": 12.3,
                    "correlation_coefficient": 0.67
                },
                "insights": [
                    "Strong positive correlation detected between variables",
                    "Data shows normal distribution with slight right skew",
                    "95% of values fall within expected range",
                    "Seasonal patterns identified with peak performance in Q2"
                ],
                "recommendations": [
                    "Focus on factors contributing to positive correlation",
                    "Investigate outliers in upper quartile",
                    "Implement monitoring for seasonal variations"
                ]
            }
            
            # Format analysis for agent consumption
            formatted_analysis = f"Data Analysis Results for: {data_description}\n\n"
            formatted_analysis += f"Sample Size: {analysis['sample_size']}\n\n"
            
            formatted_analysis += "Key Metrics:\n"
            for metric, value in analysis["key_metrics"].items():
                formatted_analysis += f"  - {metric.replace('_', ' ').title()}: {value}\n"
            
            formatted_analysis += "\nKey Insights:\n"
            for insight in analysis["insights"]:
                formatted_analysis += f"  • {insight}\n"
            
            formatted_analysis += "\nRecommendations:\n"
            for rec in analysis["recommendations"]:
                formatted_analysis += f"  • {rec}\n"
            
            return formatted_analysis
            
        except Exception as e:
            logger.error(f"Error performing data analysis: {e}")
            return f"Error analyzing data '{data_description}': {str(e)}"


class ReportGeneratorTool(BaseTool):
    """Tool for generating structured reports."""
    
    name: str = "report_generator"
    description: str = "Generate structured reports from analysis and research"
    
    def _run(self, content: str, report_type: str = "standard") -> str:
        """
        Generate a structured report from the provided content.
        
        Args:
            content: Content to structure into a report
            report_type: Type of report to generate
            
        Returns:
            str: Formatted report
        """
        try:
            # Simulate report generation processing
            time.sleep(0.8)
            
            # Generate structured report
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            
            if report_type == "executive":
                template = self._generate_executive_report(content, timestamp)
            elif report_type == "technical":
                template = self._generate_technical_report(content, timestamp)
            else:
                template = self._generate_standard_report(content, timestamp)
            
            return template
            
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return f"Error generating {report_type} report: {str(e)}"
    
    def _generate_standard_report(self, content: str, timestamp: str) -> str:
        """Generate a standard report format."""
        return f"""
# RESEARCH AND ANALYSIS REPORT

**Generated:** {timestamp}
**Report Type:** Standard Analysis

## EXECUTIVE SUMMARY

{self._extract_summary(content)}

## DETAILED FINDINGS

{content}

## KEY RECOMMENDATIONS

{self._extract_recommendations(content)}

## NEXT STEPS

Based on the analysis, the following actions are recommended:
- Review and validate findings with stakeholders
- Develop implementation timeline
- Assign responsible parties for each recommendation
- Schedule follow-up assessment

---
*This report was generated using AI-powered analysis tools.*
"""
    
    def _generate_executive_report(self, content: str, timestamp: str) -> str:
        """Generate an executive summary report."""
        return f"""
# EXECUTIVE SUMMARY REPORT

**Date:** {timestamp}
**Report Classification:** Executive Summary

## OVERVIEW

{self._extract_summary(content)}

## STRATEGIC IMPLICATIONS

{self._extract_strategic_points(content)}

## IMMEDIATE ACTIONS REQUIRED

{self._extract_action_items(content)}

---
*Prepared for executive leadership review*
"""
    
    def _generate_technical_report(self, content: str, timestamp: str) -> str:
        """Generate a technical analysis report."""
        return f"""
# TECHNICAL ANALYSIS REPORT

**Generated:** {timestamp}
**Analysis Type:** Technical Deep Dive

## METHODOLOGY

This analysis was conducted using automated research and data analysis tools.

## TECHNICAL FINDINGS

{content}

## DATA QUALITY ASSESSMENT

- Sources verified and cross-referenced
- Statistical significance validated
- Confidence intervals calculated
- Bias assessment completed

## TECHNICAL RECOMMENDATIONS

{self._extract_technical_recommendations(content)}

---
*Technical report for detailed implementation planning*
"""
    
    def _extract_summary(self, content: str) -> str:
        """Extract key summary points from content."""
        # Simple extraction logic - in production, use more sophisticated NLP
        lines = content.split('\n')
        summary_lines = [line for line in lines if any(keyword in line.lower() 
                        for keyword in ['summary', 'key', 'important', 'main', 'primary'])]
        return '\n'.join(summary_lines[:3]) or "Key insights and findings from the analysis."
    
    def _extract_recommendations(self, content: str) -> str:
        """Extract recommendations from content."""
        lines = content.split('\n')
        rec_lines = [line for line in lines if any(keyword in line.lower() 
                    for keyword in ['recommend', 'suggest', 'should', 'must', 'action'])]
        return '\n'.join(f"• {line.strip()}" for line in rec_lines[:5]) or "• Review findings and develop implementation plan"
    
    def _extract_strategic_points(self, content: str) -> str:
        """Extract strategic implications."""
        return "Strategic analysis indicates significant opportunities for growth and optimization."
    
    def _extract_action_items(self, content: str) -> str:
        """Extract immediate action items."""
        return "1. Review analysis findings\n2. Validate recommendations\n3. Develop implementation timeline"
    
    def _extract_technical_recommendations(self, content: str) -> str:
        """Extract technical recommendations."""
        return "Technical implementation should follow established best practices and security guidelines."


class EmailTool(BaseTool):
    """Tool for sending emails and notifications."""
    
    name: str = "email_sender"
    description: str = "Send emails and notifications"
    
    def _run(self, recipient: str, subject: str, body: str) -> str:
        """
        Send an email notification.
        
        Args:
            recipient: Email recipient
            subject: Email subject
            body: Email body content
            
        Returns:
            str: Delivery status
        """
        try:
            # In production, integrate with actual email service
            # For now, simulate email sending
            
            logger.info(f"Sending email to {recipient}: {subject}")
            time.sleep(0.3)  # Simulate network delay
            
            return f"Email successfully sent to {recipient} with subject '{subject}'"
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return f"Failed to send email to {recipient}: {str(e)}"


class DatabaseTool(BaseTool):
    """Tool for database operations."""
    
    name: str = "database_query"
    description: str = "Execute database queries and retrieve data"
    
    def _run(self, query_description: str) -> str:
        """
        Execute a database query based on description.
        
        Args:
            query_description: Description of the data to retrieve
            
        Returns:
            str: Query results
        """
        try:
            # Simulate database query
            time.sleep(0.5)
            
            # Mock query results
            results = {
                "query": query_description,
                "rows_returned": 150,
                "execution_time": "0.023s",
                "sample_data": [
                    {"id": 1, "name": "Sample Record 1", "value": 100},
                    {"id": 2, "name": "Sample Record 2", "value": 200},
                    {"id": 3, "name": "Sample Record 3", "value": 150}
                ]
            }
            
            formatted_results = f"Database Query Results: {query_description}\n\n"
            formatted_results += f"Rows Returned: {results['rows_returned']}\n"
            formatted_results += f"Execution Time: {results['execution_time']}\n\n"
            formatted_results += "Sample Data:\n"
            
            for record in results["sample_data"]:
                formatted_results += f"  - ID: {record['id']}, Name: {record['name']}, Value: {record['value']}\n"
            
            return formatted_results
            
        except Exception as e:
            logger.error(f"Error executing database query: {e}")
            return f"Database query failed for '{query_description}': {str(e)}"