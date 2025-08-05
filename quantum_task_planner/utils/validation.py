"""
Quantum Task Planner Validation

Comprehensive validation utilities for quantum task planning operations
with security, data integrity, and quantum state validation.
"""

import re
from typing import Any, Dict, List, Optional, Union, Callable
from datetime import datetime, timedelta
from dataclasses import dataclass
import numpy as np
from pydantic import BaseModel, validator

from .exceptions import TaskValidationError, QuantumCoherenceError, ConfigurationError


@dataclass
class ValidationRule:
    """Validation rule definition"""
    name: str
    validator_func: Callable[[Any], bool]
    error_message: str
    is_critical: bool = True


class QuantumValidator:
    """Quantum-specific validation utilities"""
    
    @staticmethod
    def validate_quantum_coherence(coherence: float, task_id: str = None) -> bool:
        """Validate quantum coherence value"""
        if not isinstance(coherence, (int, float)):
            raise TaskValidationError("coherence", coherence, "Must be a number")
        
        if not 0.0 <= coherence <= 1.0:
            raise TaskValidationError("coherence", coherence, "Must be between 0.0 and 1.0")
        
        # Check for coherence loss threshold
        if coherence < 0.1 and task_id:
            raise QuantumCoherenceError(task_id, coherence, 0.1)
        
        return True
    
    @staticmethod
    def validate_probability_amplitude(amplitude: complex) -> bool:
        """Validate quantum probability amplitude"""
        if not isinstance(amplitude, complex):
            raise TaskValidationError("amplitude", amplitude, "Must be a complex number")
        
        magnitude = abs(amplitude)
        if magnitude > 1.0:
            raise TaskValidationError("amplitude", amplitude, f"Magnitude {magnitude:.3f} exceeds 1.0")
        
        return True
    
    @staticmethod
    def validate_state_probabilities(probabilities: Dict[str, float]) -> bool:
        """Validate quantum state probability distribution"""
        if not isinstance(probabilities, dict):
            raise TaskValidationError("probabilities", probabilities, "Must be a dictionary")
        
        total_probability = sum(probabilities.values())
        if not 0.99 <= total_probability <= 1.01:  # Allow small numerical errors
            raise TaskValidationError(
                "probabilities", 
                probabilities, 
                f"Total probability {total_probability:.3f} must sum to 1.0"
            )
        
        for state, prob in probabilities.items():
            if not 0.0 <= prob <= 1.0:
                raise TaskValidationError(
                    "probability", 
                    prob, 
                    f"Probability for state {state} must be between 0.0 and 1.0"
                )
        
        return True
    
    @staticmethod
    def validate_entanglement_strength(strength: float) -> bool:
        """Validate entanglement strength"""
        if not isinstance(strength, (int, float)):
            raise TaskValidationError("strength", strength, "Must be a number")
        
        if not 0.0 <= strength <= 1.0:
            raise TaskValidationError("strength", strength, "Must be between 0.0 and 1.0")
        
        return True


class TaskValidator:
    """Task-specific validation utilities"""
    
    # Validation rules
    TITLE_MAX_LENGTH = 200
    DESCRIPTION_MAX_LENGTH = 5000
    TAG_MAX_LENGTH = 50
    MAX_TAGS = 20
    
    @staticmethod
    def validate_task_id(task_id: str) -> bool:
        """Validate task ID format"""
        if not isinstance(task_id, str):
            raise TaskValidationError("task_id", task_id, "Must be a string")
        
        if not task_id.strip():
            raise TaskValidationError("task_id", task_id, "Cannot be empty")
        
        # UUID v4 format validation
        uuid_pattern = r'^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$'
        if not re.match(uuid_pattern, task_id, re.IGNORECASE):
            raise TaskValidationError("task_id", task_id, "Must be a valid UUID v4")
        
        return True
    
    @staticmethod
    def validate_title(title: str) -> bool:
        """Validate task title"""
        if not isinstance(title, str):
            raise TaskValidationError("title", title, "Must be a string")
        
        title = title.strip()
        if not title:
            raise TaskValidationError("title", title, "Cannot be empty")
        
        if len(title) > TaskValidator.TITLE_MAX_LENGTH:
            raise TaskValidationError(
                "title", 
                title, 
                f"Cannot exceed {TaskValidator.TITLE_MAX_LENGTH} characters"
            )
        
        # Check for malicious content
        if TaskValidator._contains_malicious_content(title):
            raise TaskValidationError("title", title, "Contains potentially malicious content")
        
        return True
    
    @staticmethod
    def validate_description(description: str) -> bool:
        """Validate task description"""
        if not isinstance(description, str):
            raise TaskValidationError("description", description, "Must be a string")
        
        if len(description) > TaskValidator.DESCRIPTION_MAX_LENGTH:
            raise TaskValidationError(
                "description", 
                description, 
                f"Cannot exceed {TaskValidator.DESCRIPTION_MAX_LENGTH} characters"
            )
        
        if TaskValidator._contains_malicious_content(description):
            raise TaskValidationError("description", description, "Contains potentially malicious content")
        
        return True
    
    @staticmethod
    def validate_priority(priority: str) -> bool:
        """Validate task priority"""
        valid_priorities = ['critical', 'high', 'medium', 'low', 'minimal']
        
        if not isinstance(priority, str):
            raise TaskValidationError("priority", priority, "Must be a string")
        
        if priority.lower() not in valid_priorities:
            raise TaskValidationError(
                "priority", 
                priority, 
                f"Must be one of: {', '.join(valid_priorities)}"
            )
        
        return True
    
    @staticmethod
    def validate_duration(duration: timedelta) -> bool:
        """Validate task duration"""
        if not isinstance(duration, timedelta):
            raise TaskValidationError("duration", duration, "Must be a timedelta object")
        
        if duration.total_seconds() <= 0:
            raise TaskValidationError("duration", duration, "Must be positive")
        
        # Maximum duration: 1 year
        max_duration = timedelta(days=365)
        if duration > max_duration:
            raise TaskValidationError("duration", duration, "Cannot exceed 1 year")
        
        return True
    
    @staticmethod
    def validate_due_date(due_date: datetime, created_at: datetime = None) -> bool:
        """Validate task due date"""
        if not isinstance(due_date, datetime):
            raise TaskValidationError("due_date", due_date, "Must be a datetime object")
        
        # Due date cannot be in the past (with 1 minute tolerance)
        now = datetime.utcnow()
        if due_date < now - timedelta(minutes=1):
            raise TaskValidationError("due_date", due_date, "Cannot be in the past")
        
        # Due date should be reasonable (within 10 years)
        max_future = now + timedelta(days=3650)  # 10 years
        if due_date > max_future:
            raise TaskValidationError("due_date", due_date, "Cannot be more than 10 years in the future")
        
        # If created_at provided, due_date should be after creation
        if created_at and due_date <= created_at:
            raise TaskValidationError("due_date", due_date, "Must be after creation date")
        
        return True
    
    @staticmethod
    def validate_tags(tags: List[str]) -> bool:
        """Validate task tags"""
        if not isinstance(tags, list):
            raise TaskValidationError("tags", tags, "Must be a list")
        
        if len(tags) > TaskValidator.MAX_TAGS:
            raise TaskValidationError("tags", tags, f"Cannot have more than {TaskValidator.MAX_TAGS} tags")
        
        for tag in tags:
            if not isinstance(tag, str):
                raise TaskValidationError("tag", tag, "Each tag must be a string")
            
            tag = tag.strip()
            if not tag:
                raise TaskValidationError("tag", tag, "Tags cannot be empty")
            
            if len(tag) > TaskValidator.TAG_MAX_LENGTH:
                raise TaskValidationError("tag", tag, f"Tags cannot exceed {TaskValidator.TAG_MAX_LENGTH} characters")
            
            # Tag format validation (alphanumeric, dash, underscore)
            if not re.match(r'^[a-zA-Z0-9_-]+$', tag):
                raise TaskValidationError("tag", tag, "Tags can only contain letters, numbers, dashes, and underscores")
        
        # Check for duplicates
        if len(tags) != len(set(tags)):
            raise TaskValidationError("tags", tags, "Cannot contain duplicate tags")
        
        return True
    
    @staticmethod
    def validate_complexity_factor(complexity: float) -> bool:
        """Validate task complexity factor"""
        if not isinstance(complexity, (int, float)):
            raise TaskValidationError("complexity_factor", complexity, "Must be a number")
        
        if not 0.1 <= complexity <= 10.0:
            raise TaskValidationError("complexity_factor", complexity, "Must be between 0.1 and 10.0")
        
        return True
    
    @staticmethod
    def _contains_malicious_content(text: str) -> bool:
        """Check for potentially malicious content"""
        # Simple malicious pattern detection
        malicious_patterns = [
            r'<script[^>]*>',  # Script tags
            r'javascript:',     # JavaScript protocol
            r'on\w+\s*=',      # Event handlers
            r'eval\s*\(',      # eval function
            r'exec\s*\(',      # exec function
            r'system\s*\(',    # system calls
            r'\$\([^)]*\)',    # Command substitution
            r'`[^`]*`',        # Backticks
            r'\.\./',          # Directory traversal
            r'--',             # SQL comment
            r';.*drop\s+table', # SQL injection
        ]
        
        text_lower = text.lower()
        for pattern in malicious_patterns:
            if re.search(pattern, text_lower, re.IGNORECASE):
                return True
        
        return False


class APIValidator:
    """API request/response validation"""
    
    @staticmethod
    def validate_pagination(page: int, limit: int) -> bool:
        """Validate pagination parameters"""
        if not isinstance(page, int) or page < 1:
            raise TaskValidationError("page", page, "Must be a positive integer")
        
        if not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise TaskValidationError("limit", limit, "Must be between 1 and 1000")
        
        return True
    
    @staticmethod
    def validate_filter_params(filters: Dict[str, Any]) -> bool:
        """Validate filter parameters"""
        if not isinstance(filters, dict):
            raise TaskValidationError("filters", filters, "Must be a dictionary")
        
        allowed_filters = {
            'priority', 'state', 'tags', 'assignee', 'created_after', 
            'created_before', 'due_after', 'due_before', 'complexity_min', 
            'complexity_max', 'coherence_min', 'coherence_max'
        }
        
        for key in filters.keys():
            if key not in allowed_filters:
                raise TaskValidationError("filter", key, f"Invalid filter. Allowed: {allowed_filters}")
        
        return True
    
    @staticmethod
    def validate_sort_params(sort_by: str, sort_order: str) -> bool:
        """Validate sorting parameters"""
        allowed_sort_fields = {
            'created_at', 'due_date', 'priority', 'title', 'complexity_factor',
            'quantum_coherence', 'completion_probability'
        }
        
        if sort_by not in allowed_sort_fields:
            raise TaskValidationError("sort_by", sort_by, f"Invalid sort field. Allowed: {allowed_sort_fields}")
        
        if sort_order.lower() not in ['asc', 'desc']:
            raise TaskValidationError("sort_order", sort_order, "Must be 'asc' or 'desc'")
        
        return True


class ConfigValidator:
    """Configuration validation"""
    
    @staticmethod
    def validate_optimization_config(config: Dict[str, Any]) -> bool:
        """Validate optimization configuration"""
        required_fields = ['max_iterations', 'population_size', 'mutation_rate', 'crossover_rate']
        
        for field in required_fields:
            if field not in config:
                raise ConfigurationError(field, None, "Required field missing")
        
        # Validate specific fields
        if not isinstance(config['max_iterations'], int) or config['max_iterations'] < 10:
            raise ConfigurationError('max_iterations', config['max_iterations'], "Must be integer >= 10")
        
        if not isinstance(config['population_size'], int) or config['population_size'] < 10:
            raise ConfigurationError('population_size', config['population_size'], "Must be integer >= 10")
        
        if not 0.0 <= config['mutation_rate'] <= 1.0:
            raise ConfigurationError('mutation_rate', config['mutation_rate'], "Must be between 0.0 and 1.0")
        
        if not 0.0 <= config['crossover_rate'] <= 1.0:
            raise ConfigurationError('crossover_rate', config['crossover_rate'], "Must be between 0.0 and 1.0")
        
        return True
    
    @staticmethod
    def validate_scheduler_config(config: Dict[str, Any]) -> bool:
        """Validate scheduler configuration"""
        if 'temperature_schedule' in config:
            valid_schedules = ['exponential', 'linear', 'quantum']
            if config['temperature_schedule'] not in valid_schedules:
                raise ConfigurationError(
                    'temperature_schedule', 
                    config['temperature_schedule'], 
                    f"Must be one of: {valid_schedules}"
                )
        
        if 'initial_temperature' in config:
            if not isinstance(config['initial_temperature'], (int, float)) or config['initial_temperature'] <= 0:
                raise ConfigurationError(
                    'initial_temperature', 
                    config['initial_temperature'], 
                    "Must be positive number"
                )
        
        return True


class SecurityValidator:
    """Security-focused validation"""
    
    @staticmethod
    def validate_user_input(input_data: str, max_length: int = 1000) -> bool:
        """Validate user input for security"""
        if not isinstance(input_data, str):
            raise TaskValidationError("input", input_data, "Must be a string")
        
        if len(input_data) > max_length:
            raise TaskValidationError("input", input_data, f"Exceeds maximum length of {max_length}")
        
        # Check for malicious patterns
        if TaskValidator._contains_malicious_content(input_data):
            raise TaskValidationError("input", input_data, "Contains potentially malicious content")
        
        return True
    
    @staticmethod
    def validate_file_upload(filename: str, content_type: str, file_size: int) -> bool:
        """Validate file upload parameters"""
        # Allowed file types
        allowed_types = {
            'application/json', 'text/plain', 'text/csv', 
            'application/yaml', 'application/xml'
        }
        
        if content_type not in allowed_types:
            raise TaskValidationError("content_type", content_type, f"Must be one of: {allowed_types}")
        
        # File size limit (10MB)
        max_size = 10 * 1024 * 1024
        if file_size > max_size:
            raise TaskValidationError("file_size", file_size, f"Cannot exceed {max_size} bytes")
        
        # Filename validation
        if not re.match(r'^[a-zA-Z0-9._-]+$', filename):
            raise TaskValidationError("filename", filename, "Invalid filename format")
        
        # Check for directory traversal
        if '..' in filename or filename.startswith('/'):
            raise TaskValidationError("filename", filename, "Potential directory traversal detected")
        
        return True


# Validation decorators
def validate_task_input(func):
    """Decorator to validate task input parameters"""
    def wrapper(*args, **kwargs):
        # Extract task data from kwargs or args
        if 'task_data' in kwargs:
            task_data = kwargs['task_data']
        elif len(args) > 0 and hasattr(args[0], '__dict__'):
            task_data = args[0].__dict__
        else:
            return func(*args, **kwargs)
        
        # Validate task data
        if 'title' in task_data:
            TaskValidator.validate_title(task_data['title'])
        if 'description' in task_data:
            TaskValidator.validate_description(task_data['description'])
        if 'priority' in task_data and isinstance(task_data['priority'], str):
            TaskValidator.validate_priority(task_data['priority'])
        if 'tags' in task_data:
            TaskValidator.validate_tags(task_data['tags'])
        if 'complexity_factor' in task_data:
            TaskValidator.validate_complexity_factor(task_data['complexity_factor'])
        
        return func(*args, **kwargs)
    
    return wrapper


def validate_quantum_state(func):
    """Decorator to validate quantum state parameters"""
    def wrapper(*args, **kwargs):
        if 'coherence' in kwargs:
            QuantumValidator.validate_quantum_coherence(kwargs['coherence'])
        if 'strength' in kwargs:
            QuantumValidator.validate_entanglement_strength(kwargs['strength'])
        
        return func(*args, **kwargs)
    
    return wrapper