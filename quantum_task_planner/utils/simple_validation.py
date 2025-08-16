"""
Simple validation utilities for Generation 2 robustness
"""

import re
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
from pydantic import BaseModel, ValidationError


class TaskValidationError(Exception):
    """Custom exception for task validation errors"""
    pass


class InputSanitizer:
    """Simple input sanitization for security"""
    
    @staticmethod
    def sanitize_string(input_str: str, max_length: int = 1000) -> str:
        """Sanitize string input"""
        if not isinstance(input_str, str):
            raise TaskValidationError("Input must be a string")
        
        # Remove any potentially harmful characters
        sanitized = re.sub(r'[<>"\'\&]', '', input_str)
        
        # Truncate to max length
        if len(sanitized) > max_length:
            sanitized = sanitized[:max_length]
        
        return sanitized.strip()
    
    @staticmethod
    def validate_task_id(task_id: str) -> str:
        """Validate task ID format"""
        if not task_id:
            raise TaskValidationError("Task ID cannot be empty")
        
        # Only allow alphanumeric, hyphens, and underscores
        if not re.match(r'^[a-zA-Z0-9_-]+$', task_id):
            raise TaskValidationError("Task ID can only contain letters, numbers, hyphens, and underscores")
        
        if len(task_id) > 100:
            raise TaskValidationError("Task ID too long (max 100 characters)")
        
        return task_id
    
    @staticmethod
    def validate_priority(priority: Union[float, str]) -> float:
        """Validate priority value"""
        try:
            priority_float = float(priority)
        except (ValueError, TypeError):
            raise TaskValidationError("Priority must be a number")
        
        if not 0.0 <= priority_float <= 1.0:
            raise TaskValidationError("Priority must be between 0.0 and 1.0")
        
        return priority_float


class SecurityValidator:
    """Security validation for quantum task operations"""
    
    DANGEROUS_PATTERNS = [
        r'<script.*?>.*?</script>',  # Script tags
        r'javascript:',              # JavaScript URLs
        r'data:.*base64',           # Base64 data URLs
        r'eval\s*\(',               # eval() calls
        r'exec\s*\(',               # exec() calls
        r'import\s+os',             # OS imports
        r'subprocess',              # Subprocess calls
    ]
    
    @classmethod
    def is_safe_content(cls, content: str) -> bool:
        """Check if content contains potentially dangerous patterns"""
        for pattern in cls.DANGEROUS_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                return False
        return True
    
    @classmethod
    def validate_safe_content(cls, content: str, field_name: str = "content") -> str:
        """Validate content is safe and sanitize"""
        if not cls.is_safe_content(content):
            raise TaskValidationError(f"Potentially dangerous content detected in {field_name}")
        
        return InputSanitizer.sanitize_string(content)


class QuantumStateValidator:
    """Validate quantum-specific properties"""
    
    @staticmethod
    def validate_coherence(coherence: Union[float, str]) -> float:
        """Validate quantum coherence value"""
        try:
            coherence_float = float(coherence)
        except (ValueError, TypeError):
            raise TaskValidationError("Quantum coherence must be a number")
        
        if not 0.0 <= coherence_float <= 1.0:
            raise TaskValidationError("Quantum coherence must be between 0.0 and 1.0")
        
        return coherence_float
    
    @staticmethod
    def validate_probability(probability: Union[float, str]) -> float:
        """Validate probability value"""
        try:
            prob_float = float(probability)
        except (ValueError, TypeError):
            raise TaskValidationError("Probability must be a number")
        
        if not 0.0 <= prob_float <= 1.0:
            raise TaskValidationError("Probability must be between 0.0 and 1.0")
        
        return prob_float


def validate_task_creation_input(task_data: Dict[str, Any]) -> Dict[str, Any]:
    """Comprehensive validation for task creation"""
    validated_data = {}
    
    # Validate required fields
    if 'title' not in task_data or not task_data['title']:
        raise TaskValidationError("Task title is required")
    
    if 'description' not in task_data or not task_data['description']:
        raise TaskValidationError("Task description is required")
    
    # Validate and sanitize title
    validated_data['title'] = SecurityValidator.validate_safe_content(
        task_data['title'], 'title'
    )
    
    # Validate and sanitize description
    validated_data['description'] = SecurityValidator.validate_safe_content(
        task_data['description'], 'description'
    )
    
    # Validate task_id if provided
    if 'task_id' in task_data:
        validated_data['task_id'] = InputSanitizer.validate_task_id(task_data['task_id'])
    
    # Validate optional quantum properties
    if 'quantum_coherence' in task_data:
        validated_data['quantum_coherence'] = QuantumStateValidator.validate_coherence(
            task_data['quantum_coherence']
        )
    
    if 'success_probability' in task_data:
        validated_data['success_probability'] = QuantumStateValidator.validate_probability(
            task_data['success_probability']
        )
    
    return validated_data