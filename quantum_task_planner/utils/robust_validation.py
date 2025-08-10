"""
Robust Validation System

Comprehensive validation with quantum-specific rules, sanitization, and security checks.
"""

import re
import uuid
from typing import Any, Dict, List, Optional, Union, Callable, Type
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass, field
import json
import hashlib
import html

from .exceptions import (
    TaskValidationError, QuantumCoherenceError, ConfigurationError,
    SecurityError, QuantumTaskPlannerError
)


class ValidationSeverity(Enum):
    """Validation error severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ValidationResult:
    """Validation result with detailed feedback"""
    valid: bool
    severity: ValidationSeverity = ValidationSeverity.INFO
    message: str = "Validation passed"
    field: Optional[str] = None
    value: Any = None
    suggestions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'valid': self.valid,
            'severity': self.severity.value,
            'message': self.message,
            'field': self.field,
            'value': str(self.value) if self.value is not None else None,
            'suggestions': self.suggestions,
            'metadata': self.metadata
        }


class ValidationRule:
    """Base validation rule"""
    
    def __init__(self, name: str, description: str, severity: ValidationSeverity = ValidationSeverity.ERROR):
        self.name = name
        self.description = description
        self.severity = severity
    
    def validate(self, value: Any, context: Dict[str, Any] = None) -> ValidationResult:
        """Validate value against rule"""
        raise NotImplementedError
    
    def __str__(self) -> str:
        return f"{self.name}: {self.description}"


class LengthRule(ValidationRule):
    """Validate string length"""
    
    def __init__(self, min_length: int = 0, max_length: int = None, **kwargs):
        self.min_length = min_length
        self.max_length = max_length
        super().__init__(
            f"length_{min_length}_{max_length or 'inf'}",
            f"Length between {min_length} and {max_length or 'unlimited'} characters",
            **kwargs
        )
    
    def validate(self, value: Any, context: Dict[str, Any] = None) -> ValidationResult:
        if not isinstance(value, str):
            return ValidationResult(
                valid=False,
                severity=self.severity,
                message=f"Expected string, got {type(value).__name__}",
                value=value
            )
        
        length = len(value)
        
        if length < self.min_length:
            return ValidationResult(
                valid=False,
                severity=self.severity,
                message=f"String too short: {length} < {self.min_length}",
                value=value,
                suggestions=[f"Add at least {self.min_length - length} more characters"]
            )
        
        if self.max_length and length > self.max_length:
            return ValidationResult(
                valid=False,
                severity=self.severity,
                message=f"String too long: {length} > {self.max_length}",
                value=value,
                suggestions=[f"Remove at least {length - self.max_length} characters"]
            )
        
        return ValidationResult(valid=True, message="Length valid")


class PatternRule(ValidationRule):
    """Validate string pattern"""
    
    def __init__(self, pattern: str, description: str = None, **kwargs):
        self.pattern = re.compile(pattern)
        super().__init__(
            f"pattern_{hashlib.md5(pattern.encode()).hexdigest()[:8]}",
            description or f"Must match pattern: {pattern}",
            **kwargs
        )
    
    def validate(self, value: Any, context: Dict[str, Any] = None) -> ValidationResult:
        if not isinstance(value, str):
            return ValidationResult(
                valid=False,
                severity=self.severity,
                message=f"Expected string, got {type(value).__name__}",
                value=value
            )
        
        if not self.pattern.match(value):
            return ValidationResult(
                valid=False,
                severity=self.severity,
                message=f"String does not match required pattern",
                value=value,
                suggestions=["Check format requirements", "Use valid characters only"]
            )
        
        return ValidationResult(valid=True, message="Pattern valid")


class RangeRule(ValidationRule):
    """Validate numeric range"""
    
    def __init__(self, min_value: float = None, max_value: float = None, 
                 inclusive: bool = True, **kwargs):
        self.min_value = min_value
        self.max_value = max_value
        self.inclusive = inclusive
        
        op = "<=" if inclusive else "<"
        desc_parts = []
        if min_value is not None:
            desc_parts.append(f"{min_value} {op}")
        desc_parts.append("value")
        if max_value is not None:
            desc_parts.append(f"{op} {max_value}")
        
        super().__init__(
            f"range_{min_value}_{max_value}",
            " ".join(desc_parts),
            **kwargs
        )
    
    def validate(self, value: Any, context: Dict[str, Any] = None) -> ValidationResult:
        try:
            numeric_value = float(value)
        except (ValueError, TypeError):
            return ValidationResult(
                valid=False,
                severity=self.severity,
                message=f"Expected numeric value, got {type(value).__name__}",
                value=value
            )
        
        if self.min_value is not None:
            if self.inclusive and numeric_value < self.min_value:
                return ValidationResult(
                    valid=False,
                    severity=self.severity,
                    message=f"Value {numeric_value} below minimum {self.min_value}",
                    value=value,
                    suggestions=[f"Use value >= {self.min_value}"]
                )
            elif not self.inclusive and numeric_value <= self.min_value:
                return ValidationResult(
                    valid=False,
                    severity=self.severity,
                    message=f"Value {numeric_value} not above minimum {self.min_value}",
                    value=value,
                    suggestions=[f"Use value > {self.min_value}"]
                )
        
        if self.max_value is not None:
            if self.inclusive and numeric_value > self.max_value:
                return ValidationResult(
                    valid=False,
                    severity=self.severity,
                    message=f"Value {numeric_value} above maximum {self.max_value}",
                    value=value,
                    suggestions=[f"Use value <= {self.max_value}"]
                )
            elif not self.inclusive and numeric_value >= self.max_value:
                return ValidationResult(
                    valid=False,
                    severity=self.severity,
                    message=f"Value {numeric_value} not below maximum {self.max_value}",
                    value=value,
                    suggestions=[f"Use value < {self.max_value}"]
                )
        
        return ValidationResult(valid=True, message="Range valid")


class QuantumCoherenceRule(ValidationRule):
    """Validate quantum coherence values"""
    
    def __init__(self, min_coherence: float = 0.0, warn_threshold: float = 0.3, **kwargs):
        self.min_coherence = min_coherence
        self.warn_threshold = warn_threshold
        super().__init__(
            "quantum_coherence",
            f"Quantum coherence must be >= {min_coherence}, warned if < {warn_threshold}",
            **kwargs
        )
    
    def validate(self, value: Any, context: Dict[str, Any] = None) -> ValidationResult:
        try:
            coherence = float(value)
        except (ValueError, TypeError):
            return ValidationResult(
                valid=False,
                severity=ValidationSeverity.CRITICAL,
                message=f"Invalid coherence value: {value}",
                value=value,
                suggestions=["Provide numeric coherence value between 0.0 and 1.0"]
            )
        
        if coherence < 0.0 or coherence > 1.0:
            return ValidationResult(
                valid=False,
                severity=ValidationSeverity.CRITICAL,
                message=f"Coherence {coherence} outside valid range [0.0, 1.0]",
                value=value,
                suggestions=["Normalize coherence to [0.0, 1.0] range"]
            )
        
        if coherence < self.min_coherence:
            return ValidationResult(
                valid=False,
                severity=ValidationSeverity.ERROR,
                message=f"Coherence {coherence} below minimum {self.min_coherence}",
                value=value,
                suggestions=[
                    "Apply quantum error correction",
                    "Reduce environmental noise",
                    f"Increase coherence to >= {self.min_coherence}"
                ]
            )
        
        if coherence < self.warn_threshold:
            return ValidationResult(
                valid=True,
                severity=ValidationSeverity.WARNING,
                message=f"Low coherence warning: {coherence} < {self.warn_threshold}",
                value=value,
                suggestions=[
                    "Monitor decoherence rate",
                    "Consider coherence restoration",
                    "Check quantum environment stability"
                ]
            )
        
        return ValidationResult(
            valid=True, 
            message=f"Coherence optimal: {coherence}",
            metadata={"coherence_quality": "optimal" if coherence > 0.8 else "good"}
        )


class EntanglementRule(ValidationRule):
    """Validate quantum entanglement parameters"""
    
    def __init__(self, min_tasks: int = 2, max_tasks: int = 10, 
                 min_strength: float = 0.1, **kwargs):
        self.min_tasks = min_tasks
        self.max_tasks = max_tasks
        self.min_strength = min_strength
        super().__init__(
            "quantum_entanglement",
            f"Entanglement: {min_tasks}-{max_tasks} tasks, strength >= {min_strength}",
            **kwargs
        )
    
    def validate(self, value: Any, context: Dict[str, Any] = None) -> ValidationResult:
        if not isinstance(value, dict):
            return ValidationResult(
                valid=False,
                severity=self.severity,
                message="Entanglement parameters must be a dictionary",
                value=value
            )
        
        errors = []
        warnings = []
        suggestions = []
        
        # Validate task count
        task_ids = value.get('task_ids', [])
        if not isinstance(task_ids, list):
            errors.append("task_ids must be a list")
        elif len(task_ids) < self.min_tasks:
            errors.append(f"Need at least {self.min_tasks} tasks for entanglement")
            suggestions.append(f"Add {self.min_tasks - len(task_ids)} more tasks")
        elif len(task_ids) > self.max_tasks:
            warnings.append(f"Many tasks ({len(task_ids)}) may reduce entanglement quality")
            suggestions.append("Consider splitting into smaller entanglement groups")
        
        # Validate strength
        strength = value.get('strength', 0.0)
        try:
            strength = float(strength)
            if strength < self.min_strength:
                errors.append(f"Entanglement strength {strength} below minimum {self.min_strength}")
                suggestions.append(f"Increase strength to >= {self.min_strength}")
            elif strength > 1.0:
                errors.append(f"Entanglement strength {strength} above maximum 1.0")
                suggestions.append("Normalize strength to [0.0, 1.0] range")
        except (ValueError, TypeError):
            errors.append(f"Invalid strength value: {strength}")
            suggestions.append("Provide numeric strength between 0.0 and 1.0")
        
        # Validate entanglement type
        ent_type = value.get('entanglement_type')
        valid_types = ['bell_state', 'ghz_state', 'cluster_state', 'dependency', 'resource_shared', 'temporal']
        if ent_type not in valid_types:
            errors.append(f"Invalid entanglement type: {ent_type}")
            suggestions.append(f"Use one of: {', '.join(valid_types)}")
        
        if errors:
            return ValidationResult(
                valid=False,
                severity=self.severity,
                message=f"Entanglement validation failed: {'; '.join(errors)}",
                value=value,
                suggestions=suggestions
            )
        
        severity = ValidationSeverity.WARNING if warnings else ValidationSeverity.INFO
        message = '; '.join(warnings) if warnings else "Entanglement parameters valid"
        
        return ValidationResult(
            valid=True,
            severity=severity,
            message=message,
            suggestions=suggestions,
            metadata={
                'task_count': len(task_ids),
                'strength': strength,
                'type': ent_type
            }
        )


class SecurityRule(ValidationRule):
    """Validate security-related inputs"""
    
    def __init__(self, check_xss: bool = True, check_sql: bool = True, 
                 check_path_traversal: bool = True, **kwargs):
        self.check_xss = check_xss
        self.check_sql = check_sql
        self.check_path_traversal = check_path_traversal
        super().__init__(
            "security_validation",
            "Security validation for malicious content",
            ValidationSeverity.CRITICAL,
            **kwargs
        )
    
    def validate(self, value: Any, context: Dict[str, Any] = None) -> ValidationResult:
        if not isinstance(value, str):
            return ValidationResult(valid=True, message="Non-string values pass security check")
        
        threats = []
        suggestions = []
        
        # XSS detection
        if self.check_xss:
            xss_patterns = [
                r'<script[^>]*>.*?</script>',
                r'javascript:',
                r'on\w+\s*=',
                r'<iframe',
                r'<object',
                r'<embed',
                r'vbscript:'
            ]
            
            for pattern in xss_patterns:
                if re.search(pattern, value, re.IGNORECASE):
                    threats.append("Potential XSS attack detected")
                    suggestions.append("Remove script tags and event handlers")
                    break
        
        # SQL injection detection
        if self.check_sql:
            sql_patterns = [
                r"union\s+select",
                r"drop\s+table",
                r"delete\s+from",
                r"insert\s+into",
                r"update\s+.*set",
                r"exec\s*\(",
                r"'.*or.*'.*=.*'",
                r"--.*",
                r"/\*.*\*/"
            ]
            
            for pattern in sql_patterns:
                if re.search(pattern, value, re.IGNORECASE):
                    threats.append("Potential SQL injection detected")
                    suggestions.append("Use parameterized queries")
                    break
        
        # Path traversal detection
        if self.check_path_traversal:
            if "../" in value or "..\\\\" in value or "..\\\\" in value:
                threats.append("Path traversal attempt detected")
                suggestions.append("Remove directory traversal sequences")
        
        if threats:
            return ValidationResult(
                valid=False,
                severity=ValidationSeverity.CRITICAL,
                message=f"Security threats detected: {'; '.join(threats)}",
                value="[REDACTED FOR SECURITY]",
                suggestions=suggestions + ["Sanitize input before processing"]
            )
        
        return ValidationResult(valid=True, message="Security validation passed")


class QuantumValidator:
    """Comprehensive quantum task validator"""
    
    def __init__(self):
        self.rules: Dict[str, List[ValidationRule]] = {
            'task_title': [
                LengthRule(min_length=1, max_length=200),
                SecurityRule(),
                PatternRule(r'^[a-zA-Z0-9\s\-_.()]+$', "Alphanumeric with basic punctuation")
            ],
            'task_description': [
                LengthRule(min_length=1, max_length=2000),
                SecurityRule()
            ],
            'quantum_coherence': [
                QuantumCoherenceRule(min_coherence=0.1, warn_threshold=0.3)
            ],
            'priority': [
                PatternRule(r'^(critical|high|medium|low|minimal)$', "Valid priority level")
            ],
            'complexity_factor': [
                RangeRule(min_value=0.1, max_value=10.0, inclusive=True)
            ],
            'entanglement_params': [
                EntanglementRule(min_tasks=2, max_tasks=10, min_strength=0.1)
            ],
            'task_id': [
                PatternRule(r'^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$',
                           "Valid UUID format")
            ],
            'duration_hours': [
                RangeRule(min_value=0.01, max_value=8760.0)  # Up to 1 year
            ]
        }
    
    def add_rule(self, field: str, rule: ValidationRule):
        """Add validation rule for field"""
        if field not in self.rules:
            self.rules[field] = []
        self.rules[field].append(rule)
    
    def validate_field(self, field: str, value: Any, context: Dict[str, Any] = None) -> List[ValidationResult]:
        """Validate single field"""
        results = []
        
        if field in self.rules:
            for rule in self.rules[field]:
                try:
                    result = rule.validate(value, context)
                    result.field = field
                    results.append(result)
                    
                    # Stop on critical errors
                    if not result.valid and result.severity == ValidationSeverity.CRITICAL:
                        break
                        
                except Exception as e:
                    results.append(ValidationResult(
                        valid=False,
                        severity=ValidationSeverity.ERROR,
                        message=f"Validation rule error: {str(e)}",
                        field=field,
                        value=value
                    ))
        
        return results
    
    def validate_task(self, task_data: Dict[str, Any]) -> Dict[str, List[ValidationResult]]:
        """Validate complete task data"""
        all_results = {}
        
        # Standard task fields
        standard_fields = ['task_title', 'task_description', 'priority', 'complexity_factor']
        
        for field in standard_fields:
            # Map field names
            data_key = field.replace('task_', '') if field.startswith('task_') else field
            
            if data_key in task_data:
                results = self.validate_field(field, task_data[data_key])
                if results:
                    all_results[field] = results
        
        # Special validations
        if 'estimated_duration_hours' in task_data:
            results = self.validate_field('duration_hours', task_data['estimated_duration_hours'])
            if results:
                all_results['duration_hours'] = results
        
        if 'task_id' in task_data:
            results = self.validate_field('task_id', task_data['task_id'])
            if results:
                all_results['task_id'] = results
        
        # Quantum-specific validations
        if 'quantum_coherence' in task_data:
            results = self.validate_field('quantum_coherence', task_data['quantum_coherence'])
            if results:
                all_results['quantum_coherence'] = results
        
        return all_results
    
    def validate_entanglement(self, entanglement_data: Dict[str, Any]) -> List[ValidationResult]:
        """Validate entanglement parameters"""
        return self.validate_field('entanglement_params', entanglement_data)
    
    def is_valid(self, validation_results: Dict[str, List[ValidationResult]]) -> bool:
        """Check if all validation results are valid"""
        for field_results in validation_results.values():
            for result in field_results:
                if not result.valid:
                    return False
        return True
    
    def get_errors(self, validation_results: Dict[str, List[ValidationResult]]) -> List[ValidationResult]:
        """Get all validation errors"""
        errors = []
        for field_results in validation_results.values():
            for result in field_results:
                if not result.valid:
                    errors.append(result)
        return errors
    
    def sanitize_input(self, value: str, aggressive: bool = False) -> str:
        """Sanitize input string"""
        if not isinstance(value, str):
            return str(value)
        
        # HTML escape
        sanitized = html.escape(value)
        
        if aggressive:
            # Remove all HTML tags
            sanitized = re.sub(r'<[^>]+>', '', sanitized)
            
            # Remove SQL keywords
            sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'DROP', 'UNION', 'EXEC']
            for keyword in sql_keywords:
                sanitized = re.sub(f'\\b{keyword}\\b', '', sanitized, flags=re.IGNORECASE)
            
            # Remove script-related content
            sanitized = re.sub(r'javascript:', '', sanitized, flags=re.IGNORECASE)
            sanitized = re.sub(r'vbscript:', '', sanitized, flags=re.IGNORECASE)
            sanitized = re.sub(r'on\w+\s*=', '', sanitized, flags=re.IGNORECASE)
        
        return sanitized.strip()


# Global validator instance
_quantum_validator = QuantumValidator()


def get_validator() -> QuantumValidator:
    """Get global validator instance"""
    return _quantum_validator


def validate_task_data(task_data: Dict[str, Any]) -> Dict[str, List[ValidationResult]]:
    """Validate task data using global validator"""
    return _quantum_validator.validate_task(task_data)


def validate_and_sanitize(field: str, value: Any, sanitize: bool = True) -> tuple[Any, List[ValidationResult]]:
    """Validate and optionally sanitize input"""
    results = _quantum_validator.validate_field(field, value)
    
    # Sanitize if needed and valid
    if sanitize and isinstance(value, str) and all(r.valid for r in results):
        value = _quantum_validator.sanitize_input(value)
    
    return value, results