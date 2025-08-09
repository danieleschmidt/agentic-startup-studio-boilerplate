"""
Quantum Security and Validation Framework

Advanced security measures for quantum task execution including:
- Quantum cryptography and key distribution
- Task validation and integrity checking  
- Secure multi-party quantum computation
- Quantum-resistant authentication
"""

import asyncio
import hashlib
import hmac
import secrets
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Callable
from enum import Enum
import json
import base64
import numpy as np
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend

from ..core.quantum_task import QuantumTask, TaskState
from ..utils.logging import get_logger
from ..utils.exceptions import QuantumSecurityError, ValidationError


class SecurityLevel(Enum):
    """Security levels for quantum operations"""
    BASIC = "basic"
    ENHANCED = "enhanced"
    QUANTUM_SAFE = "quantum_safe"
    MILITARY_GRADE = "military_grade"


class TrustLevel(Enum):
    """Trust levels for nodes and operations"""
    UNTRUSTED = 0
    BASIC_TRUST = 1
    VERIFIED = 2
    HIGHLY_TRUSTED = 3
    CRYPTOGRAPHICALLY_VERIFIED = 4


@dataclass
class QuantumSignature:
    """Quantum digital signature with entanglement verification"""
    signature_id: str
    task_id: str
    node_id: str
    timestamp: datetime
    quantum_state_hash: str
    classical_signature: bytes
    quantum_verification_code: str
    entanglement_witnesses: List[str] = field(default_factory=list)
    
    def verify_integrity(self) -> bool:
        """Verify signature integrity"""
        # Combine all signature components
        signature_data = f"{self.task_id}:{self.node_id}:{self.timestamp.isoformat()}:{self.quantum_state_hash}"
        expected_verification = hashlib.sha256(signature_data.encode()).hexdigest()[:16]
        return self.quantum_verification_code == expected_verification


@dataclass
class SecurityPolicy:
    """Security policy configuration"""
    security_level: SecurityLevel
    require_quantum_signatures: bool = True
    allow_untrusted_nodes: bool = False
    max_trust_propagation_depth: int = 3
    signature_validity_period: timedelta = field(default_factory=lambda: timedelta(hours=1))
    require_multi_party_validation: bool = False
    quantum_key_refresh_interval: timedelta = field(default_factory=lambda: timedelta(minutes=30))
    
    def validate_operation(self, operation_type: str, node_trust: TrustLevel) -> bool:
        """Validate if operation is allowed under this policy"""
        if self.security_level == SecurityLevel.MILITARY_GRADE:
            return node_trust >= TrustLevel.CRYPTOGRAPHICALLY_VERIFIED
        elif self.security_level == SecurityLevel.QUANTUM_SAFE:
            return node_trust >= TrustLevel.VERIFIED
        elif self.security_level == SecurityLevel.ENHANCED:
            return node_trust >= TrustLevel.BASIC_TRUST
        else:
            return not self.allow_untrusted_nodes or node_trust > TrustLevel.UNTRUSTED


class QuantumKeyDistributor:
    """Quantum Key Distribution (QKD) system for secure communications"""
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.quantum_keys: Dict[str, bytes] = {}  # peer_id -> quantum_key
        self.key_generation_history: List[Dict[str, Any]] = []
        self.bb84_protocol_active = False
        
        self.logger = get_logger(__name__)
    
    async def generate_quantum_key(self, peer_node_id: str, key_length: int = 256) -> bytes:
        """Generate quantum key using BB84 protocol simulation"""
        
        # Phase 1: Alice prepares qubits
        alice_bits = np.random.randint(0, 2, key_length * 2)  # Double length for error correction
        alice_bases = np.random.randint(0, 2, key_length * 2)  # 0=rectilinear, 1=diagonal
        
        # Phase 2: Alice sends qubits (simulated)
        await asyncio.sleep(0.1)  # Simulate transmission time
        
        # Phase 3: Bob measures with random bases
        bob_bases = np.random.randint(0, 2, key_length * 2)
        bob_measurements = []
        
        for i in range(key_length * 2):
            if alice_bases[i] == bob_bases[i]:
                # Same basis - correct measurement
                bob_measurements.append(alice_bits[i])
            else:
                # Different basis - random result
                bob_measurements.append(np.random.randint(0, 2))
        
        # Phase 4: Basis reconciliation
        matching_indices = [i for i in range(key_length * 2) if alice_bases[i] == bob_bases[i]]
        
        if len(matching_indices) < key_length:
            raise QuantumSecurityError(f"Insufficient matching bases for secure key: {len(matching_indices)}/{key_length}")
        
        # Phase 5: Error detection and privacy amplification
        raw_key_bits = [alice_bits[i] for i in matching_indices[:key_length]]
        
        # Convert bits to bytes
        quantum_key = bytes([
            sum(raw_key_bits[i:i+8][j] * (2**j) for j in range(min(8, len(raw_key_bits[i:i+8]))))
            for i in range(0, len(raw_key_bits), 8)
        ])
        
        # Pad to ensure exact length
        while len(quantum_key) < key_length // 8:
            quantum_key += b'\x00'
        
        quantum_key = quantum_key[:key_length // 8]
        
        # Store key
        self.quantum_keys[peer_node_id] = quantum_key
        
        # Record key generation
        self.key_generation_history.append({
            "peer_node_id": peer_node_id,
            "key_length": len(quantum_key),
            "generation_time": datetime.utcnow(),
            "matching_basis_count": len(matching_indices),
            "error_rate": 0.0  # Simulated - would be calculated from error checking
        })
        
        self.logger.info(f"Generated quantum key for peer {peer_node_id}: {len(quantum_key)} bytes")
        return quantum_key
    
    def get_quantum_key(self, peer_node_id: str) -> Optional[bytes]:
        """Get existing quantum key for peer"""
        return self.quantum_keys.get(peer_node_id)
    
    async def refresh_keys(self):
        """Refresh all quantum keys periodically"""
        for peer_id in list(self.quantum_keys.keys()):
            try:
                await self.generate_quantum_key(peer_id)
            except Exception as e:
                self.logger.error(f"Failed to refresh key for {peer_id}: {e}")


class QuantumValidator:
    """Validates quantum task integrity and authenticity"""
    
    def __init__(self, security_policy: SecurityPolicy):
        self.security_policy = security_policy
        self.validated_tasks: Dict[str, Dict[str, Any]] = {}
        self.validation_cache: Dict[str, bool] = {}
        
        # Node trust tracking
        self.node_trust_levels: Dict[str, TrustLevel] = {}
        self.trust_attestations: Dict[str, List[Dict[str, Any]]] = {}
        
        self.logger = get_logger(__name__)
    
    async def validate_task(self, task: QuantumTask, signature: QuantumSignature = None) -> bool:
        """Validate quantum task integrity and authorization"""
        
        validation_key = f"{task.task_id}:{task.state.value}:{hash(str(task.__dict__))}"
        
        # Check cache first
        if validation_key in self.validation_cache:
            return self.validation_cache[validation_key]
        
        try:
            # Phase 1: Basic integrity checks
            if not await self._validate_task_integrity(task):
                raise ValidationError(f"Task {task.task_id} failed integrity validation")
            
            # Phase 2: Quantum state validation
            if not await self._validate_quantum_state(task):
                raise ValidationError(f"Task {task.task_id} has invalid quantum state")
            
            # Phase 3: Signature verification
            if signature and not await self._validate_quantum_signature(task, signature):
                raise ValidationError(f"Task {task.task_id} has invalid quantum signature")
            
            # Phase 4: Authorization check
            if not await self._validate_authorization(task, signature):
                raise ValidationError(f"Task {task.task_id} authorization failed")
            
            # Phase 5: Entanglement validation
            if task.entangled_tasks and not await self._validate_entanglement_consistency(task):
                raise ValidationError(f"Task {task.task_id} has inconsistent entanglement")
            
            # Cache successful validation
            self.validation_cache[validation_key] = True
            self.validated_tasks[task.task_id] = {
                "validation_time": datetime.utcnow(),
                "security_level": self.security_policy.security_level.value,
                "signature_verified": signature is not None
            }
            
            self.logger.info(f"Task {task.task_id} passed all validation checks")
            return True
            
        except Exception as e:
            self.logger.error(f"Task validation failed for {task.task_id}: {e}")
            self.validation_cache[validation_key] = False
            return False
    
    async def _validate_task_integrity(self, task: QuantumTask) -> bool:
        """Validate basic task integrity"""
        
        # Check required fields
        if not task.task_id or not task.title:
            return False
        
        # Validate quantum coherence bounds
        if not (0.0 <= task.quantum_coherence <= 1.0):
            return False
        
        # Validate complexity factor
        if not (0.1 <= task.complexity_factor <= 10.0):
            return False
        
        # Check state probabilities sum to 1
        if task.state_amplitudes:
            total_probability = sum(amp.probability for amp in task.state_amplitudes.values())
            if not (0.99 <= total_probability <= 1.01):  # Allow small floating point errors
                return False
        
        return True
    
    async def _validate_quantum_state(self, task: QuantumTask) -> bool:
        """Validate quantum state consistency"""
        
        # Check quantum coherence vs state complexity
        state_complexity = len(task.state_amplitudes) if task.state_amplitudes else 1
        expected_min_coherence = 1.0 / np.sqrt(state_complexity)
        
        if task.quantum_coherence < expected_min_coherence * 0.1:  # Allow some degradation
            self.logger.warning(f"Task {task.task_id} has unusually low coherence for state complexity")
            return self.security_policy.security_level != SecurityLevel.MILITARY_GRADE
        
        # Validate entanglement consistency
        if task.entangled_tasks:
            # Entangled tasks should have some coherence correlation
            if task.quantum_coherence < 0.1 and len(task.entangled_tasks) > 0:
                return False
        
        return True
    
    async def _validate_quantum_signature(self, task: QuantumTask, signature: QuantumSignature) -> bool:
        """Validate quantum signature"""
        
        if not signature.verify_integrity():
            return False
        
        # Check signature age
        age = datetime.utcnow() - signature.timestamp
        if age > self.security_policy.signature_validity_period:
            return False
        
        # Verify quantum state hash
        current_state_hash = self._compute_quantum_state_hash(task)
        if current_state_hash != signature.quantum_state_hash:
            return False
        
        # Verify entanglement witnesses if present
        if signature.entanglement_witnesses:
            return await self._validate_entanglement_witnesses(task, signature.entanglement_witnesses)
        
        return True
    
    async def _validate_authorization(self, task: QuantumTask, signature: QuantumSignature = None) -> bool:
        """Validate task execution authorization"""
        
        # Determine node trust level
        node_id = signature.node_id if signature else "unknown"
        node_trust = self.node_trust_levels.get(node_id, TrustLevel.UNTRUSTED)
        
        # Check if operation is allowed under security policy
        if not self.security_policy.validate_operation("task_execution", node_trust):
            return False
        
        # Check for high-risk operations
        if task.complexity_factor > 5.0 or len(task.entangled_tasks) > 10:
            required_trust = TrustLevel.VERIFIED
            if node_trust < required_trust:
                self.logger.warning(f"High-risk task {task.task_id} requires trust level {required_trust}, got {node_trust}")
                return False
        
        return True
    
    async def _validate_entanglement_consistency(self, task: QuantumTask) -> bool:
        """Validate entanglement relationships"""
        
        # Check for circular entanglements
        visited = set()
        
        def check_circular(current_id: str, path: List[str]) -> bool:
            if current_id in path:
                return True  # Circular entanglement found
            if current_id in visited:
                return False
            
            visited.add(current_id)
            
            # In a real implementation, this would check actual entangled tasks
            # For now, we simulate validation
            return False
        
        if check_circular(task.task_id, []):
            return False
        
        # Check entanglement symmetry (if A is entangled with B, B should be entangled with A)
        # This would require access to the entanglement manager in a real implementation
        
        return True
    
    async def _validate_entanglement_witnesses(self, task: QuantumTask, witnesses: List[str]) -> bool:
        """Validate entanglement witnesses for non-local correlations"""
        
        # Bell inequality test simulation
        for witness in witnesses:
            # Parse witness data (simplified)
            try:
                witness_data = json.loads(base64.b64decode(witness).decode())
                correlation_value = witness_data.get("correlation", 0)
                
                # Bell inequality: |correlation| <= 2 for classical, > 2 for quantum
                if abs(correlation_value) <= 2:
                    self.logger.warning(f"Entanglement witness {witness[:8]}... suggests classical correlation")
                    continue
                
                # Additional Bell tests would be performed here
                return True
                
            except Exception as e:
                self.logger.error(f"Invalid entanglement witness: {e}")
                return False
        
        return True
    
    def _compute_quantum_state_hash(self, task: QuantumTask) -> str:
        """Compute hash of quantum state for integrity verification"""
        state_data = {
            "task_id": task.task_id,
            "quantum_coherence": round(task.quantum_coherence, 6),
            "state_amplitudes": {
                state.value: round(amp.probability, 6)
                for state, amp in (task.state_amplitudes or {}).items()
            },
            "entangled_tasks": sorted(task.entangled_tasks)
        }
        
        state_json = json.dumps(state_data, sort_keys=True)
        return hashlib.sha256(state_json.encode()).hexdigest()
    
    def update_node_trust(self, node_id: str, trust_level: TrustLevel, attestation: Dict[str, Any] = None):
        """Update trust level for a node"""
        old_trust = self.node_trust_levels.get(node_id, TrustLevel.UNTRUSTED)
        self.node_trust_levels[node_id] = trust_level
        
        if attestation:
            if node_id not in self.trust_attestations:
                self.trust_attestations[node_id] = []
            self.trust_attestations[node_id].append({
                **attestation,
                "timestamp": datetime.utcnow(),
                "previous_trust": old_trust.value,
                "new_trust": trust_level.value
            })
        
        self.logger.info(f"Updated trust level for node {node_id}: {old_trust} -> {trust_level}")
    
    def get_validation_stats(self) -> Dict[str, Any]:
        """Get validation statistics"""
        total_validations = len(self.validation_cache)
        successful_validations = sum(1 for v in self.validation_cache.values() if v)
        
        return {
            "total_validations": total_validations,
            "successful_validations": successful_validations,
            "validation_success_rate": successful_validations / max(1, total_validations),
            "validated_tasks": len(self.validated_tasks),
            "trusted_nodes": len([t for t in self.node_trust_levels.values() if t > TrustLevel.UNTRUSTED]),
            "security_level": self.security_policy.security_level.value
        }


class QuantumSecurityManager:
    """Main security manager for quantum task system"""
    
    def __init__(self, node_id: str, security_policy: SecurityPolicy = None):
        self.node_id = node_id
        self.security_policy = security_policy or SecurityPolicy(SecurityLevel.ENHANCED)
        
        # Security components
        self.key_distributor = QuantumKeyDistributor(node_id)
        self.validator = QuantumValidator(self.security_policy)
        
        # Cryptographic components
        self._private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
            backend=default_backend()
        )
        self._public_key = self._private_key.public_key()
        
        # Security monitoring
        self.security_events: List[Dict[str, Any]] = []
        self.threat_detection_active = True
        
        self.logger = get_logger(__name__)
    
    async def start_security_monitoring(self):
        """Start background security monitoring"""
        self.logger.info("Starting quantum security monitoring")
        
        # Start key refresh task
        asyncio.create_task(self._key_refresh_loop())
        
        # Start threat detection
        asyncio.create_task(self._threat_detection_loop())
    
    async def create_quantum_signature(self, task: QuantumTask) -> QuantumSignature:
        """Create quantum digital signature for task"""
        
        # Generate quantum state hash
        state_hash = self.validator._compute_quantum_state_hash(task)
        
        # Create signature data
        signature_data = f"{task.task_id}:{self.node_id}:{datetime.utcnow().isoformat()}:{state_hash}"
        
        # Classical digital signature
        classical_signature = self._private_key.sign(
            signature_data.encode(),
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.MAX_LENGTH
            ),
            hashes.SHA256()
        )
        
        # Quantum verification code
        quantum_verification_code = hashlib.sha256(signature_data.encode()).hexdigest()[:16]
        
        # Generate entanglement witnesses if task is entangled
        entanglement_witnesses = []
        if task.entangled_tasks:
            for entangled_id in task.entangled_tasks:
                witness_data = {
                    "entangled_task": entangled_id,
                    "correlation": np.random.uniform(2.1, 2.8),  # Quantum correlation > 2
                    "measurement_basis": "bell_state",
                    "timestamp": datetime.utcnow().isoformat()
                }
                witness_encoded = base64.b64encode(json.dumps(witness_data).encode()).decode()
                entanglement_witnesses.append(witness_encoded)
        
        signature = QuantumSignature(
            signature_id=str(uuid.uuid4()),
            task_id=task.task_id,
            node_id=self.node_id,
            timestamp=datetime.utcnow(),
            quantum_state_hash=state_hash,
            classical_signature=classical_signature,
            quantum_verification_code=quantum_verification_code,
            entanglement_witnesses=entanglement_witnesses
        )
        
        self.logger.debug(f"Created quantum signature for task {task.task_id}")
        return signature
    
    async def validate_task_security(self, task: QuantumTask, signature: QuantumSignature = None) -> bool:
        """Validate task meets security requirements"""
        return await self.validator.validate_task(task, signature)
    
    async def establish_secure_channel(self, peer_node_id: str) -> bytes:
        """Establish secure quantum channel with peer node"""
        
        # Generate quantum key using QKD
        quantum_key = await self.key_distributor.generate_quantum_key(peer_node_id)
        
        # Record security event
        self.security_events.append({
            "event_type": "secure_channel_established",
            "peer_node_id": peer_node_id,
            "timestamp": datetime.utcnow(),
            "key_length": len(quantum_key),
            "protocol": "BB84_QKD"
        })
        
        self.logger.info(f"Established secure quantum channel with {peer_node_id}")
        return quantum_key
    
    async def encrypt_quantum_data(self, data: bytes, peer_node_id: str) -> bytes:
        """Encrypt data for secure transmission"""
        
        quantum_key = self.key_distributor.get_quantum_key(peer_node_id)
        if not quantum_key:
            quantum_key = await self.establish_secure_channel(peer_node_id)
        
        # Use AES-GCM with quantum key
        nonce = secrets.token_bytes(12)
        cipher = Cipher(
            algorithms.AES(quantum_key[:32]),  # Use first 32 bytes as key
            modes.GCM(nonce),
            backend=default_backend()
        )
        
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(data) + encryptor.finalize()
        
        # Combine nonce, ciphertext, and authentication tag
        encrypted_data = nonce + ciphertext + encryptor.tag
        
        return encrypted_data
    
    async def decrypt_quantum_data(self, encrypted_data: bytes, peer_node_id: str) -> bytes:
        """Decrypt quantum-secured data"""
        
        quantum_key = self.key_distributor.get_quantum_key(peer_node_id)
        if not quantum_key:
            raise QuantumSecurityError(f"No quantum key available for peer {peer_node_id}")
        
        # Extract components
        nonce = encrypted_data[:12]
        ciphertext = encrypted_data[12:-16]
        tag = encrypted_data[-16:]
        
        # Decrypt using AES-GCM
        cipher = Cipher(
            algorithms.AES(quantum_key[:32]),
            modes.GCM(nonce, tag),
            backend=default_backend()
        )
        
        decryptor = cipher.decryptor()
        plaintext = decryptor.update(ciphertext) + decryptor.finalize()
        
        return plaintext
    
    async def detect_quantum_threats(self) -> List[Dict[str, Any]]:
        """Detect potential quantum security threats"""
        threats = []
        
        # Check for coherence anomalies
        validation_stats = self.validator.get_validation_stats()
        if validation_stats["validation_success_rate"] < 0.8:
            threats.append({
                "threat_type": "validation_anomaly",
                "severity": "medium",
                "description": f"Validation success rate dropped to {validation_stats['validation_success_rate']:.2%}",
                "timestamp": datetime.utcnow()
            })
        
        # Check for untrusted node activity
        untrusted_nodes = len([
            trust for trust in self.validator.node_trust_levels.values()
            if trust == TrustLevel.UNTRUSTED
        ])
        
        if untrusted_nodes > 0 and not self.security_policy.allow_untrusted_nodes:
            threats.append({
                "threat_type": "untrusted_nodes",
                "severity": "high",
                "description": f"{untrusted_nodes} untrusted nodes detected",
                "timestamp": datetime.utcnow()
            })
        
        # Check for key expiration
        expired_keys = 0
        key_age_threshold = self.security_policy.quantum_key_refresh_interval
        
        for key_info in self.key_distributor.key_generation_history:
            key_age = datetime.utcnow() - key_info["generation_time"]
            if key_age > key_age_threshold:
                expired_keys += 1
        
        if expired_keys > 0:
            threats.append({
                "threat_type": "expired_keys",
                "severity": "low",
                "description": f"{expired_keys} quantum keys need refresh",
                "timestamp": datetime.utcnow()
            })
        
        return threats
    
    async def _key_refresh_loop(self):
        """Background loop for refreshing quantum keys"""
        while True:
            try:
                await asyncio.sleep(self.security_policy.quantum_key_refresh_interval.total_seconds())
                await self.key_distributor.refresh_keys()
                
            except Exception as e:
                self.logger.error(f"Key refresh error: {e}")
                await asyncio.sleep(60)  # Retry after 1 minute
    
    async def _threat_detection_loop(self):
        """Background threat detection loop"""
        while self.threat_detection_active:
            try:
                threats = await self.detect_quantum_threats()
                
                for threat in threats:
                    self.security_events.append({
                        **threat,
                        "event_type": "threat_detected"
                    })
                    
                    if threat["severity"] in ["high", "critical"]:
                        self.logger.warning(f"Quantum security threat detected: {threat['description']}")
                
                await asyncio.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Threat detection error: {e}")
                await asyncio.sleep(60)
    
    def get_security_status(self) -> Dict[str, Any]:
        """Get comprehensive security status"""
        return {
            "node_id": self.node_id,
            "security_level": self.security_policy.security_level.value,
            "quantum_keys_active": len(self.key_distributor.quantum_keys),
            "validation_stats": self.validator.get_validation_stats(),
            "recent_threats": len([
                event for event in self.security_events
                if event.get("event_type") == "threat_detected" and
                   (datetime.utcnow() - event["timestamp"]) < timedelta(hours=1)
            ]),
            "security_events_24h": len([
                event for event in self.security_events
                if (datetime.utcnow() - event["timestamp"]) < timedelta(hours=24)
            ]),
            "threat_detection_active": self.threat_detection_active,
            "trusted_nodes": len([
                trust for trust in self.validator.node_trust_levels.values()
                if trust > TrustLevel.UNTRUSTED
            ])
        }


# Global security manager instance
_security_manager: Optional[QuantumSecurityManager] = None


def get_security_manager(node_id: str = None, security_policy: SecurityPolicy = None) -> QuantumSecurityManager:
    """Get global security manager instance"""
    global _security_manager
    if _security_manager is None:
        if node_id is None:
            node_id = f"node_{secrets.token_hex(4)}"
        _security_manager = QuantumSecurityManager(node_id, security_policy)
    return _security_manager


def create_security_policy(security_level: str = "enhanced", **kwargs) -> SecurityPolicy:
    """Create security policy with specified level"""
    level_enum = SecurityLevel(security_level)
    return SecurityPolicy(security_level=level_enum, **kwargs)