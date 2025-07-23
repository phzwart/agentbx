"""
Agent security manager for handling permissions, validation, and security policies.

This module provides comprehensive security management for async agents,
including permission validation, module whitelisting, and security audits.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from pydantic import BaseModel
from pydantic import Field
from pydantic import field_validator

from agentbx.core.redis_manager import RedisManager
from agentbx.schemas.generated import AgentSecurityBundle


@dataclass
class SecurityViolation:
    """Represents a security violation."""

    violation_type: str
    message: str
    severity: str  # "low", "medium", "high", "critical"
    timestamp: datetime
    agent_id: str
    details: Optional[Dict[str, Any]] = None


class AgentRegistration(BaseModel):
    """Agent registration information."""

    agent_id: str
    agent_name: str
    agent_type: str
    version: str
    permissions: List[str]
    capabilities: Optional[List[Dict[str, Any]]] = None
    created_at: datetime = Field(default_factory=datetime.now)
    last_updated: datetime = Field(default_factory=datetime.now)

    @field_validator("agent_type")
    @classmethod
    def validate_agent_type(cls, v):
        """Validate agent type."""
        valid_types = [
            "geometry_agent",
            "structure_factor_agent",
            "target_agent",
            "gradient_agent",
        ]
        if v not in valid_types:
            raise ValueError(f"Invalid agent type: {v}. Must be one of {valid_types}")
        return v

    @field_validator("permissions")
    @classmethod
    def validate_permissions(cls, v):
        """Validate permissions."""
        valid_permissions = {
            "geometry_calculation",
            "structure_factor_calculation",
            "target_calculation",
            "gradient_calculation",
            "bundle_read",
            "bundle_write",
            "bundle_delete",
            "coordinate_update",
            "experimental_data_processing",
        }
        for permission in v:
            if permission not in valid_permissions:
                raise ValueError(f"Invalid permission: {permission}")
        return v


class SecurityPolicy(BaseModel):
    """Security policy configuration."""

    max_execution_time: int = 300  # seconds
    memory_limit_mb: int = 1024
    cpu_limit_percent: int = 80
    allowed_file_paths: List[str] = []
    network_access: bool = False
    sandbox_mode: bool = True
    max_import_depth: int = 3
    whitelisted_modules: List[str] = []
    blacklisted_modules: List[str] = []


class AgentSecurityManager:
    """
    Manages agent security, permissions, and validation.

    Features:
    - Agent registration and validation
    - Permission checking and enforcement
    - Module import validation
    - Security policy enforcement
    - Security audit logging
    """

    def __init__(self, redis_manager: RedisManager):
        """
        Initialize the security manager.

        Args:
            redis_manager: Redis manager for bundle operations
        """
        self.redis_manager = redis_manager
        self.logger = logging.getLogger("AgentSecurityManager")

        # Security state
        self.registered_agents: Dict[str, AgentRegistration] = {}
        self.security_policies: Dict[str, SecurityPolicy] = {}
        self.violations: List[SecurityViolation] = []

        # Default security policy
        self.default_policy = SecurityPolicy(
            whitelisted_modules=[
                "cctbx",
                "mmtbx",
                "iotbx",
                "libtbx",
                "scitbx",
                "agentbx.processors",
                "agentbx.core",
                "agentbx.schemas",
                "torch",
                "numpy",
                "scipy",
                "asyncio",
                "json",
                "time",
                "datetime",
                "logging",
                "uuid",
                "dataclasses",
            ],
            blacklisted_modules=[
                "os",
                "sys",
                "subprocess",
                "multiprocessing",
                "socket",
                "urllib",
                "ftplib",
                "smtplib",
                "telnetlib",
            ],
        )

    def register_agent(self, registration: AgentRegistration) -> bool:
        """
        Register an agent with security validation.

        Args:
            registration: Agent registration information

        Returns:
            True if registration successful, False otherwise
        """
        try:
            # Validate registration
            self._validate_registration(registration)

            # Check for conflicts
            if registration.agent_id in self.registered_agents:
                self.logger.warning(f"Agent {registration.agent_id} already registered")
                return False

            # Store registration
            self.registered_agents[registration.agent_id] = registration

            # Create security bundle
            security_bundle = self._create_security_bundle(registration)
            bundle_id = f"{registration.agent_id}_security"
            self.redis_manager.store_bundle(security_bundle, bundle_id)

            self.logger.info(f"Registered agent {registration.agent_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to register agent {registration.agent_id}: {e}")
            self._log_violation(
                "registration_failed",
                f"Agent registration failed: {e}",
                "high",
                registration.agent_id,
            )
            return False

    def unregister_agent(self, agent_id: str) -> bool:
        """
        Unregister an agent.

        Args:
            agent_id: Agent ID to unregister

        Returns:
            True if unregistration successful
        """
        if agent_id in self.registered_agents:
            del self.registered_agents[agent_id]
            self.logger.info(f"Unregistered agent {agent_id}")
            return True
        return False

    def check_permission(self, agent_id: str, permission: str) -> bool:
        """
        Check if an agent has a specific permission.

        Args:
            agent_id: Agent ID
            permission: Permission to check

        Returns:
            True if agent has permission
        """
        if agent_id not in self.registered_agents:
            self.logger.warning(f"Agent {agent_id} not registered")
            return False

        registration = self.registered_agents[agent_id]
        has_permission = permission in registration.permissions

        if not has_permission:
            self._log_violation(
                "permission_denied",
                f"Agent {agent_id} denied permission: {permission}",
                "medium",
                agent_id,
                {
                    "requested_permission": permission,
                    "available_permissions": registration.permissions,
                },
            )

        return has_permission

    def validate_module_import(self, agent_id: str, module_name: str) -> bool:
        """
        Validate if an agent can import a specific module.

        Args:
            agent_id: Agent ID
            module_name: Module name to validate

        Returns:
            True if import is allowed
        """
        policy = self._get_agent_policy(agent_id)

        # Check blacklist first
        if module_name in policy.blacklisted_modules:
            self._log_violation(
                "blacklisted_import",
                f"Agent {agent_id} attempted to import blacklisted module: {module_name}",
                "high",
                agent_id,
                {"module_name": module_name},
            )
            return False

        # Check whitelist
        if module_name not in policy.whitelisted_modules:
            self._log_violation(
                "unwhitelisted_import",
                f"Agent {agent_id} attempted to import unwhitelisted module: {module_name}",
                "medium",
                agent_id,
                {"module_name": module_name},
            )
            return False

        return True

    def validate_file_access(
        self, agent_id: str, file_path: str, access_type: str
    ) -> bool:
        """
        Validate if an agent can access a specific file.

        Args:
            agent_id: Agent ID
            file_path: File path to validate
            access_type: Type of access ("read", "write", "delete")

        Returns:
            True if access is allowed
        """
        policy = self._get_agent_policy(agent_id)

        # Check if path is in allowed paths
        for allowed_path in policy.allowed_file_paths:
            if file_path.startswith(allowed_path):
                return True

        self._log_violation(
            "file_access_denied",
            f"Agent {agent_id} denied {access_type} access to: {file_path}",
            "medium",
            agent_id,
            {"file_path": file_path, "access_type": access_type},
        )
        return False

    def validate_network_access(self, agent_id: str, host: str, port: int) -> bool:
        """
        Validate if an agent can make network connections.

        Args:
            agent_id: Agent ID
            host: Target host
            port: Target port

        Returns:
            True if network access is allowed
        """
        policy = self._get_agent_policy(agent_id)

        if not policy.network_access:
            self._log_violation(
                "network_access_denied",
                f"Agent {agent_id} denied network access to {host}:{port}",
                "high",
                agent_id,
                {"host": host, "port": port},
            )
            return False

        return True

    def audit_agent_activity(
        self, agent_id: str, activity: str, details: Dict[str, Any]
    ) -> None:
        """
        Audit agent activity for security monitoring.

        Args:
            agent_id: Agent ID
            activity: Activity type
            details: Activity details
        """
        if agent_id not in self.registered_agents:
            self._log_violation(
                "unregistered_activity",
                f"Unregistered agent {agent_id} attempted activity: {activity}",
                "high",
                agent_id,
                details,
            )
            return

        # Log activity for monitoring
        self.logger.info(f"Agent {agent_id} activity: {activity}")

        # Check for suspicious patterns
        self._check_suspicious_activity(agent_id, activity, details)

    def get_agent_security_report(self, agent_id: str) -> Dict[str, Any]:
        """
        Get security report for an agent.

        Args:
            agent_id: Agent ID

        Returns:
            Security report dictionary
        """
        if agent_id not in self.registered_agents:
            return {"error": "Agent not registered"}

        registration = self.registered_agents[agent_id]
        policy = self._get_agent_policy(agent_id)

        # Get violations for this agent
        agent_violations = [v for v in self.violations if v.agent_id == agent_id]

        return {
            "agent_id": agent_id,
            "registration": registration.model_dump(),
            "security_policy": policy.model_dump(),
            "violations": [v.__dict__ for v in agent_violations],
            "violation_count": len(agent_violations),
            "last_activity": registration.last_updated.isoformat(),
        }

    def _validate_registration(self, registration: AgentRegistration) -> None:
        """Validate agent registration."""
        # Check agent ID format
        if not registration.agent_id or len(registration.agent_id) < 3:
            raise ValueError("Agent ID must be at least 3 characters")

        # Check agent name
        if not registration.agent_name or len(registration.agent_name) < 2:
            raise ValueError("Agent name must be at least 2 characters")

        # Check version format
        if not registration.version or "." not in registration.version:
            raise ValueError("Version must be in format X.Y.Z")

        # Check permissions
        if not registration.permissions:
            raise ValueError("Agent must have at least one permission")

    def _create_security_bundle(
        self, registration: AgentRegistration
    ) -> AgentSecurityBundle:
        """Create security bundle for agent registration."""
        # Create agent registration dict
        agent_registration = {
            "agent_id": registration.agent_id,
            "agent_name": registration.agent_name,
            "agent_type": registration.agent_type,
            "version": registration.version,
            "permissions": registration.permissions,
            "capabilities": registration.capabilities or [],
            "created_at": registration.created_at.isoformat(),
            "last_updated": registration.last_updated.isoformat(),
        }

        # Create security policies
        security_policies = {
            "max_execution_time": 300,
            "memory_limit_mb": 1024,
            "cpu_limit_percent": 80,
            "allowed_file_paths": [],
            "network_access": False,
            "sandbox_mode": True,
        }

        return AgentSecurityBundle(
            agent_registration=agent_registration,
            permissions=registration.permissions,
            capabilities=registration.capabilities or [],
            whitelisted_modules=self.default_policy.whitelisted_modules,
            security_policies=security_policies,
        )

    def _get_agent_policy(self, agent_id: str) -> SecurityPolicy:
        """Get security policy for an agent."""
        if agent_id in self.security_policies:
            return self.security_policies[agent_id]
        return self.default_policy

    def _log_violation(
        self,
        violation_type: str,
        message: str,
        severity: str,
        agent_id: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Log a security violation."""
        violation = SecurityViolation(
            violation_type=violation_type,
            message=message,
            severity=severity,
            timestamp=datetime.now(),
            agent_id=agent_id,
            details=details,
        )

        self.violations.append(violation)
        self.logger.warning(f"Security violation: {message}")

        # Store violation in Redis for monitoring
        try:
            violation_key = (
                f"agentbx:violations:{agent_id}:{violation.timestamp.isoformat()}"
            )
            self.redis_manager.cache_set(
                violation_key, violation.__dict__, ttl=86400
            )  # 24 hours
        except Exception as e:
            self.logger.error(f"Failed to store violation: {e}")

    def _check_suspicious_activity(
        self, agent_id: str, activity: str, details: Dict[str, Any]
    ) -> None:
        """Check for suspicious activity patterns."""
        # Check for rapid repeated activities
        recent_activities = [
            v
            for v in self.violations
            if v.agent_id == agent_id and (datetime.now() - v.timestamp).seconds < 60
        ]

        if len(recent_activities) > 10:
            self._log_violation(
                "suspicious_activity",
                f"Agent {agent_id} showing suspicious activity pattern",
                "medium",
                agent_id,
                {"activity_count": len(recent_activities), "time_window": "60s"},
            )

        # Check for unusual resource usage
        if "memory_usage" in details and details["memory_usage"] > 1024:
            self._log_violation(
                "high_memory_usage",
                f"Agent {agent_id} using excessive memory: {details['memory_usage']}MB",
                "medium",
                agent_id,
                details,
            )

    def get_all_violations(self) -> List[SecurityViolation]:
        """Get all security violations."""
        return self.violations.copy()

    def clear_violations(self, agent_id: Optional[str] = None) -> None:
        """Clear security violations."""
        if agent_id:
            self.violations = [v for v in self.violations if v.agent_id != agent_id]
        else:
            self.violations.clear()
