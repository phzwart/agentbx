"""
Configuration management for agentbx.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class RedisConfig:
    """Redis configuration settings."""

    host: str = "localhost"
    port: int = 6379
    db: int = 0
    password: Optional[str] = None
    max_connections: int = 10
    socket_timeout: int = 5
    socket_connect_timeout: int = 5
    retry_on_timeout: bool = True
    health_check_interval: int = 30
    default_ttl: int = 3600  # 1 hour

    @classmethod
    def from_env(cls) -> "RedisConfig":
        """Create Redis config from environment variables."""
        return cls(
            host=os.getenv("AGENTBX_REDIS_HOST", "localhost"),
            port=int(os.getenv("AGENTBX_REDIS_PORT", "6379")),
            db=int(os.getenv("AGENTBX_REDIS_DB", "0")),
            password=os.getenv("AGENTBX_REDIS_PASSWORD"),
            max_connections=int(os.getenv("AGENTBX_REDIS_MAX_CONNECTIONS", "10")),
            socket_timeout=int(os.getenv("AGENTBX_REDIS_SOCKET_TIMEOUT", "5")),
            socket_connect_timeout=int(
                os.getenv("AGENTBX_REDIS_SOCKET_CONNECT_TIMEOUT", "5")
            ),
            retry_on_timeout=os.getenv("AGENTBX_REDIS_RETRY_ON_TIMEOUT", "true").lower()
            == "true",
            health_check_interval=int(
                os.getenv("AGENTBX_REDIS_HEALTH_CHECK_INTERVAL", "30")
            ),
            default_ttl=int(os.getenv("AGENTBX_REDIS_DEFAULT_TTL", "3600")),
        )


@dataclass
class AgentConfig:
    """Agent configuration settings."""

    agent_id: str
    redis_config: RedisConfig

    @classmethod
    def from_env(cls, agent_id: str) -> "AgentConfig":
        """Create agent config from environment variables."""
        return cls(agent_id=agent_id, redis_config=RedisConfig.from_env())
