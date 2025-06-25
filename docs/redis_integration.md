# Redis Integration for Agentbx

This document describes how to use Redis with agentbx for data storage and caching.

## Overview

The Redis integration provides:

- Persistent storage for data bundles
- Caching for computed results
- Connection pooling and health monitoring
- Automatic serialization/deserialization of complex objects

## Setup

### 1. Install Redis

**Ubuntu/Debian:**

```bash
sudo apt-get install redis-server
sudo systemctl start redis-server
```

**macOS:**

```bash
brew install redis
brew services start redis
```

**Docker:**

```bash
docker run -d -p 6379:6379 redis:alpine
```

### 2. Install agentbx with Redis support

```bash
poetry install --with redis-agents
```

## Usage

### Basic Redis Manager

```python
from agentbx.core.redis_manager import RedisManager
from agentbx.agents.structure_factor_agent import StructureFactorAgent

# Initialize Redis manager
redis_manager = RedisManager(
    host="localhost",
    port=6379,
    db=0,
    default_ttl=3600  # 1 hour
)

# Create agent
agent = StructureFactorAgent(redis_manager, "sf_agent_001")

#
```
