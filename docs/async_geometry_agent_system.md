# Async Geometry Agent System

## Overview

The Async Geometry Agent System is a comprehensive solution for background geometry calculations in crystallographic software using CCTBX and Redis. It provides reliable, scalable, and secure processing of geometry gradient calculations with full integration into the AgentBX ecosystem.

**⚠️ Experimental Status**: This system is currently experimental and subject to change. The API, features, and implementation details may evolve as development continues. We recommend using this system for research and development purposes only.

**🎯 Focus Areas**: The system is specifically designed for handling crystallographic data and interfacing with protein foundation models. It provides specialized tools and workflows optimized for structural biology applications.

## Experimental Status & Development Focus

### Current Status

This system is in active development and should be considered experimental:

- **API Stability**: The API is not yet stable and may change between versions
- **Feature Completeness**: Some features may be incomplete or subject to modification
- **Performance**: Performance characteristics may vary and are being optimized
- **Documentation**: Documentation may not reflect the latest implementation

### Development Priorities

The system is being developed with a specific focus on:

1. **Crystallographic Data Processing**

   - PDB structure handling and validation
   - MTZ file processing for experimental data
   - Electron density map generation and analysis
   - Structure factor calculations

2. **Protein Foundation Model Integration**

   - Interface with protein language models
   - Structure prediction and refinement
   - Sequence-structure relationship modeling
   - Multi-modal protein data processing

3. **Structural Biology Workflows**
   - Automated structure determination pipelines
   - Model building and refinement
   - Validation and quality assessment
   - Collaborative structure analysis

### Future Directions

Planned development areas include:

- Enhanced integration with protein foundation models (ESMFold, AlphaFold, etc.)
- Advanced crystallographic data processing capabilities
- Improved performance and scalability
- Extended API stability and documentation

## Architecture

### Core Components

1. **AsyncGeometryAgent** - Main background service for geometry calculations
2. **AgentSecurityManager** - Security and permission management
3. **CoordinateTranslator** - PyTorch module for coordinate conversion
4. **RedisStreamManager** - Reliable message processing with consumer groups

### Data Flow

```
PDB + MTZ Files → Macromolecule Bundle → Geometry Agent → Geometry Gradients
     ↓                    ↓                    ↓              ↓
Redis Storage ← Bundle System ← Async Processing ← CCTBX Geometry
```

## Features

### 🚀 Async Processing

- Non-blocking geometry calculations
- Redis streams for reliable messaging
- Consumer groups for load balancing
- Automatic retry logic with exponential backoff

### 🔒 Security & Permissions

- Agent registration and validation
- Permission-based access control
- Module import whitelisting
- Security audit logging
- Violation detection and reporting

### 🔧 Coordinate Translation

- Seamless CCTBX ↔ PyTorch conversion
- Automatic gradient registration
- Redis bundle integration
- Memory-efficient operations

### 📊 Monitoring & Metrics

- Real-time performance metrics
- Health check monitoring
- Stream processing statistics
- Security violation tracking

## Installation

> **Note**: This is an experimental system. Please ensure you have a development environment set up and be prepared for potential API changes.

### Prerequisites

- Python 3.8+
- Redis server
- CCTBX installation
- PyTorch (optional, for coordinate translation)

### Dependencies

```bash
pip install redis asyncio pydantic click torch numpy
```

## Quick Start

### 1. Start Redis Server

```bash
redis-server
```

### 2. Register a Geometry Agent

```python
from agentbx.core.agent_security_manager import AgentSecurityManager, AgentRegistration
from agentbx.core.redis_manager import RedisManager

# Initialize Redis manager
redis_manager = RedisManager(host="localhost", port=6379)

# Create security manager
security_manager = AgentSecurityManager(redis_manager)

# Register agent
registration = AgentRegistration(
    agent_id="geometry_agent_1",
    agent_name="My Geometry Agent",
    agent_type="geometry_agent",
    version="1.0.0",
    permissions=["geometry_calculation", "bundle_read", "bundle_write"]
)

success = security_manager.register_agent(registration)
```

### 3. Start the Agent

```python
from agentbx.core.async_geometry_agent import AsyncGeometryAgent
import asyncio

async def start_agent():
    # Create agent
    agent = AsyncGeometryAgent(
        agent_id="geometry_agent_1",
        redis_manager=redis_manager,
        stream_name="geometry_requests",
        consumer_group="geometry_agents"
    )

    # Initialize and start
    await agent.initialize()
    await agent.start()

# Run agent
asyncio.run(start_agent())
```

### 4. Send Geometry Request

```python
from agentbx.core.async_geometry_agent import GeometryRequest
import redis.asyncio as redis

async def send_request():
    # Create Redis client
    redis_client = redis.Redis(decode_responses=True)

    # Create request
    request = GeometryRequest(
        request_id="req_123",
        macromolecule_bundle_id="bundle_456",
        priority=1,
        timeout_seconds=300
    )

    # Send to stream
    await redis_client.xadd(
        "geometry_requests",
        {"request": json.dumps(request.__dict__)}
    )

asyncio.run(send_request())
```

## CLI Usage

The system includes a comprehensive CLI tool for management:

### Start an Agent

```bash
python -m agentbx.utils.geometry_agent_cli start-agent \
    --agent-id geometry_agent_1 \
    --stream-name geometry_requests \
    --consumer-group geometry_agents
```

### Send a Request

```bash
python -m agentbx.utils.geometry_agent_cli send-request \
    --agent-id geometry_agent_1 \
    --macromolecule-bundle-id bundle_123
```

### Check Status

```bash
python -m agentbx.utils.geometry_agent_cli status
```

### Register Agent

```bash
python -m agentbx.utils.geometry_agent_cli register-agent \
    --agent-id geometry_agent_1 \
    --agent-name "My Geometry Agent" \
    --permissions geometry_calculation bundle_read bundle_write
```

## Coordinate Translation

The `CoordinateTranslator` provides seamless conversion between CCTBX and PyTorch:

```python
from agentbx.core.coordinate_translator import CoordinateTranslator
import torch

# Create translator
translator = CoordinateTranslator(
    redis_manager=redis_manager,
    coordinate_system="cartesian",
    requires_grad=True
)

# Convert CCTBX coordinates to PyTorch tensor
cctbx_coords = model_manager.get_sites_cart()
tensor_coords = translator.cctbx_to_torch(cctbx_coords)

# Perform computations
energy = torch.sum(tensor_coords ** 2)
energy.backward()

# Convert gradients back to CCTBX
gradients = translator.torch_to_cctbx(tensor_coords.grad)

# Register as Redis bundle
bundle_id = translator.register_bundle("coords_123", tensor_coords)
```

## Security Features

### Agent Registration

Agents must be registered with specific permissions:

```python
registration = AgentRegistration(
    agent_id="geometry_agent_1",
    agent_name="Geometry Agent",
    agent_type="geometry_agent",
    version="1.0.0",
    permissions=[
        "geometry_calculation",
        "bundle_read",
        "bundle_write",
        "coordinate_update"
    ]
)
```

### Permission Checking

```python
# Check if agent has permission
has_permission = security_manager.check_permission(
    agent_id="geometry_agent_1",
    permission="geometry_calculation"
)
```

### Security Monitoring

```python
# Get security report
report = security_manager.get_agent_security_report("geometry_agent_1")

# Get all violations
violations = security_manager.get_all_violations()
```

## Redis Stream Patterns

### Reliable Message Processing

The system uses Redis streams with consumer groups for reliable processing:

```python
from agentbx.core.redis_stream_manager import RedisStreamManager, MessageHandler

# Create stream manager
stream_manager = RedisStreamManager(
    redis_client=redis_client,
    stream_name="geometry_requests",
    consumer_group="geometry_agents",
    consumer_name="agent_1"
)

# Register custom handler
async def geometry_handler(message):
    # Process geometry calculation
    return {"status": "success"}

handler = MessageHandler(
    handler_name="geometry_calculation",
    handler_func=geometry_handler,
    timeout_seconds=300
)

stream_manager.register_handler(handler)
```

### Dead Letter Queue

Failed messages are automatically sent to a dead letter queue:

```python
# Clean up old failed messages
await stream_manager.cleanup_dead_letter_queue(max_age_hours=24)
```

## Integration with Workflow Engines

### Prefect Integration

```python
from prefect import flow, task
from agentbx.core.async_geometry_agent import AsyncGeometryAgent

@task
def create_macromolecule_bundle(pdb_file, mtz_file):
    # Create macromolecule bundle
    return bundle_id

@task
def send_geometry_request(bundle_id):
    # Send geometry calculation request
    return geometry_bundle_id

@task
def process_results(geometry_bundle_id):
    # Process geometry results
    return results

@flow
def geometry_workflow(pdb_file, mtz_file):
    bundle_id = create_macromolecule_bundle(pdb_file, mtz_file)
    geometry_bundle_id = send_geometry_request(bundle_id)
    results = process_results(geometry_bundle_id)
    return results
```

### LangGraph Integration

```python
from langgraph.graph import StateGraph
from agentbx.core.async_geometry_agent import AsyncGeometryAgent

def create_geometry_workflow():
    workflow = StateGraph()

    # Add nodes
    workflow.add_node("create_bundle", create_macromolecule_bundle)
    workflow.add_node("calculate_geometry", send_geometry_request)
    workflow.add_node("process_results", process_geometry_results)

    # Add edges
    workflow.add_edge("create_bundle", "calculate_geometry")
    workflow.add_edge("calculate_geometry", "process_results")

    return workflow.compile()
```

## Monitoring and Metrics

### Agent Metrics

```python
# Get agent statistics
stats = agent.get_stats()
print(f"Requests processed: {stats['requests_processed']}")
print(f"Requests failed: {stats['requests_failed']}")
print(f"Average processing time: {stats['total_processing_time'] / stats['requests_processed']:.2f}s")
```

### Stream Metrics

```python
# Get stream processing metrics
metrics = stream_manager.get_metrics()
print(f"Messages processed: {metrics.messages_processed}")
print(f"Error rate: {metrics.error_rate:.2%}")
print(f"Average processing time: {metrics.average_processing_time:.2f}s")
```

### Health Monitoring

```python
# Check agent health
status = await redis_client.hgetall(f"agentbx:agents:{agent_id}")
print(f"Status: {status.get('status')}")
print(f"Last heartbeat: {status.get('last_heartbeat')}")
```

## Configuration

### Agent Configuration

```yaml
# agent_configuration.yaml
agent_definition:
  agent_id: "geometry_agent_1"
  agent_name: "Geometry Agent"
  agent_type: "geometry_agent"
  version: "1.0.0"

capabilities:
  - name: "geometry_gradient_calculation"
    description: "Calculate geometry gradients"
    timeout_seconds: 300
    max_retries: 3

resource_limits:
  max_memory_mb: 1024
  max_cpu_percent: 80
  max_execution_time: 300
```

### Security Configuration

```yaml
# agent_security.yaml
permissions:
  - "geometry_calculation"
  - "bundle_read"
  - "bundle_write"
  - "coordinate_update"

whitelisted_modules:
  - "cctbx"
  - "mmtbx"
  - "agentbx.processors"
  - "torch"
  - "numpy"

security_policies:
  max_execution_time: 300
  memory_limit_mb: 1024
  network_access: false
  sandbox_mode: true
```

## Error Handling

### Retry Logic

The system implements exponential backoff retry logic:

```python
retry_policy = RetryPolicy(
    max_retries=3,
    initial_delay_ms=1000,
    backoff_multiplier=2.0,
    max_delay_ms=30000
)
```

### Error Recovery

```python
# Handle failed messages
async def handle_failed_message(message, error):
    # Log error
    logger.error(f"Message {message.message_id} failed: {error}")

    # Send to dead letter queue
    await send_to_dead_letter_queue(message, error)

    # Notify monitoring system
    await notify_monitoring_system(message, error)
```

## Performance Optimization

### Memory Management

```python
# Use gradient checkpointing for large models
translator = CoordinateTranslator(
    redis_manager=redis_manager,
    requires_grad=True,
    device=torch.device("cuda")  # Use GPU if available
)

# Monitor memory usage
memory_info = translator.get_memory_usage()
print(f"GPU memory: {memory_info['cuda_memory_allocated']} bytes")
```

### Batch Processing

```python
# Process multiple requests in batch
async def process_batch(requests):
    # Group similar requests
    grouped_requests = group_requests_by_type(requests)

    # Process each group
    results = []
    for group in grouped_requests:
        result = await process_group(group)
        results.extend(result)

    return results
```

## Troubleshooting

### Common Issues

1. **Agent not responding**

   - Check Redis connection
   - Verify agent is registered
   - Check permissions

2. **Messages not being processed**

   - Verify consumer group exists
   - Check stream configuration
   - Monitor dead letter queue

3. **Performance issues**
   - Monitor memory usage
   - Check CPU utilization
   - Review retry policies

### Debug Commands

```bash
# Check agent status
python -m agentbx.utils.geometry_agent_cli status

# Inspect stream
python -m agentbx.utils.geometry_agent_cli stream-info

# Check security violations
python -m agentbx.utils.geometry_agent_cli security-report

# List bundles
python -m agentbx.utils.geometry_agent_cli list-bundles
```

## API Reference

### AsyncGeometryAgent

```python
class AsyncGeometryAgent:
    def __init__(self, agent_id, redis_manager, stream_name, consumer_group)
    async def initialize() -> None
    async def start() -> None
    async def stop() -> None
    def get_stats() -> Dict[str, Any]
```

### AgentSecurityManager

```python
class AgentSecurityManager:
    def register_agent(registration: AgentRegistration) -> bool
    def check_permission(agent_id: str, permission: str) -> bool
    def get_agent_security_report(agent_id: str) -> Dict[str, Any]
    def get_all_violations() -> List[SecurityViolation]
```

### CoordinateTranslator

```python
class CoordinateTranslator(nn.Module):
    def cctbx_to_torch(cctbx_array: Any) -> torch.Tensor
    def torch_to_cctbx(tensor: torch.Tensor) -> Any
    def register_bundle(bundle_id: str, tensor: torch.Tensor) -> str
    def load_bundle(bundle_id: str) -> torch.Tensor
```

### RedisStreamManager

```python
class RedisStreamManager:
    async def initialize() -> None
    def register_handler(handler: MessageHandler) -> None
    async def start_processing() -> None
    async def stop_processing() -> None
    def get_metrics() -> StreamMetrics
```

## Contributing

> **Experimental Development**: This system is in active development. When contributing, please be aware that the codebase is evolving and your contributions may need updates as the system matures.

### Development Setup

1. Clone the repository
2. Install dependencies: `pip install -e .`
3. Start Redis server
4. Run tests: `python -m pytest tests/`

### Adding New Agents

1. Create agent class extending `AsyncGeometryAgent`
2. Implement required methods
3. Add security permissions
4. Create tests
5. Update documentation

### Security Guidelines

1. Always validate input data
2. Use least privilege principle
3. Monitor for security violations
4. Regular security audits
5. Keep dependencies updated

## License

This project is licensed under the MIT License - see the LICENSE file for details.
