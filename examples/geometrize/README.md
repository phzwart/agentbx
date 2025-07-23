# Async Geometry Agent System - Worked Example

This document provides a comprehensive guide to the Async Geometry Agent System, a sophisticated workflow for background geometry calculations using CCTBX, Redis streams, and PyTorch optimization.

## 🏗️ System Architecture

### Core Components

1. **AsyncGeometryAgent** - Background service for geometry calculations
2. **GeometryMinimizer** - PyTorch-style optimizer with Redis integration
3. **CctbxGeometryProcessor** - CCTBX-based geometry gradient computation
4. **MacromoleculeProcessor** - PDB file loading and bundle creation
5. **Redis Streams** - Asynchronous message passing system

### Data Flow Overview

```
PDB File → Macromolecule Bundle → Geometry Agent → Geometry Gradients → Minimizer → Updated Coordinates
    ↓              ↓                    ↓              ↓                    ↓              ↓
Redis Storage ← Bundle System ← Async Processing ← CCTBX Geometry ← PyTorch Optimizer ← Coordinate Updates
```

## 🔄 Redis Streams Communication

### Primary Streams

#### 1. `geometry_requests` Stream

**Purpose**: Main communication channel for geometry calculation requests and coordinate updates

**Message Types**:

**A. Geometry Calculation Request**

```json
{
  "request": "{\"request_id\": \"uuid\", \"macromolecule_bundle_id\": \"bundle_id\", \"priority\": 1, \"refresh_restraints\": false}",
  "timestamp": 1234567890.123,
  "source": "geometry_minimizer"
}
```

**B. Coordinate Update Message**

```json
{
  "coordinate_update": "{\"type\": \"coordinate_update\", \"bundle_id\": \"coord_bundle_id\", \"parent_bundle_id\": \"macromolecule_bundle_id\", \"step\": 42, \"timestamp\": 1234567890.123, \"dialect\": \"numpy\"}",
  "timestamp": 1234567890.123,
  "source": "geometry_minimizer"
}
```

#### 2. `geometry_requests_responses` Stream

**Purpose**: Response channel for geometry calculation results

**Message Format**:

```json
{
  "response": "{\"request_id\": \"uuid\", \"success\": true, \"geometry_bundle_id\": \"result_bundle_id\", \"processing_time\": 1.234, \"timestamp\": \"2024-01-01T12:00:00\"}",
  "timestamp": "2024-01-01T12:00:00",
  "agent_id": "test_agent"
}
```

### Consumer Groups

#### 1. `geometry_agents` Consumer Group

- **Purpose**: Load balancing for multiple geometry agents
- **Consumers**: Multiple AsyncGeometryAgent instances
- **Stream**: `geometry_requests`
- **Behavior**: Round-robin message distribution

#### 2. `minimizer_consumer` Consumer Group

- **Purpose**: Geometry minimizer response handling
- **Consumers**: GeometryMinimizer instances
- **Stream**: `geometry_requests_responses`
- **Behavior**: Dedicated response processing

### Stream Configuration

**No More Magic Constants!** The system now uses configurable stream parameters:

```python
# Centralized stream configuration
STREAM_CONFIG = {
    "request_stream_name": "geometry_requests",
    "response_stream_name": "geometry_requests_responses",
    "agent_consumer_group": "geometry_agents",
    "minimizer_consumer_group": "minimizer_consumer",
}

# Agent configuration
agent = AsyncGeometryAgent(
    agent_id="test_agent",
    redis_manager=redis_manager,
    stream_name=STREAM_CONFIG["request_stream_name"],
    consumer_group=STREAM_CONFIG["agent_consumer_group"],
)

# Minimizer configuration
minimizer = GeometryMinimizer(
    redis_manager=redis_manager,
    macromolecule_bundle_id=bundle_id,
    # ... other parameters ...
    request_stream_name=STREAM_CONFIG["request_stream_name"],
    response_stream_name=STREAM_CONFIG["response_stream_name"],
    consumer_group=STREAM_CONFIG["minimizer_consumer_group"],
    consumer_name=None,  # Auto-generated
)
```

## 📦 Bundle System

### Bundle Types

#### 1. Macromolecule Bundle (`macromolecule_data`)

**Purpose**: Stores atomic model and associated data

**Assets**:

- `model_manager`: CCTBX model manager with atomic model
- `xray_structure`: CCTBX xray structure object
- `coordinates`: Atomic coordinates (numpy/cctbx dialect)
- `restraint_manager`: Geometry restraints manager
- `metadata`: Bundle metadata

**Dialect**: `cctbx` (default) or `numpy`

#### 2. Geometry Gradient Bundle (`geometry_gradient_data`)

**Purpose**: Stores computed geometry gradients and energies

**Assets**:

- `geometry_gradients`: Gradient array (numpy bytes)
- `shape`: Gradient array shape
- `dtype`: Gradient array data type
- `parent_bundle_id`: Source macromolecule bundle ID
- `total_geometry_energy`: Total geometry energy value
- `gradient_norm`: Gradient magnitude

**Dialect**: `numpy` (packed as bytes)

#### 3. Coordinate Update Bundle (`coordinate_update`)

**Purpose**: Stores updated coordinates for agent communication

**Assets**:

- `coordinates`: Updated coordinates as list
- `parent_bundle_id`: Parent macromolecule bundle ID
- `step`: Optimization step number
- `timestamp`: Update timestamp

**Dialect**: `numpy` (as list)

## 🔧 Implementation Details

### 1. AsyncGeometryAgent

**Key Features**:

- Background service with Redis stream consumption
- CCTBX geometry gradient computation
- Bundle caching for performance
- Error handling and retry logic
- Security and permission validation

**Initialization**:

```python
agent = AsyncGeometryAgent(
    agent_id="test_agent",
    redis_manager=redis_manager,
    stream_name="geometry_requests",
    consumer_group="geometry_agents",
    consumer_name="test_agent_12345678"
)
```

**Processing Loop**:

1. Polls `geometry_requests` stream for messages
2. Processes geometry requests or coordinate updates
3. Sends responses to `geometry_requests_responses` stream
4. Maintains health check and statistics

### 2. GeometryMinimizer

**Key Features**:

- PyTorch-style optimization interface
- Redis bundle integration
- Multiple optimizer support (Adam, SGD)
- Convergence monitoring
- Coordinate translation (CCTBX ↔ PyTorch)

**Optimization Loop**:

1. **Forward Pass**: Request geometry gradients from agent
2. **Backward Pass**: Update coordinates using gradients
3. **Coordinate Update**: Send updated coordinates to agent
4. **Convergence Check**: Monitor gradient norm

**Coordinate Translation**:

```python
# CCTBX to PyTorch
translator = ArrayTranslator()
coordinates_tensor = translator.convert(cctbx_coordinates, "torch", requires_grad=True)

# PyTorch to CCTBX
coordinates_cctbx = translator.convert(pytorch_coordinates, "cctbx")
```

### 3. CctbxGeometryProcessor

**Key Features**:

- CCTBX geometry restraint computation
- Gradient and energy calculation
- Bundle creation and storage
- Dialect-aware data handling

**Processing**:

1. Extract model manager from macromolecule bundle
2. Compute geometry gradients using CCTBX restraints
3. Create gradient bundle with metadata
4. Store bundle in Redis

## 🚀 Workflow Execution

### Step-by-Step Process

#### 1. System Initialization

```python
# Initialize Redis manager
redis_manager = RedisManager(host="localhost", port=6379, db=0)

# Start async geometry agent
agent = AsyncGeometryAgent(agent_id="test_agent", redis_manager=redis_manager)
await agent.initialize()
agent_task = asyncio.create_task(agent.start())
```

#### 2. Data Preparation

```python
# Load PDB file and create macromolecule bundle
processor = MacromoleculeProcessor(redis_manager, "workflow_processor")
macromolecule_bundle_id = processor.create_macromolecule_bundle(pdb_path)

# Optionally shake coordinates for testing
shaken_bundle = shake_coordinates_in_bundle(macromolecule_bundle, magnitude=1.00)
shaken_bundle_id = redis_manager.store_bundle(shaken_bundle)
```

#### 3. Minimization Setup

```python
# Create geometry minimizer with configured streams
minimizer = GeometryMinimizer(
    redis_manager=redis_manager,
    macromolecule_bundle_id=shaken_bundle_id,
    learning_rate=0.1,
    optimizer="adam",
    max_iterations=1000,
    convergence_threshold=1e-6,
    timeout_seconds=3.0,
    # Stream configuration - no more magic constants!
    request_stream_name=STREAM_CONFIG["request_stream_name"],
    response_stream_name=STREAM_CONFIG["response_stream_name"],
    consumer_group=STREAM_CONFIG["minimizer_consumer_group"],
    consumer_name=None,  # Auto-generated
)
```

#### 4. Optimization Execution

```python
# Run minimization loop
results = await minimizer.minimize(refresh_restraints=False)
```

### Message Flow Example

#### Iteration 1: Initial Geometry Calculation

1. **Minimizer → Agent** (via `geometry_requests`):

```json
{
  "request": "{\"request_id\": \"req_123\", \"macromolecule_bundle_id\": \"bundle_456\", \"priority\": 1, \"refresh_restraints\": false}",
  "timestamp": 1234567890.123,
  "source": "geometry_minimizer"
}
```

2. **Agent Processing**:
   - Load macromolecule bundle
   - Compute geometry gradients using CCTBX
   - Create gradient bundle
   - Store bundle in Redis

3. **Agent → Minimizer** (via `geometry_requests_responses`):

```json
{
  "response": "{\"request_id\": \"req_123\", \"success\": true, \"geometry_bundle_id\": \"grad_bundle_789\", \"processing_time\": 0.456, \"timestamp\": \"2024-01-01T12:00:00\"}",
  "timestamp": "2024-01-01T12:00:00",
  "agent_id": "test_agent"
}
```

4. **Minimizer → Agent** (via `geometry_requests`):

```json
{
  "coordinate_update": "{\"type\": \"coordinate_update\", \"bundle_id\": \"coord_bundle_101\", \"parent_bundle_id\": \"bundle_456\", \"step\": 0, \"timestamp\": 1234567890.789, \"dialect\": \"numpy\"}",
  "timestamp": 1234567890.789,
  "source": "geometry_minimizer"
}
```

## 🔍 Monitoring and Debugging

### Redis Stream Inspection

#### Check Stream Information

```bash
# Get stream info
redis-cli XINFO STREAM geometry_requests

# Get consumer group info
redis-cli XINFO GROUPS geometry_requests

# Read recent messages
redis-cli XREAD COUNT 10 STREAMS geometry_requests 0
```

#### Monitor Messages in Real-time

```bash
# Monitor geometry requests
redis-cli XREAD BLOCK 0 STREAMS geometry_requests $

# Monitor responses
redis-cli XREAD BLOCK 0 STREAMS geometry_requests_responses $
```

### Agent Statistics

The AsyncGeometryAgent maintains statistics:

```python
stats = agent.get_stats()
# Returns: {
#   "requests_processed": 42,
#   "requests_failed": 0,
#   "total_processing_time": 12.345,
#   "last_request_time": "2024-01-01T12:00:00",
#   "agent_id": "test_agent",
#   "is_running": true,
#   "consumer_name": "test_agent_12345678"
# }
```

### Minimization Results

The GeometryMinimizer provides detailed results:

```python
results = await minimizer.minimize()
# Returns: {
#   "converged": true,
#   "final_gradient_norm": 1.23e-7,
#   "best_gradient_norm": 1.23e-7,
#   "iterations": 42,
#   "total_time": 12.345,
#   "final_bundle_id": "bundle_789",
#   "iteration_history": [...],
#   "final_total_geometry_energy": 123.456
# }
```

## 🛠️ Configuration Options

### AsyncGeometryAgent Configuration

```python
agent = AsyncGeometryAgent(
    agent_id="test_agent",
    redis_manager=redis_manager,
    stream_name="geometry_requests",           # Custom stream name
    consumer_group="geometry_agents",          # Consumer group
    consumer_name=None,                        # Auto-generated if None
    max_processing_time=300,                   # Max processing time (seconds)
    health_check_interval=30,                  # Health check interval (seconds)
)
```

### GeometryMinimizer Configuration

```python
minimizer = GeometryMinimizer(
    redis_manager=redis_manager,
    macromolecule_bundle_id=bundle_id,
    learning_rate=0.01,                       # Optimization learning rate
    optimizer="adam",                         # "adam" or "gd"
    max_iterations=100,                       # Maximum iterations
    convergence_threshold=1e-6,               # Convergence threshold
    timeout_seconds=30.0,                     # Request timeout
    device=torch.device("cpu"),               # PyTorch device
    dtype=torch.float32,                      # PyTorch data type
    # Stream configuration
    request_stream_name="geometry_requests",   # Stream for sending requests
    response_stream_name="geometry_requests_responses", # Stream for receiving responses
    consumer_group="minimizer_consumer",      # Consumer group for responses
    consumer_name=None,                       # Auto-generated if None
)
```

## 🔒 Security and Permissions

### Agent Security Bundles

The system supports agent security through bundles:

```python
# Security bundle structure
security_bundle = AgentSecurityBundle(
    agent_id="test_agent",
    permissions=["geometry_calculation", "bundle_read", "bundle_write"],
    allowed_modules=["cctbx", "numpy", "torch"],
    max_processing_time=300,
    resource_limits={"memory_mb": 1024, "cpu_percent": 50}
)
```

### Permission Validation

The AsyncGeometryAgent validates permissions before processing:

- `geometry_calculation`: Can compute geometry gradients
- `bundle_read`: Can read macromolecule bundles
- `bundle_write`: Can create gradient bundles

## 🚨 Error Handling

### Common Error Scenarios

1. **Redis Connection Issues**:
   - Automatic reconnection attempts
   - Graceful shutdown on connection loss
   - Health check monitoring

2. **Geometry Calculation Failures**:
   - Invalid macromolecule bundle
   - CCTBX restraint computation errors
   - Memory allocation issues

3. **Optimization Failures**:
   - Gradient computation timeouts
   - Coordinate update failures
   - Convergence issues

### Error Recovery

```python
# Handle geometry calculation errors
try:
    gradients, bundle_id = await minimizer.forward()
except TimeoutError:
    print("Geometry calculation timed out")
except Exception as e:
    print(f"Geometry calculation failed: {e}")

# Handle optimization errors
try:
    results = await minimizer.minimize()
except Exception as e:
    print(f"Minimization failed: {e}")
    # Access best coordinates found so far
    best_coords = minimizer.get_best_coordinates()
```

## 📊 Performance Considerations

### Optimization Strategies

1. **Bundle Caching**: Agents cache model managers and restraints
2. **Dialect Optimization**: Use appropriate data formats (numpy vs cctbx)
3. **Stream Batching**: Process multiple messages efficiently
4. **Memory Management**: Automatic cleanup of temporary bundles

### Scaling Considerations

1. **Multiple Agents**: Run multiple AsyncGeometryAgent instances
2. **Consumer Groups**: Distribute load across consumers
3. **Redis Clustering**: Use Redis cluster for high availability
4. **Resource Monitoring**: Monitor CPU, memory, and network usage

## 🔧 Troubleshooting

### Common Issues

1. **Agent Not Responding**:
   - Check Redis connection
   - Verify consumer group setup
   - Check agent logs for errors

2. **Slow Performance**:
   - Monitor Redis stream lengths
   - Check bundle sizes and compression
   - Verify network latency

3. **Memory Issues**:
   - Monitor bundle cache sizes
   - Check for memory leaks in CCTBX objects
   - Adjust resource limits

### Debug Commands

```python
# Check agent status
await agent._update_agent_status()

# Monitor stream processing
messages = await redis_client.xreadgroup(
    "geometry_agents", "debug_consumer",
    {"geometry_requests": ">"}, count=10
)

# Inspect bundle contents
bundle = redis_manager.get_bundle(bundle_id)
print(f"Bundle type: {bundle.bundle_type}")
print(f"Assets: {list(bundle.assets.keys())}")
```

## 📚 Further Reading

- [Async Geometry Agent System Documentation](../docs/async_geometry_agent_system.md)
- [Redis Streams Documentation](https://redis.io/docs/data-types/streams/)
- [CCTBX Geometry Restraints](https://cctbx.github.io/)
- [PyTorch Optimization](https://pytorch.org/docs/stable/optim.html)

## 🤝 Contributing

When extending this system:

1. **Message Formats**: Document new message types in this README
2. **Bundle Schemas**: Update YAML schema definitions
3. **Error Handling**: Add appropriate error recovery mechanisms
4. **Testing**: Include integration tests for new features
5. **Documentation**: Update this README with new workflows
