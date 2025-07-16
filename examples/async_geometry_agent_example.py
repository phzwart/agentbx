"""
Comprehensive example demonstrating the async geometry agent system.

This example shows:
1. Agent registration and security setup
2. Starting async geometry agents
3. Sending geometry calculation requests
4. Coordinate translation with PyTorch
5. Integration with external workflow engines
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any

import torch
import click

from agentbx.core.async_geometry_agent import AsyncGeometryAgent, GeometryRequest
from agentbx.core.agent_security_manager import AgentSecurityManager, AgentRegistration
from agentbx.core.coordinate_translator import CoordinateTranslator
from agentbx.core.redis_manager import RedisManager
from agentbx.core.redis_stream_manager import RedisStreamManager, MessageHandler
from agentbx.processors.geometry_processor import CctbxGeometryProcessor


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


async def create_sample_macromolecule_bundle(redis_manager: RedisManager) -> str:
    """Create a sample macromolecule bundle for testing."""
    from agentbx.core.bundle_base import Bundle
    
    # Create a mock macromolecule bundle
    bundle = Bundle(bundle_type="macromolecule_data")
    
    # Add mock data (in real usage, this would be actual CCTBX objects)
    bundle.add_asset("pdb_hierarchy", "mock_pdb_hierarchy")
    bundle.add_asset("crystal_symmetry", "mock_crystal_symmetry")
    bundle.add_asset("model_manager", "mock_model_manager")
    bundle.add_asset("xray_structure", "mock_xray_structure")
    bundle.add_metadata("source", "example")
    bundle.add_metadata("created_by", "async_geometry_agent_example")
    
    # Store in Redis
    bundle_id = redis_manager.store_bundle(bundle)
    print(f"Created sample macromolecule bundle: {bundle_id}")
    
    return bundle_id


async def register_geometry_agent(redis_manager: RedisManager, agent_id: str) -> bool:
    """Register a geometry agent with security manager."""
    security_manager = AgentSecurityManager(redis_manager)
    
    # Create agent registration
    registration = AgentRegistration(
        agent_id=agent_id,
        agent_name=f"Geometry Agent {agent_id}",
        agent_type="geometry_agent",
        version="1.0.0",
        permissions=[
            "geometry_calculation",
            "bundle_read",
            "bundle_write",
            "coordinate_update"
        ],
        capabilities=[
            {
                "name": "geometry_gradient_calculation",
                "description": "Calculate geometry gradients from macromolecule bundles",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "macromolecule_bundle_id": {"type": "string"}
                    },
                    "required": ["macromolecule_bundle_id"]
                },
                "output_schema": {
                    "type": "object",
                    "properties": {
                        "geometry_bundle_id": {"type": "string"}
                    }
                },
                "timeout_seconds": 300,
                "max_retries": 3
            }
        ]
    )
    
    # Register agent
    success = security_manager.register_agent(registration)
    
    if success:
        print(f"✅ Registered geometry agent: {agent_id}")
    else:
        print(f"❌ Failed to register geometry agent: {agent_id}")
    
    return success


async def start_geometry_agent(redis_manager: RedisManager, agent_id: str) -> AsyncGeometryAgent:
    """Start a geometry agent."""
    # Create agent
    agent = AsyncGeometryAgent(
        agent_id=agent_id,
        redis_manager=redis_manager,
        stream_name="geometry_requests",
        consumer_group="geometry_agents",
        max_processing_time=300,
        health_check_interval=30
    )
    
    # Initialize agent
    await agent.initialize()
    
    # Start agent in background
    asyncio.create_task(agent.start())
    
    print(f"🚀 Started geometry agent: {agent_id}")
    return agent


async def send_geometry_request(redis_manager: RedisManager, macromolecule_bundle_id: str) -> str:
    """Send a geometry calculation request."""
    import redis.asyncio as redis
    
    # Create Redis client
    redis_client = redis.Redis(
        host=redis_manager.host,
        port=redis_manager.port,
        db=redis_manager.db,
        password=redis_manager.password,
        decode_responses=True
    )
    
    # Create request
    request = GeometryRequest(
        request_id=f"req_{int(time.time())}",
        macromolecule_bundle_id=macromolecule_bundle_id,
        priority=1,
        timeout_seconds=300
    )
    
    # Send request
    message_id = await redis_client.xadd(
        "geometry_requests",
        {
            "request": json.dumps(request.__dict__),
            "timestamp": datetime.now().isoformat(),
            "agent_id": "geometry_agent_1"
        }
    )
    
    print(f"📤 Sent geometry request: {request.request_id}")
    
    # Wait for response
    response_stream = "geometry_requests_responses"
    print(f"⏳ Waiting for response on {response_stream}...")
    
    # Read response
    while True:
        messages = await redis_client.xread(
            {response_stream: "0"},
            count=1,
            block=1000
        )
        
        if messages:
            for stream, stream_messages in messages:
                for msg_id, fields in stream_messages:
                    response_data = json.loads(fields.get("response", "{}"))
                    
                    if response_data.get("request_id") == request.request_id:
                        if response_data.get("success"):
                            geometry_bundle_id = response_data.get("geometry_bundle_id")
                            processing_time = response_data.get("processing_time", 0)
                            print(f"✅ Geometry calculation completed!")
                            print(f"   Bundle ID: {geometry_bundle_id}")
                            print(f"   Processing time: {processing_time:.2f}s")
                            
                            await redis_client.close()
                            return geometry_bundle_id
                        else:
                            error = response_data.get("error_message", "Unknown error")
                            print(f"❌ Geometry calculation failed: {error}")
                            
                            await redis_client.close()
                            return None
        
        await asyncio.sleep(0.1)


async def demonstrate_coordinate_translation(redis_manager: RedisManager, geometry_bundle_id: str):
    """Demonstrate coordinate translation with PyTorch."""
    print("\n🔧 Demonstrating coordinate translation...")
    
    # Create coordinate translator
    translator = CoordinateTranslator(
        redis_manager=redis_manager,
        coordinate_system="cartesian",
        requires_grad=True,
        dtype=torch.float32,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    )
    
    # Create sample CCTBX coordinates (mock)
    # In real usage, this would be actual CCTBX flex arrays
    sample_coordinates = torch.randn(100, 3, requires_grad=True)  # 100 atoms, 3 coordinates
    
    # Convert to tensor
    tensor_coordinates = translator.cctbx_to_torch(sample_coordinates)
    print(f"   Converted coordinates shape: {tensor_coordinates.shape}")
    
    # Register tensor as bundle
    coordinate_bundle_id = translator.register_bundle(
        f"coordinates_{int(time.time())}",
        tensor_coordinates
    )
    print(f"   Registered coordinate bundle: {coordinate_bundle_id}")
    
    # Demonstrate gradient computation
    if tensor_coordinates.requires_grad:
        # Simulate some computation
        energy = torch.sum(tensor_coordinates ** 2)
        energy.backward()
        
        gradients = tensor_coordinates.grad
        print(f"   Computed gradients shape: {gradients.shape}")
        
        # Convert back to CCTBX format
        cctbx_gradients = translator.torch_to_cctbx(gradients)
        print(f"   Converted gradients back to CCTBX format")
    
    # Get conversion history
    history = translator.get_conversion_history()
    print(f"   Conversion history: {len(history)} entries")
    
    # Get memory usage
    memory_info = translator.get_memory_usage()
    print(f"   Memory usage: {memory_info}")


async def demonstrate_stream_management(redis_manager: RedisManager):
    """Demonstrate Redis stream management."""
    print("\n📊 Demonstrating stream management...")
    
    import redis.asyncio as redis
    
    # Create Redis client
    redis_client = redis.Redis(
        host=redis_manager.host,
        port=redis_manager.port,
        db=redis_manager.db,
        password=redis_manager.password,
        decode_responses=True
    )
    
    # Create stream manager
    stream_manager = RedisStreamManager(
        redis_client=redis_client,
        stream_name="geometry_requests",
        consumer_group="geometry_agents",
        consumer_name="example_consumer"
    )
    
    # Initialize stream manager
    await stream_manager.initialize()
    
    # Register a custom handler
    async def custom_geometry_handler(message):
        print(f"   Custom handler processing: {message.message_id}")
        return {"status": "processed"}
    
    handler = MessageHandler(
        handler_name="custom_geometry",
        handler_func=custom_geometry_handler,
        timeout_seconds=60
    )
    
    stream_manager.register_handler(handler)
    
    # Get stream info
    stream_info = await stream_manager.get_stream_info()
    print(f"   Stream info: {len(stream_info.get('stream_info', {}))} fields")
    
    # Get metrics
    metrics = stream_manager.get_metrics()
    print(f"   Stream metrics: {metrics.messages_processed} processed, {metrics.messages_failed} failed")
    
    await redis_client.close()


async def demonstrate_workflow_integration(redis_manager: RedisManager):
    """Demonstrate integration with external workflow engines."""
    print("\n🔄 Demonstrating workflow integration...")
    
    # Simulate Prefect workflow
    try:
        from prefect import flow, task
        
        @task
        def create_macromolecule_bundle():
            """Create macromolecule bundle task."""
            return asyncio.run(create_sample_macromolecule_bundle(redis_manager))
        
        @task
        def send_geometry_request(bundle_id: str):
            """Send geometry request task."""
            return asyncio.run(send_geometry_request(redis_manager, bundle_id))
        
        @task
        def process_geometry_results(geometry_bundle_id: str):
            """Process geometry results task."""
            print(f"   Processing geometry results: {geometry_bundle_id}")
            return geometry_bundle_id
        
        @flow
        def geometry_workflow():
            """Complete geometry calculation workflow."""
            print("   Starting Prefect geometry workflow...")
            
            # Create macromolecule bundle
            bundle_id = create_macromolecule_bundle()
            
            # Send geometry request
            geometry_bundle_id = send_geometry_request(bundle_id)
            
            # Process results
            if geometry_bundle_id:
                result = process_geometry_results(geometry_bundle_id)
                print(f"   Workflow completed: {result}")
                return result
            else:
                print("   Workflow failed")
                return None
        
        # Run workflow
        result = geometry_workflow()
        print(f"   Prefect workflow result: {result}")
        
    except ImportError:
        print("   Prefect not available, skipping workflow integration")
    
    # Simulate LangGraph workflow
    try:
        # This would be actual LangGraph code
        print("   LangGraph integration would be implemented here")
        
    except Exception as e:
        print(f"   LangGraph integration error: {e}")


async def demonstrate_security_monitoring(redis_manager: RedisManager, agent_id: str):
    """Demonstrate security monitoring and auditing."""
    print("\n🔒 Demonstrating security monitoring...")
    
    security_manager = AgentSecurityManager(redis_manager)
    
    # Get security report
    report = security_manager.get_agent_security_report(agent_id)
    
    if "error" not in report:
        print(f"   Agent {agent_id} security report:")
        print(f"     Permissions: {report['registration']['permissions']}")
        print(f"     Violations: {report['violation_count']}")
        print(f"     Last Activity: {report['last_activity']}")
    else:
        print(f"   No security report for agent {agent_id}")
    
    # Check permissions
    permissions_to_check = [
        "geometry_calculation",
        "bundle_read",
        "bundle_write",
        "invalid_permission"
    ]
    
    for permission in permissions_to_check:
        has_permission = security_manager.check_permission(agent_id, permission)
        status = "✅" if has_permission else "❌"
        print(f"   {status} Permission '{permission}': {has_permission}")
    
    # Get all violations
    violations = security_manager.get_all_violations()
    print(f"   Total security violations: {len(violations)}")


async def main():
    """Main example function."""
    print("🚀 AgentBX Async Geometry Agent System Demo")
    print("=" * 50)
    
    # Setup
    setup_logging(verbose=True)
    
    # Initialize Redis manager
    redis_manager = RedisManager(
        host="localhost",
        port=6379,
        db=0
    )
    
    agent_id = "geometry_agent_1"
    
    try:
        # 1. Register agent
        print("\n1️⃣ Registering geometry agent...")
        await register_geometry_agent(redis_manager, agent_id)
        
        # 2. Start agent
        print("\n2️⃣ Starting geometry agent...")
        agent = await start_geometry_agent(redis_manager, agent_id)
        
        # Wait for agent to initialize
        await asyncio.sleep(2)
        
        # 3. Create sample data
        print("\n3️⃣ Creating sample macromolecule bundle...")
        macromolecule_bundle_id = await create_sample_macromolecule_bundle(redis_manager)
        
        # 4. Send geometry request
        print("\n4️⃣ Sending geometry calculation request...")
        geometry_bundle_id = await send_geometry_request(redis_manager, macromolecule_bundle_id)
        
        if geometry_bundle_id:
            # 5. Demonstrate coordinate translation
            await demonstrate_coordinate_translation(redis_manager, geometry_bundle_id)
        
        # 6. Demonstrate stream management
        await demonstrate_stream_management(redis_manager)
        
        # 7. Demonstrate workflow integration
        await demonstrate_workflow_integration(redis_manager)
        
        # 8. Demonstrate security monitoring
        await demonstrate_security_monitoring(redis_manager, agent_id)
        
        print("\n✅ Demo completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        if 'agent' in locals():
            await agent.stop()


if __name__ == "__main__":
    asyncio.run(main()) 