"""
CLI tool for managing async geometry agents.

This module provides command-line interface for:
- Starting and stopping geometry agents
- Sending geometry calculation requests
- Monitoring agent performance and health
- Managing agent security and permissions
"""

import asyncio
import json
import logging
import signal
import sys
import time
from datetime import datetime

import click
import redis.asyncio as redis

from agentbx.core.agents.agent_security_manager import AgentRegistration
from agentbx.core.agents.agent_security_manager import AgentSecurityManager
from agentbx.core.agents.async_geometry_agent import AsyncGeometryAgent
from agentbx.core.agents.async_geometry_agent import GeometryRequest
from agentbx.core.agents.async_geometry_agent import GeometryResponse
from agentbx.core.redis_manager import RedisManager


@click.group()
@click.option("--redis-host", default="localhost", help="Redis host")
@click.option("--redis-port", default=6379, help="Redis port")
@click.option("--redis-db", default=0, help="Redis database")
@click.option("--redis-password", default=None, help="Redis password")
@click.option("--verbose", "-v", is_flag=True, help="Verbose logging")
@click.pass_context
def cli(ctx, redis_host, redis_port, redis_db, redis_password, verbose):
    """Agentbx Geometry Agent CLI"""
    # Setup logging
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Initialize Redis manager
    redis_manager = RedisManager(
        host=redis_host, port=redis_port, db=redis_db, password=redis_password
    )

    ctx.obj = {
        "redis_manager": redis_manager,
        "redis_host": redis_host,
        "redis_port": redis_port,
        "redis_db": redis_db,
        "redis_password": redis_password,
    }


@cli.command()
@click.option("--agent-id", required=True, help="Agent ID")
@click.option("--stream-name", default="geometry_requests", help="Redis stream name")
@click.option("--consumer-group", default="geometry_agents", help="Consumer group name")
@click.option(
    "--max-processing-time", default=300, help="Max processing time in seconds"
)
@click.option(
    "--health-check-interval", default=30, help="Health check interval in seconds"
)
@click.pass_context
def start_agent(
    ctx,
    agent_id,
    stream_name,
    consumer_group,
    max_processing_time,
    health_check_interval,
):
    """Start a geometry agent"""
    redis_manager = ctx.obj["redis_manager"]

    async def run_agent():
        # Create agent
        agent = AsyncGeometryAgent(
            agent_id=agent_id,
            redis_manager=redis_manager,
            stream_name=stream_name,
            consumer_group=consumer_group,
            max_processing_time=max_processing_time,
            health_check_interval=health_check_interval,
        )

        # Initialize agent
        await agent.initialize()

        # Setup signal handlers
        def signal_handler(signum, frame):
            click.echo(f"\nShutting down agent {agent_id}...")
            asyncio.create_task(agent.stop())

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            # Start agent
            click.echo(f"Starting geometry agent {agent_id}...")
            await agent.start()
        except KeyboardInterrupt:
            click.echo("Received interrupt signal")
        except Exception as e:
            click.echo(f"Agent error: {e}")
            sys.exit(1)
        finally:
            await agent.stop()
            click.echo(f"Agent {agent_id} stopped")

    asyncio.run(run_agent())


@cli.command()
@click.option("--agent-id", required=True, help="Agent ID")
@click.option(
    "--macromolecule-bundle-id", required=True, help="Macromolecule bundle ID"
)
@click.option("--priority", default=1, help="Request priority")
@click.option("--timeout", default=300, help="Request timeout in seconds")
@click.option("--stream-name", default="geometry_requests", help="Redis stream name")
@click.pass_context
def send_request(
    ctx, agent_id, macromolecule_bundle_id, priority, timeout, stream_name
):
    """Send a geometry calculation request"""
    ctx.obj["redis_manager"]
    redis_host = ctx.obj["redis_host"]
    redis_port = ctx.obj["redis_port"]
    redis_db = ctx.obj["redis_db"]
    redis_password = ctx.obj["redis_password"]

    async def send_request_async():
        # Create Redis client
        redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
            decode_responses=True,
        )

        # Create request
        request = GeometryRequest(
            request_id=f"req_{int(time.time())}",
            macromolecule_bundle_id=macromolecule_bundle_id,
            priority=priority,
            timeout_seconds=timeout,
        )

        # Send request
        message_id = await redis_client.xadd(
            stream_name,
            {
                "request": json.dumps(request.__dict__),
                "timestamp": datetime.now().isoformat(),
                "agent_id": agent_id,
            },
        )

        click.echo(
            f"Sent geometry request {request.request_id} (message ID: {message_id})"
        )

        # Wait for response
        response_stream = f"{stream_name}_responses"
        click.echo(f"Waiting for response on {response_stream}...")

        # Read response
        while True:
            messages = await redis_client.xread(
                {response_stream: "0"}, count=1, block=1000
            )

            if messages:
                for stream, stream_messages in messages:
                    for msg_id, fields in stream_messages:
                        response_data = json.loads(fields.get("response", "{}"))
                        response = GeometryResponse(**response_data)

                        if response.request_id == request.request_id:
                            if response.success:
                                click.echo("✅ Geometry calculation completed!")
                                click.echo(
                                    f"   Bundle ID: {response.geometry_bundle_id}"
                                )
                                click.echo(
                                    f"   Processing time: {response.processing_time:.2f}s"
                                )
                            else:
                                click.echo("❌ Geometry calculation failed!")
                                click.echo(f"   Error: {response.error_message}")
                                click.echo(
                                    f"   Processing time: {response.processing_time:.2f}s"
                                )

                            await redis_client.close()
                            return

            await asyncio.sleep(0.1)

    asyncio.run(send_request_async())


@cli.command()
@click.option("--agent-id", help="Specific agent ID to check")
@click.pass_context
def status(ctx, agent_id):
    """Check agent status and health"""
    ctx.obj["redis_manager"]
    redis_host = ctx.obj["redis_host"]
    redis_port = ctx.obj["redis_port"]
    redis_db = ctx.obj["redis_db"]
    redis_password = ctx.obj["redis_password"]

    async def check_status():
        redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
            decode_responses=True,
        )

        # Get all agents or specific agent
        if agent_id:
            agent_keys = [f"agentbx:agents:{agent_id}"]
        else:
            agent_keys = await redis_client.keys("agentbx:agents:*")

        if not agent_keys:
            click.echo("No agents found")
            return

        click.echo("Agent Status:")
        click.echo("=" * 50)

        for key in agent_keys:
            agent_data = await redis_client.hgetall(key)
            if agent_data:
                click.echo(f"Agent ID: {agent_data.get('agent_id', 'Unknown')}")
                click.echo(f"Status: {agent_data.get('status', 'Unknown')}")
                click.echo(
                    f"Last Heartbeat: {agent_data.get('last_heartbeat', 'Unknown')}"
                )
                click.echo(f"Consumer: {agent_data.get('consumer_name', 'Unknown')}")

                # Parse stats
                stats_str = agent_data.get("stats", "{}")
                try:
                    stats = json.loads(stats_str)
                    click.echo(
                        f"Requests Processed: {stats.get('requests_processed', 0)}"
                    )
                    click.echo(f"Requests Failed: {stats.get('requests_failed', 0)}")
                    click.echo(
                        f"Total Processing Time: {stats.get('total_processing_time', 0):.2f}s"
                    )
                except (json.JSONDecodeError, TypeError):
                    click.echo("Stats: Unable to parse")
                    click.echo(f"Raw stats: {stats_str}")

                click.echo("-" * 30)

        await redis_client.close()

    asyncio.run(check_status())


@cli.command()
@click.option("--agent-id", required=True, help="Agent ID")
@click.option("--agent-name", required=True, help="Agent name")
@click.option("--agent-type", default="geometry_agent", help="Agent type")
@click.option("--version", default="1.0.0", help="Agent version")
@click.option(
    "--permissions",
    multiple=True,
    default=["geometry_calculation", "bundle_read", "bundle_write"],
    help="Agent permissions",
)
@click.pass_context
def register_agent(ctx, agent_id, agent_name, agent_type, version, permissions):
    """Register an agent with security manager"""
    redis_manager = ctx.obj["redis_manager"]

    # Create security manager
    security_manager = AgentSecurityManager(redis_manager)

    # Create registration
    registration = AgentRegistration(
        agent_id=agent_id,
        agent_name=agent_name,
        agent_type=agent_type,
        version=version,
        permissions=list(permissions),
    )

    # Register agent
    success = security_manager.register_agent(registration)

    if success:
        click.echo(f"✅ Agent {agent_id} registered successfully")
    else:
        click.echo(f"❌ Failed to register agent {agent_id}")
        sys.exit(1)


@cli.command()
@click.option("--agent-id", required=True, help="Agent ID")
@click.pass_context
def unregister_agent(ctx, agent_id):
    """Unregister an agent"""
    redis_manager = ctx.obj["redis_manager"]

    # Create security manager
    security_manager = AgentSecurityManager(redis_manager)

    # Unregister agent
    success = security_manager.unregister_agent(agent_id)

    if success:
        click.echo(f"✅ Agent {agent_id} unregistered successfully")
    else:
        click.echo(f"❌ Failed to unregister agent {agent_id}")


@cli.command()
@click.option("--agent-id", help="Specific agent ID")
@click.pass_context
def security_report(ctx, agent_id):
    """Get security report for agents"""
    redis_manager = ctx.obj["redis_manager"]

    # Create security manager
    security_manager = AgentSecurityManager(redis_manager)

    if agent_id:
        # Get report for specific agent
        report = security_manager.get_agent_security_report(agent_id)
        if "error" in report:
            click.echo(f"❌ {report['error']}")
        else:
            click.echo(f"Security Report for Agent {agent_id}:")
            click.echo("=" * 50)
            click.echo(f"Registration: {json.dumps(report['registration'], indent=2)}")
            click.echo(f"Violations: {report['violation_count']}")
            click.echo(f"Last Activity: {report['last_activity']}")
    else:
        # Get all violations
        violations = security_manager.get_all_violations()
        click.echo(f"Security Violations ({len(violations)}):")
        click.echo("=" * 50)

        for violation in violations:
            click.echo(f"Type: {violation.violation_type}")
            click.echo(f"Agent: {violation.agent_id}")
            click.echo(f"Severity: {violation.severity}")
            click.echo(f"Message: {violation.message}")
            click.echo(f"Time: {violation.timestamp}")
            if violation.details:
                click.echo(f"Details: {json.dumps(violation.details, indent=2)}")
            click.echo("-" * 30)


@cli.command()
@click.option("--stream-name", default="geometry_requests", help="Stream name")
@click.option("--consumer-group", default="geometry_agents", help="Consumer group name")
@click.pass_context
def stream_info(ctx, stream_name, consumer_group):
    """Get stream information and metrics"""
    ctx.obj["redis_manager"]
    redis_host = ctx.obj["redis_host"]
    redis_port = ctx.obj["redis_port"]
    redis_db = ctx.obj["redis_db"]
    redis_password = ctx.obj["redis_password"]

    async def get_stream_info():
        redis_client = redis.Redis(
            host=redis_host,
            port=redis_port,
            db=redis_db,
            password=redis_password,
            decode_responses=True,
        )

        try:
            # Get stream info
            stream_info = await redis_client.xinfo_stream(stream_name)
            click.echo(f"Stream: {stream_name}")
            click.echo("=" * 30)
            click.echo(f"Length: {stream_info['length']}")
            click.echo(f"Groups: {stream_info['groups']}")
            click.echo(f"Last Generated ID: {stream_info['last-generated-id']}")
            click.echo(f"First Entry: {stream_info['first-entry']}")
            click.echo(f"Last Entry: {stream_info['last-entry']}")

            # Get consumer group info
            groups = await redis_client.xinfo_groups(stream_name)
            click.echo("\nConsumer Groups:")
            click.echo("=" * 30)

            for group in groups:
                if group["name"] == consumer_group:
                    click.echo(f"Group: {group['name']}")
                    click.echo(f"Consumers: {group['consumers']}")
                    click.echo(f"Pending: {group['pending']}")
                    click.echo(f"Last Delivered ID: {group['last-delivered-id']}")

                    # Get consumer info
                    consumers = await redis_client.xinfo_consumers(
                        stream_name, consumer_group
                    )
                    click.echo("\nConsumers:")
                    for consumer in consumers:
                        click.echo(
                            f"  {consumer['name']}: {consumer['pending']} pending"
                        )

        except Exception as e:
            click.echo(f"❌ Error getting stream info: {e}")
        finally:
            await redis_client.close()

    asyncio.run(get_stream_info())


@cli.command()
@click.option("--bundle-id", required=True, help="Bundle ID to inspect")
@click.pass_context
def inspect_bundle(ctx, bundle_id):
    """Inspect a bundle in Redis"""
    redis_manager = ctx.obj["redis_manager"]

    try:
        bundle = redis_manager.get_bundle(bundle_id)
        click.echo(f"Bundle: {bundle_id}")
        click.echo("=" * 30)
        click.echo(f"Type: {bundle.bundle_type}")
        click.echo(f"Created: {bundle.created_at}")
        click.echo(f"Assets: {list(bundle.assets.keys())}")
        click.echo(f"Metadata: {bundle.metadata}")

        # Get bundle metadata from Redis
        metadata = redis_manager.get_bundle_metadata(bundle_id)
        click.echo(f"Size: {metadata.get('size_bytes', 'Unknown')} bytes")
        click.echo(f"Checksum: {metadata.get('checksum', 'Unknown')}")

    except KeyError:
        click.echo(f"❌ Bundle {bundle_id} not found")
    except Exception as e:
        click.echo(f"❌ Error inspecting bundle: {e}")


@cli.command()
@click.option("--bundle-type", help="Filter by bundle type")
@click.pass_context
def list_bundles(ctx, bundle_type):
    """List bundles in Redis"""
    redis_manager = ctx.obj["redis_manager"]

    try:
        bundles = redis_manager.list_bundles_with_metadata(bundle_type)

        if not bundles:
            click.echo("No bundles found")
            return

        click.echo(f"Bundles ({len(bundles)}):")
        click.echo("=" * 50)

        for bundle_info in bundles:
            click.echo(f"ID: {bundle_info['bundle_id']}")
            click.echo(f"Type: {bundle_info['bundle_type']}")
            click.echo(f"Created: {bundle_info['created_at']}")
            click.echo(f"Size: {bundle_info['size_bytes']} bytes")
            click.echo("-" * 30)

    except Exception as e:
        click.echo(f"❌ Error listing bundles: {e}")


if __name__ == "__main__":
    cli()
