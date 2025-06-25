"""
Utility functions for Redis operations.
"""

from typing import Optional

import click

from ..core.config import RedisConfig
from ..core.redis_manager import RedisManager


@click.group()
def redis_cli() -> None:
    """Redis management utilities for agentbx."""


@redis_cli.command()
@click.option("--host", default="localhost", help="Redis host")
@click.option("--port", default=6379, help="Redis port")
@click.option("--db", default=0, help="Redis database")
@click.option("--password", help="Redis password")
def test_connection(host: str, port: int, db: int, password: Optional[str]) -> None:
    """Test Redis connection."""
    try:
        config = RedisConfig(host=host, port=port, db=db, password=password)
        redis_manager = RedisManager(**config.__dict__)

        if redis_manager.is_healthy():
            click.echo("✅ Redis connection successful!")
        else:
            click.echo("❌ Redis connection failed!")

    except Exception as e:
        click.echo(f"❌ Redis connection error: {e}")


@redis_cli.command()
@click.option("--host", default="localhost", help="Redis host")
@click.option("--port", default=6379, help="Redis port")
@click.option("--db", default=0, help="Redis database")
@click.option("--password", help="Redis password")
def list_bundles(host: str, port: int, db: int, password: Optional[str]) -> None:
    """List all bundles in Redis."""
    try:
        config = RedisConfig(host=host, port=port, db=db, password=password)
        redis_manager = RedisManager(**config.__dict__)

        bundles = redis_manager.list_bundles()
        if bundles:
            click.echo(f"Found {len(bundles)} bundles:")
            for bundle_id in bundles:
                click.echo(f"  - {bundle_id}")
        else:
            click.echo("No bundles found.")

    except Exception as e:
        click.echo(f"❌ Error listing bundles: {e}")


@redis_cli.command()
@click.argument("bundle_id")
@click.option("--host", default="localhost", help="Redis host")
@click.option("--port", default=6379, help="Redis port")
@click.option("--db", default=0, help="Redis database")
@click.option("--password", help="Redis password")
def show_bundle(
    bundle_id: str, host: str, port: int, db: int, password: Optional[str]
) -> None:
    """Show bundle details."""
    try:
        config = RedisConfig(host=host, port=port, db=db, password=password)
        redis_manager = RedisManager(**config.__dict__)

        bundle = redis_manager.get_bundle(bundle_id)
        click.echo(f"Bundle ID: {bundle_id}")
        click.echo(f"Type: {bundle.bundle_type}")
        click.echo(f"Created: {bundle.created_at}")
        click.echo(f"Assets: {list(bundle.assets.keys())}")
        click.echo(f"Size estimate: {bundle.get_size_estimate()} bytes")

    except Exception as e:
        click.echo(f"❌ Error showing bundle: {e}")


@redis_cli.command()
@click.option("--host", default="localhost", help="Redis host")
@click.option("--port", default=6379, help="Redis port")
@click.option("--db", default=0, help="Redis database")
@click.option("--password", help="Redis password")
@click.confirmation_option(prompt="Are you sure you want to clear all agentbx data?")
def clear_all(host: str, port: int, db: int, password: Optional[str]) -> None:
    """Clear all agentbx data from Redis."""
    try:
        config = RedisConfig(host=host, port=port, db=db, password=password)
        redis_manager = RedisManager(**config.__dict__)

        bundles = redis_manager.list_bundles()
        deleted_count = 0

        for bundle_id in bundles:
            if redis_manager.delete_bundle(bundle_id):
                deleted_count += 1

        click.echo(f"✅ Deleted {deleted_count} bundles.")

    except Exception as e:
        click.echo(f"❌ Error clearing data: {e}")


if __name__ == "__main__":
    redis_cli()
