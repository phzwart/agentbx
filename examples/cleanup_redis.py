"""
Cleanup script for Redis data management.
"""

import logging

import click

from agentbx.core.redis_manager import RedisManager


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--list-only", "-l", is_flag=True, help="Only list bundles, do not delete"
)
@click.option(
    "--keep-latest", "-k", is_flag=True, help="Keep only the latest bundle of each type"
)
@click.option("--force", "-f", is_flag=True, help="Force deletion without confirmation")
def main(list_only, keep_latest, force):
    """
    Clean up Redis data from agentbx examples.
    """

    try:
        redis_manager = RedisManager()

        if not redis_manager.is_healthy():
            click.echo("❌ Redis connection failed")
            return

        # List all bundles
        all_bundles = redis_manager.list_bundles()
        click.echo(f"Found {len(all_bundles)} bundles in Redis:")

        for bundle_id in all_bundles:
            try:
                bundle = redis_manager.get_bundle(bundle_id)
                click.echo(f"  {bundle_id}: {bundle.bundle_type} ({bundle.created_at})")
            except Exception as e:
                click.echo(f"  {bundle_id}: Error reading bundle - {e}")

        if list_only:
            return

        if not force:
            if not click.confirm(f"Delete all {len(all_bundles)} bundles?"):
                click.echo("Cancelled.")
                return

        # Delete bundles
        deleted_count = 0
        for bundle_id in all_bundles:
            if redis_manager.delete_bundle(bundle_id):
                deleted_count += 1
                click.echo(f"Deleted: {bundle_id}")

        click.echo(f"✅ Deleted {deleted_count} bundles")

    except Exception as e:
        click.echo(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
