"""
Redis utility functions for agentbx.
"""

import argparse
import json
import logging
import sys

from agentbx.core.redis_manager import RedisManager


logger = logging.getLogger(__name__)


def inspect_bundles_cli():
    """CLI tool to inspect bundles in Redis."""
    parser = argparse.ArgumentParser(description="Inspect bundles in Redis")
    parser.add_argument(
        "--host", default="localhost", help="Redis host (default: localhost)"
    )
    parser.add_argument(
        "--port", type=int, default=6379, help="Redis port (default: 6379)"
    )
    parser.add_argument("--type", help="Filter by bundle type")
    parser.add_argument("--bundle-id", help="Inspect specific bundle by ID")
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Show only metadata, not full content analysis",
    )
    parser.add_argument("--json", action="store_true", help="Output in JSON format")

    args = parser.parse_args()

    # Initialize Redis manager
    redis_manager = RedisManager(host=args.host, port=args.port)

    if not redis_manager.is_healthy():
        print("Error: Redis connection is not healthy", file=sys.stderr)
        sys.exit(1)

    try:
        if args.bundle_id:
            # Inspect specific bundle
            if args.metadata_only:
                result = redis_manager.get_bundle_metadata(args.bundle_id)
            else:
                result = redis_manager.inspect_bundle(args.bundle_id)
        else:
            # List bundles
            result = redis_manager.list_bundles_with_metadata(args.type)

        if args.json:
            print(json.dumps(result, indent=2, default=str))
        else:
            if isinstance(result, list):
                if not result:
                    print("No bundles found")
                else:
                    for bundle_info in result:
                        print(f"Bundle ID: {bundle_info['bundle_id']}")
                        print(f"  Type: {bundle_info['bundle_type']}")
                        print(f"  Created: {bundle_info['created_at']}")
                        print(f"  Size: {bundle_info['size_bytes']} bytes")
                        print(f"  Checksum: {bundle_info['checksum']}")
                        print("---")
            else:
                print("Bundle Information:")
                print(json.dumps(result, indent=2, default=str))

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    inspect_bundles_cli()
