"""
Base bundle class for agentbx data containers.
"""

import hashlib
import json
from abc import ABC
from datetime import datetime
from typing import Any
from typing import Dict
from typing import Optional


class Bundle(ABC):
    """
    Base class for data bundles in agentbx.

    Bundles are containers that hold related data assets and metadata.
    Each bundle has a specific type and can contain multiple named assets.
    """

    def __init__(self, bundle_type: str, bundle_id: Optional[str] = None):
        """
        Initialize bundle.

        Args:
            bundle_type: Type identifier for this bundle
            bundle_id: Optional custom ID
        """
        self.bundle_type = bundle_type
        self.bundle_id = bundle_id
        self.created_at = datetime.now()
        self.assets: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}

    def add_asset(self, name: str, asset: Any) -> None:
        """
        Add an asset to the bundle.

        Args:
            name: Asset name
            asset: Asset data
        """
        self.assets[name] = asset

    def get_asset(self, name: str) -> Any:
        """
        Get an asset from the bundle.

        Args:
            name: Asset name

        Returns:
            Asset data

        Raises:
            KeyError: If asset not found
        """
        if name not in self.assets:
            raise KeyError(f"Asset '{name}' not found in bundle")
        return self.assets[name]

    def has_asset(self, name: str) -> bool:
        """
        Check if bundle has an asset.

        Args:
            name: Asset name

        Returns:
            True if asset exists
        """
        return name in self.assets

    def remove_asset(self, name: str) -> bool:
        """
        Remove an asset from the bundle.

        Args:
            name: Asset name

        Returns:
            True if asset was removed, False if not found
        """
        if name in self.assets:
            del self.assets[name]
            return True
        return False

    def add_metadata(self, key: str, value: Any) -> None:
        """
        Add metadata to the bundle.

        Args:
            key: Metadata key
            value: Metadata value
        """
        self.metadata[key] = value

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """
        Get metadata from the bundle.

        Args:
            key: Metadata key
            default: Default value if key not found

        Returns:
            Metadata value
        """
        return self.metadata.get(key, default)

    def calculate_checksum(self) -> str:
        """
        Calculate checksum of bundle contents.

        Returns:
            SHA256 checksum (first 16 characters)
        """
        # Create a serializable representation
        content = {
            "bundle_type": self.bundle_type,
            "assets": self.assets,
            "metadata": self.metadata,
        }

        # Convert to JSON string and hash
        content_str = json.dumps(content, sort_keys=True, default=str)
        return hashlib.sha256(content_str.encode()).hexdigest()[:16]

    def validate(self) -> bool:
        """
        Validate bundle contents.

        Returns:
            True if bundle is valid

        Raises:
            ValueError: If bundle is invalid
        """
        if not self.bundle_type:
            raise ValueError("Bundle must have a type")

        if not self.assets:
            raise ValueError("Bundle must have at least one asset")

        return True

    def get_size_estimate(self) -> int:
        """
        Estimate bundle size in bytes.

        Returns:
            Estimated size in bytes
        """
        content = {
            "bundle_type": self.bundle_type,
            "assets": self.assets,
            "metadata": self.metadata,
        }
        return len(json.dumps(content, default=str).encode())

    def __str__(self) -> str:
        """String representation of bundle."""
        return f"Bundle(type={self.bundle_type}, assets={list(self.assets.keys())})"

    def __repr__(self) -> str:
        """Detailed string representation of bundle."""
        return (
            f"Bundle(bundle_type='{self.bundle_type}', "
            f"bundle_id='{self.bundle_id}', "
            f"assets={list(self.assets.keys())}, "
            f"created_at={self.created_at})"
        )
