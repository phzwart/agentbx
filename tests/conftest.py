"""Test configuration and fixtures for agentbx tests."""

import builtins
import sys
from unittest.mock import MagicMock
from unittest.mock import Mock

import pytest


_original_import = builtins.__import__


def mock_import(name, *args, **kwargs):
    # Only mock the top-level cctbx, iotbx, mmtbx modules
    if name.split(".")[0] in {"cctbx", "iotbx", "mmtbx"}:
        if name not in sys.modules:
            sys.modules[name] = MagicMock()
        return sys.modules[name]
    return _original_import(name, *args, **kwargs)


def pytest_sessionstart(session):
    builtins.__import__ = mock_import


def pytest_sessionfinish(session, exitstatus):
    builtins.__import__ = _original_import


@pytest.fixture
def mock_redis():
    """Mock Redis for testing."""
    mock_redis = Mock()
    mock_redis.ping = Mock(return_value=True)
    return mock_redis


@pytest.fixture
def mock_redis_exceptions():
    """Mock Redis exceptions."""

    class MockRedisError(Exception):
        pass

    class MockConnectionError(Exception):
        pass

    return MockRedisError, MockConnectionError
