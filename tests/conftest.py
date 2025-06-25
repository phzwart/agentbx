"""Test configuration and fixtures for agentbx tests."""

import pytest
from unittest.mock import Mock, MagicMock


@pytest.fixture(autouse=True)
def mock_cctbx_imports(monkeypatch):
    """Mock CCTBX/MMTBX imports to avoid import errors in CI."""
    
    # Mock cctbx modules
    mock_cctbx = Mock()
    mock_cctbx.crystal = Mock()
    mock_cctbx.miller = Mock()
    mock_cctbx.xray = Mock()
    mock_cctbx.array_family = Mock()
    mock_cctbx.french_wilson = Mock()
    mock_cctbx.examples = Mock()
    
    # Mock cctbx.array_family.flex
    mock_flex = Mock()
    mock_flex.vec3_double = Mock()
    mock_flex.double = Mock()
    mock_cctbx.array_family.flex = mock_flex
    
    # Mock cctbx.xray.structure
    mock_xray_structure = Mock()
    mock_xray_structure.scatterers = []
    mock_xray_structure.unit_cell = Mock()
    mock_cctbx.xray.structure = mock_xray_structure
    
    # Mock cctbx.miller.array
    mock_miller_array = Mock()
    mock_miller_array.indices = Mock()
    mock_miller_array.data = Mock()
    mock_miller_array.is_complex_array = Mock(return_value=True)
    mock_miller_array.is_real_array = Mock(return_value=True)
    mock_miller_array.is_bool_array = Mock(return_value=True)
    mock_cctbx.miller.array = mock_miller_array
    
    # Mock cctbx.miller.set
    mock_cctbx.miller.set = Mock()
    
    # Mock iotbx modules
    mock_iotbx = Mock()
    mock_iotbx.reflection_file_reader = Mock()
    
    # Mock mmtbx modules
    mock_mmtbx = Mock()
    mock_mmtbx.refinement = Mock()
    mock_mmtbx.refinement.targets = Mock()
    mock_mmtbx.bulk_solvent = Mock()
    mock_mmtbx.bulk_solvent.bulk_solvent_and_scaling = Mock()
    
    # Apply the mocks
    monkeypatch.setattr("builtins.__import__", mock_import)
    
    # Store the mocks for potential use in tests
    return {
        "cctbx": mock_cctbx,
        "iotbx": mock_iotbx,
        "mmtbx": mock_mmtbx,
        "flex": mock_flex,
        "miller_array": mock_miller_array,
        "xray_structure": mock_xray_structure
    }


def mock_import(name, *args, **kwargs):
    """Mock import function to handle CCTBX/MMTBX imports."""
    if name in ["cctbx", "iotbx", "mmtbx"]:
        # Return appropriate mock based on the module name
        if name == "cctbx":
            mock_module = Mock()
            mock_module.crystal = Mock()
            mock_module.miller = Mock()
            mock_module.xray = Mock()
            mock_module.array_family = Mock()
            mock_module.french_wilson = Mock()
            mock_module.examples = Mock()
            
            # Set up flex
            mock_flex = Mock()
            mock_flex.vec3_double = Mock()
            mock_flex.double = Mock()
            mock_module.array_family.flex = mock_flex
            
            # Set up miller array
            mock_array = Mock()
            mock_array.indices = Mock()
            mock_array.data = Mock()
            mock_array.is_complex_array = Mock(return_value=True)
            mock_array.is_real_array = Mock(return_value=True)
            mock_array.is_bool_array = Mock(return_value=True)
            mock_module.miller.array = mock_array
            
            # Set up xray structure
            mock_structure = Mock()
            mock_structure.scatterers = []
            mock_structure.unit_cell = Mock()
            mock_module.xray.structure = mock_structure
            
            return mock_module
            
        elif name == "iotbx":
            mock_module = Mock()
            mock_module.reflection_file_reader = Mock()
            return mock_module
            
        elif name == "mmtbx":
            mock_module = Mock()
            mock_module.refinement = Mock()
            mock_module.refinement.targets = Mock()
            mock_module.bulk_solvent = Mock()
            mock_module.bulk_solvent.bulk_solvent_and_scaling = Mock()
            return mock_module
    
    # For all other imports, use the real import
    return __import__(name, *args, **kwargs)


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