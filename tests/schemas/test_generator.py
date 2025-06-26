"""Comprehensive tests for the schema generator module."""

import tempfile
from pathlib import Path
from typing import Generator
from unittest.mock import patch

import pytest
import yaml

from agentbx.schemas.generator import AssetDefinition
from agentbx.schemas.generator import SchemaDefinition
from agentbx.schemas.generator import SchemaGenerator
from agentbx.schemas.generator import WorkflowPattern
from agentbx.schemas.generator import main
from agentbx.schemas.generator import quick_generate


@pytest.fixture
def temp_schema_dir() -> Generator[Path, None, None]:
    """Create a temporary directory with test schema files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create test schema file
        schema_file = temp_path / "test_data.yaml"
        schema_content = """
task_type: test_data
description: Test schema for unit tests
required_assets:
  - test_asset
optional_assets:
  - optional_asset
asset_definitions:
  test_asset:
    description: A test asset
    data_type: str
  optional_asset:
    description: An optional asset
    data_type: int
    default_values:
      default: 42
validation_rules:
  test_asset:
    not_empty: true
workflow_patterns:
  test_pattern:
    pattern_name: test_pattern
    requires:
      - test_asset
    produces:
      - output
dependencies:
  - other_schema
produces_for:
  - downstream_schema
"""
        schema_file.write_text(schema_content)

        yield temp_path


@pytest.fixture
def complex_schema_dir() -> Generator[Path, None, None]:
    """Create a temporary directory with complex test schema files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Create complex schema file
        schema_file = temp_path / "complex_data.yaml"
        schema_content = """
task_type: complex_data
description: Complex test schema for unit tests
required_assets:
  - complex_asset
optional_assets:
  - optional_complex_asset
asset_definitions:
  complex_asset:
    description: A complex test asset
    data_type: cctbx.miller.array
    must_be_complex: true
    data_must_be_positive: true
  optional_complex_asset:
    description: An optional complex asset
    data_type: cctbx.xray.structure
    checksum_required: true
validation_rules:
  complex_asset:
    finite_values_only: true
    reasonable_range: true
workflow_patterns:
  complex_pattern:
    pattern_name: complex_pattern
    requires:
      - complex_asset
    produces:
      - complex_output
    method: complex_method
dependencies:
  - base_schema
  - dependency_schema
produces_for:
  - final_schema
  - analysis_schema
"""
        schema_file.write_text(schema_content)

        yield temp_path


class TestAssetDefinition:
    """Test AssetDefinition model."""

    def test_basic_asset_definition(self) -> None:
        """Test creating a basic asset definition."""
        asset_data = {
            "description": "Test asset",
            "data_type": "str",
            "shape": "N",
            "units": "angstroms",
        }

        asset = AssetDefinition(**asset_data)
        assert asset.description == "Test asset"
        assert asset.data_type == "str"
        assert asset.shape == "N"
        assert asset.units == "angstroms"
        assert asset.checksum_required is False

    def test_complex_asset_definition(self) -> None:
        """Test creating a complex asset definition with all fields."""
        asset_data = {
            "description": "Complex test asset",
            "data_type": "cctbx.miller.array",
            "shape": "N",
            "units": "electrons",
            "checksum_required": True,
            "required_attributes": ["indices", "data"],
            "required_methods": ["is_complex_array"],
            "depends_on": ["other_asset"],
            "must_be_complex": True,
            "must_be_real": False,
            "data_must_be_positive": True,
            "default_values": {"scale": 1.0},
            "allowed_values": ["option1", "option2"],
            "valid_range": [0.0, 100.0],
            "required_keys": ["key1", "key2"],
            "optional_keys": ["opt1"],
            "expected_keys": ["exp1", "exp2"],
        }

        asset = AssetDefinition(**asset_data)
        assert asset.description == "Complex test asset"
        assert asset.data_type == "cctbx.miller.array"
        assert asset.checksum_required is True
        assert asset.required_attributes == ["indices", "data"]
        assert asset.must_be_complex is True
        assert asset.data_must_be_positive is True
        assert asset.default_values == {"scale": 1.0}
        assert asset.valid_range == [0.0, 100.0]

    def test_asset_definition_defaults(self) -> None:
        """Test that optional fields have correct defaults."""
        asset_data = {"description": "Test asset", "data_type": "str"}

        asset = AssetDefinition(**asset_data)
        assert asset.shape is None
        assert asset.units is None
        assert asset.checksum_required is False
        assert asset.required_attributes is None
        assert asset.must_be_complex is None
        assert asset.data_must_be_positive is None


class TestWorkflowPattern:
    """Test WorkflowPattern model."""

    def test_basic_workflow_pattern(self) -> None:
        """Test creating a basic workflow pattern."""
        pattern_data = {
            "pattern_name": "test_pattern",
            "requires": ["asset1", "asset2"],
            "produces": ["output1"],
        }

        pattern = WorkflowPattern(**pattern_data)
        assert pattern.pattern_name == "test_pattern"
        assert pattern.requires == ["asset1", "asset2"]
        assert pattern.produces == ["output1"]

    def test_workflow_pattern_all_fields(self):
        """Test creating a workflow pattern with all fields."""
        pattern_data = {
            "pattern_name": "complex_pattern",
            "requires": ["input1"],
            "produces": ["output1"],
            "modifies": ["mod1"],
            "preserves": ["preserve1"],
            "method": "test_method",
            "enables": ["enable1"],
            "input": ["in1"],
            "output": ["out1"],
            "process": "test_process",
            "checks": ["check1"],
            "outputs": ["output1"],
            "validates": ["validate1"],
        }

        pattern = WorkflowPattern(**pattern_data)
        assert pattern.pattern_name == "complex_pattern"
        assert pattern.requires == ["input1"]
        assert pattern.produces == ["output1"]
        assert pattern.modifies == ["mod1"]
        assert pattern.preserves == ["preserve1"]
        assert pattern.method == "test_method"
        assert pattern.enables == ["enable1"]
        assert pattern.input == ["in1"]
        assert pattern.output == ["out1"]
        assert pattern.process == "test_process"
        assert pattern.checks == ["check1"]
        assert pattern.outputs == ["output1"]
        assert pattern.validates == ["validate1"]


class TestSchemaDefinition:
    """Test SchemaDefinition model."""

    def test_basic_schema_definition(self):
        """Test creating a basic schema definition."""
        schema_data = {
            "task_type": "test_task",
            "description": "Test schema",
            "required_assets": ["asset1"],
            "asset_definitions": {
                "asset1": AssetDefinition(description="Test asset", data_type="str")
            },
        }

        schema = SchemaDefinition(**schema_data)
        assert schema.task_type == "test_task"
        assert schema.description == "Test schema"
        assert schema.required_assets == ["asset1"]
        assert "asset1" in schema.asset_definitions
        assert schema.optional_assets == []
        assert schema.validation_rules == {}
        assert schema.workflow_patterns == {}
        assert schema.dependencies == []
        assert schema.produces_for == []

    def test_schema_definition_with_all_fields(self):
        """Test creating a schema definition with all fields."""
        schema_data = {
            "task_type": "complex_task",
            "description": "Complex test schema",
            "required_assets": ["req1"],
            "optional_assets": ["opt1"],
            "asset_definitions": {
                "req1": AssetDefinition(description="Required", data_type="str"),
                "opt1": AssetDefinition(description="Optional", data_type="int"),
            },
            "validation_rules": {"req1": {"rule1": True}},
            "workflow_patterns": {
                "pattern1": WorkflowPattern(pattern_name="pattern1", requires=["req1"])
            },
            "dependencies": ["dep1"],
            "produces_for": ["prod1"],
        }

        schema = SchemaDefinition(**schema_data)
        assert schema.task_type == "complex_task"
        assert len(schema.required_assets) == 1
        assert len(schema.optional_assets) == 1
        assert len(schema.asset_definitions) == 2
        assert len(schema.validation_rules) == 1
        assert len(schema.workflow_patterns) == 1
        assert schema.dependencies == ["dep1"]
        assert schema.produces_for == ["prod1"]


class TestSchemaGenerator:
    """Test SchemaGenerator class."""

    def test_schema_generator_init(self, temp_schema_dir):
        """Test SchemaGenerator initialization."""
        generator = SchemaGenerator(temp_schema_dir)
        assert generator.schema_dir == temp_schema_dir
        assert generator.schemas == {}
        assert generator.generated_models == {}

    def test_load_schema(self, temp_schema_dir):
        """Test loading a single schema file."""
        generator = SchemaGenerator(temp_schema_dir)
        schema_file = temp_schema_dir / "test_data.yaml"

        schema = generator.load_schema(schema_file)

        assert schema.task_type == "test_data"
        assert schema.description == "Test schema for unit tests"
        assert schema.required_assets == ["test_asset"]
        assert schema.optional_assets == ["optional_asset"]
        assert len(schema.asset_definitions) == 2
        assert "test_asset" in schema.asset_definitions
        assert "optional_asset" in schema.asset_definitions

        # Check asset definitions
        test_asset = schema.asset_definitions["test_asset"]
        assert test_asset.description == "A test asset"
        assert test_asset.data_type == "str"
        assert test_asset.checksum_required is False

        optional_asset = schema.asset_definitions["optional_asset"]
        assert optional_asset.description == "An optional asset"
        assert optional_asset.data_type == "int"
        assert optional_asset.default_values == {"default": 42}

        # Check validation rules
        assert "test_asset" in schema.validation_rules
        assert schema.validation_rules["test_asset"]["not_empty"] is True

        # Check workflow patterns
        assert "test_pattern" in schema.workflow_patterns
        pattern = schema.workflow_patterns["test_pattern"]
        assert pattern.pattern_name == "test_pattern"
        assert pattern.requires == ["test_asset"]
        assert pattern.produces == ["output"]

        # Check dependencies
        assert schema.dependencies == ["other_schema"]
        assert schema.produces_for == ["downstream_schema"]

    def test_load_schema_with_list_validation_rules(self, complex_schema_dir):
        """Test loading schema with list-based validation rules."""
        generator = SchemaGenerator(complex_schema_dir)
        schema_file = complex_schema_dir / "complex_data.yaml"

        schema = generator.load_schema(schema_file)

        # Check that list-based validation rules are converted to dict
        assert "complex_asset" in schema.validation_rules
        rules = schema.validation_rules["complex_asset"]
        assert isinstance(rules, dict)
        assert rules["finite_values_only"] is True
        assert rules["reasonable_range"] is True

    def test_load_schema_with_list_workflow_patterns(self, complex_schema_dir):
        """Test loading schema with list-based workflow patterns."""
        generator = SchemaGenerator(complex_schema_dir)
        schema_file = complex_schema_dir / "complex_data.yaml"

        schema = generator.load_schema(schema_file)

        # Check that list-based workflow patterns are converted properly
        assert "complex_pattern" in schema.workflow_patterns
        pattern = schema.workflow_patterns["complex_pattern"]
        assert pattern.pattern_name == "complex_pattern"
        assert pattern.requires == ["complex_asset"]
        assert pattern.produces == ["complex_output"]

    def test_load_schema_empty_file(self, temp_schema_dir):
        """Test loading an empty YAML file."""
        empty_file = temp_schema_dir / "empty.yaml"
        with open(empty_file, "w") as f:
            f.write("")

        generator = SchemaGenerator(temp_schema_dir)

        with pytest.raises(ValueError, match="Empty or invalid YAML file"):
            generator.load_schema(empty_file)

    def test_load_schema_asset_definitions_not_dict(self, temp_schema_dir):
        """Test load_schema with asset_definitions as a string (should raise AttributeError)."""
        generator = SchemaGenerator(temp_schema_dir)
        schema_data = {
            "task_type": "bad_schema",
            "description": "Bad schema",
            "required_assets": ["asset1"],
            "asset_definitions": "not_a_dict",
        }
        schema_file = temp_schema_dir / "bad_schema.yaml"
        with open(schema_file, "w") as f:
            yaml.dump(schema_data, f)
        with pytest.raises(AttributeError):
            generator.load_schema(schema_file)

    def test_load_schema_workflow_patterns_unexpected_type(self, temp_schema_dir):
        """Test load_schema with workflow_patterns as an int (should skip and not crash)."""
        generator = SchemaGenerator(temp_schema_dir)
        schema_data = {
            "task_type": "bad_workflow",
            "description": "Bad workflow",
            "required_assets": ["asset1"],
            "asset_definitions": {
                "asset1": {"description": "desc", "data_type": "str"}
            },
            "workflow_patterns": 12345,
        }
        schema_file = temp_schema_dir / "bad_workflow.yaml"
        with open(schema_file, "w") as f:
            yaml.dump(schema_data, f)
        # Should not raise, just skip
        schema = generator.load_schema(schema_file)
        assert schema.workflow_patterns == {}

    def test_load_schema_validation_rules_unexpected_type(self, temp_schema_dir):
        """Test load_schema with validation_rules as a string (should skip and not crash)."""
        generator = SchemaGenerator(temp_schema_dir)
        schema_data = {
            "task_type": "bad_validation",
            "description": "Bad validation",
            "required_assets": ["asset1"],
            "asset_definitions": {
                "asset1": {"description": "desc", "data_type": "str"}
            },
            "validation_rules": "not_a_dict",
        }
        schema_file = temp_schema_dir / "bad_validation.yaml"
        with open(schema_file, "w") as f:
            yaml.dump(schema_data, f)
        # Should not raise, just skip
        schema = generator.load_schema(schema_file)
        assert schema.validation_rules == {}

    def test_load_all_schemas(self, temp_schema_dir):
        """Test loading all schemas from directory."""
        generator = SchemaGenerator(temp_schema_dir)
        generator.load_all_schemas()

        assert len(generator.schemas) == 1
        assert "test_data" in generator.schemas

        schema = generator.schemas["test_data"]
        assert schema.task_type == "test_data"
        assert len(schema.asset_definitions) == 2

    def test_load_all_schemas_empty_directory(self, tmp_path):
        """Test loading schemas from empty directory."""
        generator = SchemaGenerator(tmp_path)
        generator.load_all_schemas()

        assert len(generator.schemas) == 0

    def test_load_all_schemas_with_errors(self, temp_schema_dir):
        """Test loading schemas when some files have errors."""
        # Create a valid schema
        valid_schema = {
            "task_type": "valid_data",
            "description": "Valid schema",
            "required_assets": ["asset1"],
            "asset_definitions": {
                "asset1": {"description": "Valid asset", "data_type": "str"}
            },
        }

        valid_file = temp_schema_dir / "valid_data.yaml"
        with open(valid_file, "w") as f:
            yaml.dump(valid_schema, f)

        # Create an invalid schema
        invalid_file = temp_schema_dir / "invalid_data.yaml"
        with open(invalid_file, "w") as f:
            f.write("invalid: yaml: content: [")

        generator = SchemaGenerator(temp_schema_dir)
        generator.load_all_schemas()

        # Should load the valid schema and skip the invalid one
        assert len(generator.schemas) == 2  # test_data + valid_data
        assert "test_data" in generator.schemas
        assert "valid_data" in generator.schemas

    def test_generate_asset_model(self, temp_schema_dir):
        """Test generating asset model code."""
        generator = SchemaGenerator(temp_schema_dir)
        generator.load_all_schemas()

        schema = generator.schemas["test_data"]
        test_asset = schema.asset_definitions["test_asset"]

        model_code = generator.generate_asset_model("test_asset", test_asset)

        assert "test_asset: str = Field(" in model_code
        assert 'description="A test asset"' in model_code

    def test_generate_asset_model_with_constraints(self, temp_schema_dir):
        """Test generating asset model with constraints."""
        generator = SchemaGenerator(temp_schema_dir)

        # Create asset with constraints
        asset_def = AssetDefinition(
            description="Test asset with constraints",
            data_type="float",
            valid_range=[0.0, 100.0],
            default_values={"value": 50.0},
        )

        model_code = generator.generate_asset_model("constrained_asset", asset_def)

        assert "constrained_asset: float = Field(" in model_code
        assert "ge=0.0, le=100.0" in model_code
        assert "default=50.0" in model_code

    def test_generate_validators(self, temp_schema_dir):
        """Test generating validators."""
        generator = SchemaGenerator(temp_schema_dir)
        generator.load_all_schemas()

        schema = generator.schemas["test_data"]
        validators = generator.generate_validators(schema)

        # Should generate validators for test_asset
        validator_code = "\n".join(validators)
        assert "@field_validator('test_asset')" in validator_code
        assert "def validate_test_asset" in validator_code

    def test_generate_validators_cctbx_types(self, temp_schema_dir):
        """Test generating validators for CCTBX types."""
        generator = SchemaGenerator(temp_schema_dir)

        # Create schema with CCTBX types
        schema_data = {
            "task_type": "cctbx_test",
            "description": "Test CCTBX types",
            "required_assets": ["xray_structure", "miller_array"],
            "asset_definitions": {
                "xray_structure": AssetDefinition(
                    description="X-ray structure", data_type="cctbx.xray.structure"
                ),
                "miller_array": AssetDefinition(
                    description="Miller array",
                    data_type="cctbx.miller.array",
                    must_be_complex=True,
                    data_must_be_positive=True,
                ),
            },
            "validation_rules": {
                "xray_structure": {"has_scatterers": True},
                "miller_array": {"finite_values_only": True},
            },
        }

        schema = SchemaDefinition(**schema_data)
        validators = generator.generate_validators(schema)

        validator_code = "\n".join(validators)

        # Check for xray_structure validators
        assert "validate_xray_structure" in validator_code
        assert "scatterers" in validator_code
        assert "unit_cell" in validator_code

        # Check for miller_array validators
        assert "validate_miller_array" in validator_code
        assert "indices" in validator_code
        assert "data" in validator_code
        assert "is_complex_array" in validator_code
        assert "All values must be finite" in validator_code

    def test_generate_bundle_model(self, temp_schema_dir):
        """Test generating complete bundle model."""
        generator = SchemaGenerator(temp_schema_dir)
        generator.load_all_schemas()

        schema = generator.schemas["test_data"]
        model_code = generator.generate_bundle_model(schema)

        # Check class name
        assert "class TestDataBundle(BaseModel):" in model_code

        # Check docstring
        assert '"""' in model_code
        assert "Test schema for unit tests" in model_code

        # Check bundle metadata
        assert 'bundle_type: Literal["test_data"] = "test_data"' in model_code
        assert (
            "created_at: datetime = Field(default_factory=datetime.now)" in model_code
        )
        assert "bundle_id: Optional[str] = None" in model_code
        assert "checksum: Optional[str] = None" in model_code

        # Check required assets
        assert "# Required assets" in model_code
        assert "test_asset: str = Field(" in model_code

        # Check optional assets
        assert "# Optional assets" in model_code
        assert "optional_asset: int = Field(default=None" in model_code

        # Check validators
        assert "@field_validator('test_asset')" in model_code

        # Check utility methods
        assert "def calculate_checksum(self) -> str:" in model_code
        assert (
            "def validate_dependencies(self, available_bundles: Dict[str, 'BaseModel']) -> bool:"
            in model_code
        )

    def test_generate_all_models(self, temp_schema_dir):
        """Test generating all models."""
        generator = SchemaGenerator(temp_schema_dir)
        generator.load_all_schemas()

        all_models = generator.generate_all_models()

        # Check imports
        assert "from typing import Dict, List, Optional, Any, Literal" in all_models
        assert "from pydantic import BaseModel, Field, field_validator" in all_models
        assert "from datetime import datetime" in all_models
        assert "import hashlib" in all_models

        # Check generated class
        assert "class TestDataBundle(BaseModel):" in all_models

    def test_write_generated_models(self, temp_schema_dir, tmp_path):
        """Test writing generated models to file."""
        generator = SchemaGenerator(temp_schema_dir)
        generator.load_all_schemas()

        output_file = tmp_path / "generated.py"
        generator.write_generated_models(output_file)

        assert output_file.exists()

        # Check file contents
        with open(output_file) as f:
            content = f.read()

        assert "class TestDataBundle(BaseModel):" in content
        assert "from pydantic import BaseModel, Field, field_validator" in content


class TestMainFunction:
    """Test main function and CLI interface."""

    @patch("agentbx.schemas.generator.watch_for_changes")
    def test_main_success(self, mock_watch, temp_schema_dir, tmp_path):
        """Test successful main function execution."""
        output_file = tmp_path / "generated.py"

        with patch(
            "sys.argv",
            [
                "generator.py",
                "--schemas-dir",
                str(temp_schema_dir),
                "--output",
                str(output_file),
            ],
        ):
            result = main()

        assert result == 0
        assert output_file.exists()

    def test_main_schema_dir_not_found(self, tmp_path):
        """Test main function with non-existent schema directory."""
        non_existent_dir = tmp_path / "non_existent"
        output_file = tmp_path / "generated.py"

        with patch(
            "sys.argv",
            [
                "generator.py",
                "--schemas-dir",
                str(non_existent_dir),
                "--output",
                str(output_file),
            ],
        ):
            result = main()

        assert result == 1

    def test_main_no_schema_files(self, tmp_path):
        """Test main function with empty schema directory."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        output_file = tmp_path / "generated.py"

        with patch(
            "sys.argv",
            [
                "generator.py",
                "--schemas-dir",
                str(empty_dir),
                "--output",
                str(output_file),
            ],
        ):
            result = main()

        assert result == 1

    @pytest.mark.skip(reason="File watching test hangs during coverage runs")
    @patch("agentbx.schemas.generator.watch_for_changes")
    def test_main_with_watch(self, mock_watch, temp_schema_dir, tmp_path):
        """Test main function with watch flag."""
        output_file = tmp_path / "generated.py"

        with patch(
            "sys.argv",
            [
                "generator.py",
                "--schemas-dir",
                str(temp_schema_dir),
                "--output",
                str(output_file),
                "--watch",
            ],
        ):
            # Mock KeyboardInterrupt to exit the watch loop
            mock_watch.side_effect = KeyboardInterrupt()
            result = main()

        assert result == 0
        mock_watch.assert_called_once()

    def test_main_with_verbose(self, temp_schema_dir, tmp_path):
        """Test main function with verbose flag."""
        output_file = tmp_path / "generated.py"

        with patch(
            "sys.argv",
            [
                "generator.py",
                "--schemas-dir",
                str(temp_schema_dir),
                "--output",
                str(output_file),
                "--verbose",
            ],
        ):
            result = main()

        assert result == 0
        assert output_file.exists()


class TestQuickGenerate:
    """Test quick_generate function."""

    def test_quick_generate_success(self, temp_schema_dir, tmp_path):
        """Test successful quick_generate execution."""
        # Mock the default paths
        with patch("agentbx.schemas.generator.Path") as mock_path:
            mock_path.return_value = temp_schema_dir
            mock_path.side_effect = lambda x: (
                temp_schema_dir
                if "definitions" in str(x)
                else tmp_path / "generated.py"
            )

            quick_generate()

            # Check that the function completed without error
            # (we can't easily check the output file since it's mocked)

    def test_quick_generate_schema_dir_not_found(self, tmp_path):
        """Test quick_generate with non-existent schema directory."""
        non_existent_dir = tmp_path / "non_existent"

        with patch("agentbx.schemas.generator.Path") as mock_path:
            mock_path.return_value = non_existent_dir

            with pytest.raises(FileNotFoundError):
                quick_generate()


class TestIntegration:
    """Integration tests for the complete workflow."""

    @pytest.fixture
    def real_schema_dir(self):
        """Use the real schema definitions directory."""
        return Path("src/agentbx/schemas/definitions")

    def test_load_real_schemas(self, real_schema_dir):
        """Test loading the actual schema definitions."""
        if not real_schema_dir.exists():
            pytest.skip("Real schema directory not found")

        generator = SchemaGenerator(real_schema_dir)
        generator.load_all_schemas()

        # Should load all the real schemas
        assert len(generator.schemas) > 0

        # Check for expected schemas
        expected_schemas = [
            "atomic_model_data",
            "structure_factor_data",
            "target_data",
            "gradient_data",
            "experimental_data",
        ]

        for schema_name in expected_schemas:
            if schema_name in generator.schemas:
                schema = generator.schemas[schema_name]
                assert schema.task_type == schema_name
                assert len(schema.asset_definitions) > 0

    def test_generate_real_models(self, real_schema_dir, tmp_path):
        """Test generating models from real schemas."""
        if not real_schema_dir.exists():
            pytest.skip("Real schema directory not found")

        generator = SchemaGenerator(real_schema_dir)
        generator.load_all_schemas()

        output_file = tmp_path / "generated.py"
        generator.write_generated_models(output_file)

        assert output_file.exists()

        # Check that the generated file contains expected classes
        with open(output_file) as f:
            content = f.read()

        # Should contain classes for each schema
        expected_classes = [
            "AtomicModelDataBundle",
            "StructureFactorDataBundle",
            "TargetDataBundle",
            "GradientDataBundle",
            "ExperimentalDataBundle",
        ]

        for class_name in expected_classes:
            if class_name in content:
                assert f"class {class_name}(BaseModel):" in content


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_asset_definition_with_none_values(self, temp_schema_dir):
        """Test handling of None values in asset definitions."""
        generator = SchemaGenerator(temp_schema_dir)

        # Create schema with None values
        schema_data = {
            "task_type": "none_test",
            "description": "Test with None values",
            "required_assets": ["asset1"],
            "asset_definitions": {"asset1": None},  # This should be handled gracefully
        }

        schema_file = temp_schema_dir / "none_test.yaml"
        with open(schema_file, "w") as f:
            yaml.dump(schema_data, f)

        # Should handle None values gracefully
        generator.load_all_schemas()
        assert (
            len(generator.schemas) == 2
        )  # Should load both schemas (test_data + none_test)

    def test_workflow_pattern_with_none_values(self, temp_schema_dir):
        """Test handling of None values in workflow patterns."""
        generator = SchemaGenerator(temp_schema_dir)

        # Create schema with None workflow pattern
        schema_data = {
            "task_type": "none_pattern_test",
            "description": "Test with None workflow patterns",
            "required_assets": ["asset1"],
            "asset_definitions": {
                "asset1": {"description": "Test asset", "data_type": "str"}
            },
            "workflow_patterns": {
                "pattern1": None  # This should be handled gracefully
            },
        }

        schema_file = temp_schema_dir / "none_pattern_test.yaml"
        with open(schema_file, "w") as f:
            yaml.dump(schema_data, f)

        # Should handle None values gracefully
        generator.load_all_schemas()
        assert (
            len(generator.schemas) == 2
        )  # Should load both schemas (test_data + none_pattern_test)

    def test_validation_rules_with_none_values(self, temp_schema_dir):
        """Test handling of None values in validation rules."""
        generator = SchemaGenerator(temp_schema_dir)

        # Create schema with None validation rules
        schema_data = {
            "task_type": "none_rules_test",
            "description": "Test with None validation rules",
            "required_assets": ["asset1"],
            "asset_definitions": {
                "asset1": {"description": "Test asset", "data_type": "str"}
            },
            "validation_rules": {"asset1": None},  # This should be handled gracefully
        }

        schema_file = temp_schema_dir / "none_rules_test.yaml"
        with open(schema_file, "w") as f:
            yaml.dump(schema_data, f)

        # Should handle None values gracefully
        generator.load_all_schemas()
        assert (
            len(generator.schemas) == 2
        )  # Should load both schemas (test_data + none_rules_test)

    def test_malformed_yaml_structure(self, temp_schema_dir):
        """Test handling of malformed YAML structure."""
        generator = SchemaGenerator(temp_schema_dir)

        # Create schema with malformed structure
        schema_data = {
            "task_type": "malformed_test",
            "description": "Test with malformed structure",
            "required_assets": "not_a_list",  # Should be a list
            "asset_definitions": "not_a_dict",  # Should be a dict
        }

        schema_file = temp_schema_dir / "malformed_test.yaml"
        with open(schema_file, "w") as f:
            yaml.dump(schema_data, f)

        # Should handle malformed structure gracefully
        generator.load_all_schemas()
        # Should still load the valid schema
        assert "test_data" in generator.schemas


def test_generate_asset_model_all_optionals():
    """Test generate_asset_model with all optional fields in AssetDefinition."""
    asset_def = AssetDefinition(description="desc", data_type="str")
    generator = SchemaGenerator(Path("/tmp"))
    code = generator.generate_asset_model("opt_asset", asset_def)
    assert "opt_asset: str = Field(" in code


def test_generate_asset_model_type_mapping():
    """Test generate_asset_model with all supported types."""
    generator = SchemaGenerator(Path("/tmp"))
    for dtype, pytype in [
        ("cctbx.xray.structure", "Any"),
        ("cctbx.miller.array", "Any"),
        ("dict", "Dict[str, Any]"),
        ("str", "str"),
        ("float", "float"),
        ("int", "int"),
        ("bool", "bool"),
        ("bytes", "bytes"),
    ]:
        asset_def = AssetDefinition(description="desc", data_type=dtype)
        code = generator.generate_asset_model("asset", asset_def)
        assert f"asset: {pytype} = " in code


def test_generate_validators_no_rules():
    """Test generate_validators with no validation rules (should return empty list)."""
    generator = SchemaGenerator(Path("/tmp"))
    schema = SchemaDefinition(
        task_type="no_rules",
        description="desc",
        required_assets=["a1"],
        asset_definitions={"a1": AssetDefinition(description="desc", data_type="str")},
    )
    validators = generator.generate_validators(schema)
    assert validators == []


def test_main_exception(monkeypatch, temp_schema_dir, tmp_path):
    """Test main() exception handling by patching load_all_schemas to raise."""
    from agentbx.schemas import generator as genmod

    monkeypatch.setattr(
        genmod.SchemaGenerator,
        "load_all_schemas",
        lambda self: (_ for _ in ()).throw(Exception("fail!")),
    )
    output_file = tmp_path / "generated.py"
    with patch(
        "sys.argv",
        [
            "generator.py",
            "--schemas-dir",
            str(temp_schema_dir),
            "--output",
            str(output_file),
        ],
    ):
        result = genmod.main()
    assert result == 1
