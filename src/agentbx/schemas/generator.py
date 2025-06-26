# src/agentbx/schemas/generator.py
"""
Schema generator for agentbx.

This module generates Pydantic schemas from YAML definitions.
"""

from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import yaml
from pydantic import BaseModel


class AssetDefinition(BaseModel):
    """Pydantic model for individual asset definitions from YAML."""

    description: str
    data_type: str
    shape: Optional[str] = None
    units: Optional[str] = None
    checksum_required: bool = False
    required_attributes: Optional[List[str]] = None
    required_methods: Optional[List[str]] = None
    depends_on: Optional[List[str]] = None
    must_be_complex: Optional[bool] = None
    must_be_real: Optional[bool] = None
    data_must_be_positive: Optional[bool] = None
    default_values: Optional[Dict[str, Any]] = None
    allowed_values: Optional[List[str]] = None
    valid_range: Optional[List[float]] = None
    required_keys: Optional[List[str]] = None
    optional_keys: Optional[List[str]] = None
    expected_keys: Optional[List[str]] = None


class ValidationRule(BaseModel):
    """Pydantic model for validation rules from YAML."""

    rule_name: str
    parameters: Optional[Dict[str, Any]] = None


class WorkflowPattern(BaseModel):
    """Pydantic model for workflow patterns from YAML."""

    pattern_name: str
    requires: Optional[List[str]] = None
    produces: Optional[List[str]] = None
    modifies: Optional[List[str]] = None
    preserves: Optional[List[str]] = None
    method: Optional[str] = None
    enables: Optional[List[str]] = None
    input: Optional[List[str]] = None
    output: Optional[List[str]] = None
    process: Optional[str] = None
    checks: Optional[List[str]] = None
    outputs: Optional[List[str]] = None
    validates: Optional[List[str]] = None


class SchemaDefinition(BaseModel):
    """Complete schema definition parsed from YAML."""

    task_type: str
    description: str
    required_assets: List[str]
    optional_assets: Optional[List[str]] = []
    asset_definitions: Dict[str, AssetDefinition]
    validation_rules: Optional[Dict[str, Dict[str, Any]]] = (
        {}
    )  # Changed from List to Dict
    workflow_patterns: Optional[Dict[str, WorkflowPattern]] = {}
    dependencies: Optional[List[str]] = []
    produces_for: Optional[List[str]] = []


class SchemaGenerator:
    """Generates Pydantic models from YAML schema definitions."""

    def __init__(self, schema_dir: Path):
        self.schema_dir = Path(schema_dir)
        self.schemas: Dict[str, SchemaDefinition] = {}
        self.generated_models: Dict[str, type] = {}

    def load_schema(self, schema_file: Path) -> SchemaDefinition:
        """Load and parse a single YAML schema file."""
        try:
            print(f"🔍 Loading {schema_file.name}...")

            with open(schema_file) as f:
                raw_schema = yaml.safe_load(f)

            if raw_schema is None:
                raise ValueError(f"Empty or invalid YAML file: {schema_file}")

            print(f"   ✅ YAML loaded, keys: {list(raw_schema.keys())}")

            # Convert asset_definitions to AssetDefinition objects
            print("   🔧 Processing asset_definitions...")
            asset_defs = {}
            asset_definitions_raw = raw_schema.get("asset_definitions", {})
            if asset_definitions_raw:
                for asset_name, asset_data in asset_definitions_raw.items():
                    if asset_data is None:
                        print(f"Warning: asset_data is None for {asset_name}")
                        continue
                    try:
                        asset_defs[asset_name] = AssetDefinition(**asset_data)
                        print(f"     ✅ {asset_name}")
                    except Exception as e:
                        print(
                            "     ❌ Error creating AssetDefinition for {}: {}".format(
                                asset_name, e
                            )
                        )
                        continue

            # Convert workflow_patterns to WorkflowPattern objects
            print("   🔧 Processing workflow_patterns...")
            workflow_patterns = {}
            workflow_patterns_raw = raw_schema.get("workflow_patterns", {})
            if workflow_patterns_raw:
                if isinstance(workflow_patterns_raw, dict):
                    for pattern_name, pattern_data in workflow_patterns_raw.items():
                        print(
                            "     🔍 Processing pattern '{}': {}".format(
                                pattern_name, type(pattern_data)
                            )
                        )
                        if pattern_data is None:
                            pattern_data = {}
                        # Handle different workflow pattern structures
                        if isinstance(pattern_data, list):
                            print(
                                "       🔄 Converting list to dict for {}".format(
                                    pattern_name
                                )
                            )
                            # Convert list format to dict format
                            combined_data = {"pattern_name": pattern_name}
                            for item in pattern_data:
                                if isinstance(item, dict):
                                    combined_data.update(item)
                                    print(f"         📝 Added: {list(item.keys())}")
                                elif isinstance(item, str):
                                    print(f"         📝 String item: {item}")
                                    # Handle simple string items
                                    if ":" in item:
                                        key, val = item.split(":", 1)
                                        combined_data[key.strip()] = val.strip().strip(
                                            '"'
                                        )
                            pattern_data = combined_data
                            print(
                                "       ✅ Converted to: {}".format(
                                    list(pattern_data.keys())
                                )
                            )
                        elif isinstance(pattern_data, dict):
                            pattern_data["pattern_name"] = pattern_name
                            print(
                                "       ✅ Dict format: {}".format(
                                    list(pattern_data.keys())
                                )
                            )
                        else:
                            print(
                                "       ⚠️ Unexpected pattern_data type: {}".format(
                                    type(pattern_data)
                                )
                            )
                            continue
                        try:
                            workflow_patterns[pattern_name] = WorkflowPattern(
                                **pattern_data
                            )
                            print(f"     ✅ {pattern_name}")
                        except Exception as e:
                            print(
                                "     ❌ Error creating WorkflowPattern for {}: {}".format(
                                    pattern_name, e
                                )
                            )
                            print(f"        Pattern data: {pattern_data}")
                            continue
                else:
                    print(
                        "⚠️ Unexpected workflow_patterns type: {} (should be dict)".format(
                            type(workflow_patterns_raw)
                        )
                    )

            # Handle validation_rules safely
            print("   🔧 Processing validation_rules...")
            validation_rules = raw_schema.get("validation_rules", {})
            if validation_rules is None:
                validation_rules = {}

            # Convert validation rules from list format to dict format
            normalized_validation_rules = {}
            if isinstance(validation_rules, dict):
                for asset_name, rules in validation_rules.items():
                    print(
                        "     🔍 Processing validation for '{}': {}".format(
                            asset_name, type(rules)
                        )
                    )
                    if isinstance(rules, list):
                        print(f"       🔄 Converting list to dict for {asset_name}")
                        # Flatten list of dicts into single dict
                        combined_rules = {}
                        for rule_item in rules:
                            if isinstance(rule_item, dict):
                                combined_rules.update(rule_item)
                                print(
                                    "         📝 Added rules: {}".format(
                                        list(rule_item.keys())
                                    )
                                )
                            elif isinstance(rule_item, str):
                                # Handle simple string rules
                                combined_rules[rule_item] = True
                                print(f"         📝 Added string rule: {rule_item}")
                        normalized_validation_rules[asset_name] = combined_rules
                        print(
                            "       ✅ Final rules: {}".format(
                                list(combined_rules.keys())
                            )
                        )
                    elif isinstance(rules, dict):
                        normalized_validation_rules[asset_name] = rules
                        print(f"       ✅ Dict format: {list(rules.keys())}")
                    else:
                        print(
                            "Warning: Expected dict for validation rules, got {} for {}".format(
                                type(rules), asset_name
                            )
                        )

            validation_rules = normalized_validation_rules
            print(
                "   ✅ Validation rules processed: {}".format(
                    list(validation_rules.keys())
                )
            )

            # Create the full schema definition
            print("   🔧 Creating SchemaDefinition...")
            schema_data = {
                "task_type": raw_schema.get("task_type", ""),
                "description": raw_schema.get("description", ""),
                "required_assets": raw_schema.get("required_assets", []),
                "optional_assets": raw_schema.get("optional_assets", []),
                "asset_definitions": asset_defs,
                "validation_rules": validation_rules,
                "workflow_patterns": workflow_patterns,
                "dependencies": raw_schema.get("dependencies", []),
                "produces_for": raw_schema.get("produces_for", []),
            }

            schema = SchemaDefinition(**schema_data)
            print("   ✅ SchemaDefinition created successfully")
            return schema

        except Exception as e:
            print(f"❌ Error loading {schema_file.name}: {e}")
            print(f"   Error type: {type(e).__name__}")
            import traceback

            traceback.print_exc()  # pragma: no cover
            raise  # pragma: no cover

    def load_all_schemas(self) -> None:
        """Load all YAML schema files from the schema directory."""
        schema_files = list(self.schema_dir.glob("*.yaml"))

        if not schema_files:
            print(f"No .yaml files found in {self.schema_dir}")
            return

        print(f"Found {len(schema_files)} YAML files:")
        for schema_file in schema_files:
            print(f"  - {schema_file.name}")

        for schema_file in schema_files:
            try:
                schema = self.load_schema(schema_file)
                self.schemas[schema.task_type] = schema
                print(f"✅ Loaded schema: {schema.task_type}")
            except Exception as e:
                print(f"❌ Error loading {schema_file.name}: {e}")
                # Continue with other files instead of stopping

    def generate_asset_model(self, asset_name: str, asset_def: AssetDefinition) -> str:
        """Generate Pydantic field definition for an asset."""

        # Map CCTBX types to Python types for Pydantic
        type_mapping = {
            "cctbx.xray.structure": "Any",  # Will need custom validation
            "cctbx.miller.array": "Any",
            "cctbx.miller.set": "Any",
            "cctbx.array_family.flex.vec3_double": "Any",
            "cctbx.array_family.flex.double": "Any",
            "dict": "Dict[str, Any]",
            "str": "str",
            "float": "float",
            "int": "int",
            "bool": "bool",
            "bytes": "bytes",
        }

        python_type = type_mapping.get(asset_def.data_type, "Any")

        # Build field definition
        field_kwargs = []

        if asset_def.description:
            field_kwargs.append(f'description="{asset_def.description}"')

        # Add constraints based on asset definition
        if asset_def.valid_range:
            if python_type == "float":
                field_kwargs.append(
                    "ge={}, le={}".format(
                        asset_def.valid_range[0], asset_def.valid_range[1]
                    )
                )

        if asset_def.default_values and len(asset_def.default_values) == 1:
            default_val = list(asset_def.default_values.values())[0]
            field_kwargs.append(f"default={repr(default_val)}")

        field_def = (
            "Field({})".format(", ".join(field_kwargs)) if field_kwargs else "..."
        )

        return f"    {asset_name}: {python_type} = {field_def}"

    def generate_validators(self, schema: SchemaDefinition) -> List[str]:
        """Generate custom validators for CCTBX-specific validation."""

        validators = []

        # Generate validators for each asset with validation rules
        if schema.validation_rules:
            for asset_name, rules in schema.validation_rules.items():
                if asset_name in schema.asset_definitions:
                    asset_def = schema.asset_definitions[asset_name]

                    validator_lines = [
                        f"    @field_validator('{asset_name}')",
                        "    @classmethod",
                        f"    def validate_{asset_name}(cls, v):",
                        f'        """Validate {asset_def.description}"""',
                    ]

                    # Add CCTBX-specific validation
                    if asset_def.data_type == "cctbx.xray.structure":
                        validator_lines.extend(
                            [
                                "        if not hasattr(v, 'scatterers'):",
                                "            raise ValueError('xray_structure must have scatterers')",
                                "        if not hasattr(v, 'unit_cell'):",
                                "            raise ValueError('xray_structure must have unit_cell')",
                            ]
                        )

                    elif asset_def.data_type == "cctbx.miller.array":
                        validator_lines.extend(
                            [
                                "        if not hasattr(v, 'indices'):",
                                "            raise ValueError('miller_array must have indices')",
                                "        if not hasattr(v, 'data'):",
                                "            raise ValueError('miller_array must have data')",
                            ]
                        )

                        if asset_def.must_be_complex:
                            validator_lines.append(
                                "        if not v.is_complex_array():"
                            )
                            validator_lines.append(
                                "            raise ValueError('miller_array must be complex')"
                            )

                        if asset_def.data_must_be_positive:
                            validator_lines.append(
                                "        if (v.data() < 0).count(True) > 0:"
                            )
                            validator_lines.append(
                                "            raise ValueError('miller_array data must be positive')"
                            )

                    # Handle validation rules - rules is now a dict, not a list
                    if isinstance(rules, dict):
                        for rule_name, rule_value in rules.items():
                            if rule_name == "finite_values_only" and rule_value:
                                validator_lines.extend(
                                    [
                                        "        import numpy as np",
                                        "        if hasattr(v, 'data') and not np.all(np.isfinite(v.data())):",
                                        "            raise ValueError('All values must be finite')",
                                    ]
                                )
                    else:
                        print(
                            "Warning: Expected dict for validation rules, got {} for {}".format(
                                type(rules), asset_name
                            )
                        )

                    validator_lines.append("        return v")
                    validator_lines.append("")

                    validators.extend(validator_lines)

        return validators

    def generate_bundle_model(self, schema: SchemaDefinition) -> str:
        """Generate a complete Pydantic model for a bundle type."""

        class_name = "{}Bundle".format(schema.task_type.title().replace("_", ""))

        lines = [
            f"class {class_name}(BaseModel):",
            '    """',
            f"    {schema.description}",
            "    ",
            f"    Generated from {schema.task_type}.yaml",
            '    """',
            "    ",
            "    # Bundle metadata",
            '    bundle_type: Literal["{}"] = "{}"'.format(
                schema.task_type, schema.task_type
            ),
            "    created_at: datetime = Field(default_factory=datetime.now)",
            "    bundle_id: Optional[str] = None",
            "    checksum: Optional[str] = None",
            "    ",
        ]

        # Add required assets
        lines.append("    # Required assets")
        for asset_name in schema.required_assets:
            if asset_name in schema.asset_definitions:
                asset_def = schema.asset_definitions[asset_name]
                lines.append(self.generate_asset_model(asset_name, asset_def))

        lines.append("")

        # Add optional assets
        if schema.optional_assets:
            lines.append("    # Optional assets")
            for asset_name in schema.optional_assets:
                if asset_name in schema.asset_definitions:
                    asset_def = schema.asset_definitions[asset_name]
                    field_line = self.generate_asset_model(asset_name, asset_def)
                    # Make it optional
                    field_line = field_line.replace(
                        " = Field(", " = Field(default=None, "
                    )
                    if " = ..." in field_line:
                        field_line = field_line.replace(" = ...", " = None")
                    lines.append(field_line)

        lines.append("")

        # Add custom validators
        validator_lines = self.generate_validators(schema)
        lines.extend(validator_lines)

        # Add utility methods
        lines.extend(
            [
                "    def calculate_checksum(self) -> str:",
                '        """Calculate checksum of bundle contents."""',
                "        # Implementation would hash all asset contents",
                "        import json",
                "        content = self.dict(exclude={'checksum', 'created_at', 'bundle_id'})",
                "        content_str = json.dumps(content, sort_keys=True, default=str)",
                "        return hashlib.sha256(content_str.encode()).hexdigest()[:16]",
                "",
                "    def validate_dependencies(self, available_bundles: Dict[str, 'BaseModel']) -> bool:",
                '        """Validate that all dependencies are satisfied."""',
                "        required_deps: List[str] = {} or []".format(
                    schema.dependencies
                ),
                "        for dep in required_deps:",
                "            if dep not in available_bundles:",
                "                raise ValueError(f'Missing dependency: {dep}')",
                "        return True",
                "",
            ]
        )

        return "\n".join(lines)

    def generate_all_models(self) -> str:
        """Generate Pydantic models for all loaded schemas."""

        if not self.schemas:
            self.load_all_schemas()

        # Generate imports
        imports = [
            "# Auto-generated Pydantic models from YAML schemas",
            "# DO NOT EDIT - regenerate using SchemaGenerator",
            "",
            "from typing import Dict, List, Optional, Any, Literal",
            "from pydantic import BaseModel, Field, field_validator",
            "from datetime import datetime",
            "import hashlib",
            "",
        ]

        # Generate each bundle model
        models = []
        for schema in self.schemas.values():
            model_code = self.generate_bundle_model(schema)
            models.append(model_code)

        return "\n".join(imports + models)

    def write_generated_models(self, output_file: Path) -> None:
        """Write generated models to a Python file."""

        generated_code = self.generate_all_models()

        with open(output_file, "w") as f:
            f.write(generated_code)

        print(f"Generated Pydantic models written to: {output_file}")


def main() -> int:
    """Main function to auto-generate models from default directories."""
    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Generate Pydantic models from YAML schema definitions"
    )
    parser.add_argument(
        "--schemas-dir",
        type=Path,
        default=Path("src/agentbx/schemas/definitions"),
        help="Directory containing YAML schema files (default: src/agentbx/schemas/definitions)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("src/agentbx/schemas/generated.py"),
        help="Output file for generated Pydantic models (default: src/agentbx/schemas/generated.py)",
    )
    parser.add_argument(
        "--watch", action="store_true", help="Watch for changes and auto-regenerate"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    args = parser.parse_args()  # pragma: no cover

    # Check if directories exist
    if not args.schemas_dir.exists():
        print(f"❌ Schema directory not found: {args.schemas_dir}")
        print(f"💡 Create it with: mkdir -p {args.schemas_dir}")
        return 1  # pragma: no cover

    # Ensure output directory exists
    args.output.parent.mkdir(parents=True, exist_ok=True)

    print("🏭 AgentBx Schema Generator")
    print("=" * 40)
    print(f"📂 Schemas directory: {args.schemas_dir}")
    print(f"📄 Output file: {args.output}")

    # Generate schemas
    generator = SchemaGenerator(args.schemas_dir)

    try:
        # Load all schemas
        if args.verbose:
            print("\n📖 Loading YAML schemas...")
        generator.load_all_schemas()

        if not generator.schemas:
            print("⚠️  No YAML schema files found!")
            print(f"   Add .yaml files to: {args.schemas_dir}")
            return 1  # pragma: no cover

        # Show loaded schemas
        print(f"\n✅ Loaded {len(generator.schemas)} schemas:")
        for schema_name, schema in generator.schemas.items():
            required_count = len(schema.required_assets)
            optional_count = len(schema.optional_assets or [])
            print(
                "   - {} ({}, {} optional assets)".format(
                    schema_name, required_count, optional_count
                )
            )

        # Generate and write models
        if args.verbose:
            print("\n🔧 Generating Pydantic models...")
        generator.write_generated_models(args.output)

        print(f"\n🎉 Successfully generated: {args.output}")
        print(f"📊 Generated {len(generator.schemas)} bundle classes")

        # Show usage hint
        print("\n💡 Usage:")
        print(f"   from {args.output.stem} import AtomicModelDataBundle")

        if args.watch:
            watch_for_changes(generator, args.schemas_dir, args.output, args.verbose)

        return 0

    except Exception as e:
        print(f"❌ Error generating schemas: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()  # pragma: no cover
        return 1  # pragma: no cover


def watch_for_changes(
    generator: Any, schemas_dir: Path, output_file: Path, verbose: bool = False
) -> None:
    """Watch for changes in schema files and auto-regenerate."""
    try:
        import time

        from watchdog.events import FileSystemEventHandler
        from watchdog.observers import Observer

        print(f"\n👀 Watching {schemas_dir} for changes... (Ctrl+C to stop)")

        class SchemaChangeHandler(FileSystemEventHandler):
            def __init__(self, generator, output_file, verbose):
                self.generator = generator
                self.output_file = output_file
                self.verbose = verbose
                self.last_regeneration = 0

            def on_modified(self, event):
                if not event.is_directory and event.src_path.endswith(".yaml"):
                    # Debounce rapid changes
                    now = time.time()
                    if now - self.last_regeneration < 1.0:
                        return
                    self.last_regeneration = now

                    print(
                        "\n🔄 {} changed, regenerating...".format(
                            Path(event.src_path).name
                        )
                    )
                    try:
                        self.generator.schemas.clear()
                        self.generator.load_all_schemas()
                        self.generator.write_generated_models(self.output_file)
                        print("✅ Regenerated successfully!")
                    except Exception as e:
                        print(f"❌ Regeneration failed: {e}")

        event_handler = SchemaChangeHandler(generator, output_file, verbose)
        observer = Observer()
        observer.schedule(event_handler, str(schemas_dir), recursive=True)
        observer.start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n🛑 Stopping file watcher...")
            observer.stop()
        observer.join()

    except ImportError:
        print(
            "⚠️  Install watchdog for file watching: pip install watchdog"
        )  # pragma: no cover


def quick_generate() -> None:
    """Quick generation using default paths."""
    schema_dir = Path("src/agentbx/schemas/definitions")
    output_file = Path("src/agentbx/schemas/generated.py")

    if not schema_dir.exists():
        raise FileNotFoundError(f"Schema directory not found: {schema_dir}")

    generator = SchemaGenerator(schema_dir)
    generator.write_generated_models(output_file)
    print(f"✅ Generated {len(generator.schemas)} schemas → {output_file}")


if __name__ == "__main__":
    exit(main())  # pragma: no cover
