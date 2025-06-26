"""Command-line interface."""

import click


@click.command()
@click.version_option()
def main() -> None:
    """Agentbx."""


if __name__ == "__main__":
    main(prog_name="agentbx")  # pragma: no cover
