"""Sphinx configuration."""
project = "Agentbx"
author = "Petrus Zwart"
copyright = "2025, Petrus Zwart"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_parser",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "show-inheritance": True,
    "special-members": "__init__",
}
html_theme = "furo"
html_theme_options = {
    "navigation_with_keys": True,
    "source_repository": "https://github.com/phzwart/agentbx",
    "source_branch": "main",
    "source_directory": "docs/",
}
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "redis": ("https://redis.readthedocs.io/en/stable/", None),
    "pydantic": ("https://docs.pydantic.dev/latest/", None),
}
