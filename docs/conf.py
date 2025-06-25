"""Sphinx configuration."""
project = "Agentbx"
author = "Petrus Zwart"
copyright = "2025, Petrus Zwart"
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_click",
    "myst_parser",
]
autodoc_typehints = "description"
html_theme = "furo"
