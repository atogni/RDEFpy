"""Sphinx configuration for the RDEFpy documentation.

Build locally:

    uv run --group docs python -m sphinx -W --keep-going -b html docs docs/_build/html
"""

from rdefpy import __version__

project = "RDEFpy"
author = "Andrea Togni"
copyright = "2026, Morehead State University"
release = __version__
version = ".".join(__version__.split(".")[:2])

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]

# NumPy-style docstrings throughout the package.
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_rtype = False
# "Attributes" sections as fields, not entries that clash with properties.
napoleon_use_ivar = True
# Link the type names written in docstrings ("Path", "np.ndarray", ...).
napoleon_preprocess_types = True
napoleon_type_aliases = {
    "Path": "pathlib.Path",
    "datetime": "datetime.datetime",
    "Fraction": "fractions.Fraction",
    "np.ndarray": "numpy.ndarray",
    "np.datetime64": "numpy.datetime64",
    "np.dtype": "numpy.dtype",
    "np.complex64": "numpy.complex64",
    "np.complex128": "numpy.complex128",
}
# Docstring types with no link target (sphinx -n); read as plain text.
nitpick_ignore = [("py:class", t) for t in (
    "list[ProductEntry]", "list[Record]", "list[Scan]", "list[str]", "numpy dtype",
    "numpy.complex64", "numpy.complex128", "text stream")]

templates_path = ["_templates"]

autosummary_generate = True
autodoc_member_order = "bysource"
# Types come from the NumPy-style docstrings; annotations would show the
# private view classes behind pf.samples and friends.
autodoc_typehints = "none"
autodoc_default_options = {"members": True, "show-inheritance": False}

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
}

exclude_patterns = ["_build"]
default_role = "code"

html_theme = "furo"
html_title = f"RDEFpy {release}"
html_static_path = []
# Furo shows API names and their module prefix in its "problematic" red; use text colours.
_api_name = {"color-api-name": "var(--color-foreground-primary)",
             "color-api-pre-name": "var(--color-foreground-secondary)"}
html_theme_options = {"light_css_variables": _api_name, "dark_css_variables": _api_name}
