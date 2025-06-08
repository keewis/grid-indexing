# Configuration file for the Sphinx documentation builder.

# -- Project information -----------------------------------------------------

import datetime as dt

project = "grid-indexing"
year = dt.datetime.now().year
author = "Justus Magin"
copyright = f"{year}, {author}"

# -- General configuration ---------------------------------------------------
extensions = [
    "sphinx.ext.autosummary",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "jupyter_sphinx",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
root_doc = "index"

# -- Extension configuration -------------------------------------------------
# autosummary + autodoc
autosummary_generate = True
autodoc_typehints = "none"

# napoleon
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_use_param = False
napoleon_use_rtype = False

# myst
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

# -- Options for HTML output -------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_static_path = ["_static"]
