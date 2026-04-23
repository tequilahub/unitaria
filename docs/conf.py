# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import subprocess
import sys

sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "unitaria"
copyright = "2025, Matthias Deiml"
author = "Matthias Deiml"

version = os.environ.get("DOCS_VERSION", "git")
release = version
try:
    tags = subprocess.check_output(["git", "tag", "--sort=-v:refname"]).decode("utf-8").splitlines()
except Exception:
    tags = []

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.viewcode",
    "sphinx.ext.todo",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.imgmath",
]

napoleon_google_docstring = True
napoleon_use_param = False
napoleon_use_ivar = True

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
autosummary_ignore_module_all = False
autosummary_generate = True
default_role = "py:obj"

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "rich": ("https://rich.readthedocs.io/en/stable/", None),
    "tequila": ("https://tequilahub.github.io/tequila-tutorials/docs/sphinx", "tequila.inv"),
}

imgmath_image_format = "svg"
imgmath_use_preview = True
imgmath_font_size = 16
imgmath_latex_preamble = "\\usepackage{newtx}"

language = "en"

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
pygments_style = "sphinx"
pygments_dark_style = "github-dark"
html_static_path = ["_static"]
html_css_files = [
    "css/imgmath_furo.css",
]
html_sidebars = {
    "**": [
        "sidebar/scroll-start.html",
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/navigation.html",
        "sidebar/scroll-end.html",
        "versions.html",
    ]
}

html_context = {
  "current_version" : version,
  "versions" : ["git"] + tags,
}

# -- Options for todo extension ----------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/extensions/todo.html#configuration

todo_include_todos = True
