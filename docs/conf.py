import datetime
import os
import shutil
import sys

# make sure that the project root directory is in path
sys.path.insert(0, os.path.abspath(".."))

# -- Project information -----------------------------------------------------

current_year = datetime.datetime.now().year
project = "scikit-fingerprints"
project_copyright = f"2019 - {current_year} (MIT License)"
author = "scikit-fingerprints developers"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx_copybutton",
    "nbsphinx",
]

autodoc_mock_imports = ["torch"]

autodoc_default_options = {
    "inherited-members": True,
    "members": "fit,fit_transform,transform",
}

templates_path = ["_templates"]
exclude_patterns = [
    "**.ipynb_checkpoints",
    ".DS_Store",
    "_build",
    "Thumbs.db",
]

intersphinx_mapping = {
    "sklearn": ("https://scikit-learn.org/stable/", None),
}

# -- Options for HTML output -------------------------------------------------

html_theme = "pydata_sphinx_theme"

copybutton_exclude = ".linenos, .gp, .go"

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/scikit-fingerprints/scikit-fingerprints",
            "icon": "fa-brands fa-github",
        },
    ],
    "logo": {
        "alt_text": "scikit-fingerprints logo",
        "image_relative": "logos/skfp-logo-no-text.png",
        "image_light": "logos/skfp-logo-horizontal-text-black.png",
        "image_dark": "logos/skfp-logo-horizontal-text-white.png",
    },
}
html_static_path = ["_static"]
html_css_files = ["custom.css"]


# Sphinx cannot reach outside to parallel dirs, so we copy "examples" directory
# we copy "examples" directory from project root to docs/examples


def all_but_ipynb(directory: str, contents: list[str]) -> list[str]:
    return [
        c
        for c in contents
        if os.path.isfile(os.path.join(directory, c)) and not c.endswith(".ipynb")
    ]


shutil.rmtree("./examples", ignore_errors=True)
shutil.copytree(
    src="../examples",
    dst="./examples",
    ignore=all_but_ipynb,
)
