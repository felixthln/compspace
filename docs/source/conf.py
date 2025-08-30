# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------
# Insert the path to the source directory to the system paths

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2] / 'src'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'compspace'
copyright = '2025, Felix Thelen'
author = 'Felix Thelen'
release = '1.0.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'nbsphinx',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary'
]

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']

# Define the theme options like title
html_theme_options = {
    'logo': {
        'text': 'compspace'
    },
    'github_url': 'https://github.com/felixthln/compspace',
    'back_to_top_button': False
}

# If True, show link to rst source on rendered HTML pages
html_show_sourcelink = False  # Remove 'view source code' from top of page
# Define that the type hints will be shown in the description of the function references rather than in the signature
autodoc_typehints = 'description'
# Include the docstring of the __init__ method in the documentation too instead of the class docstring
autoclass_content = 'init'
# When generating the documentation automatically, generate the order of the method descriptions based on the source
autodoc_member_order = 'bysource'
# Disable the execution of the notebooks as they cannot be run by readthedocs
# nbsphinx_execute = 'never'
nbsphinx_execute_arguments = ['--ExecutePreprocessor.cwd=..']
