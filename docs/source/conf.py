'''
Sphinx configuration for the AeroOpt documentation.
'''

import os
import sys
from datetime import date

# Make the package importable without installing it.
sys.path.insert(0, os.path.abspath('../..'))

import aeroopt  # noqa: E402

# -- Project information -----------------------------------------------------

project = 'AeroOpt'
author = 'Runze Li'
copyright = f'{date.today().year}, {author}'
release = aeroopt.__version__
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.todo',
]

exclude_patterns = []

# The docstrings use `'''...'''` with NumPy-style "Parameters / Returns"
# sections, sometimes with a trailing colon ("Parameters:").
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True

autosummary_generate = True

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'member-order': 'bysource',
}
autodoc_typehints = 'description'
autodoc_class_signature = 'separated'

# `smt` is an optional dependency; do not require it to build the docs.
autodoc_mock_imports = ['smt']

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
}

todo_include_todos = False

nitpick_ignore_regex = [
    ('py:class', r'.*'),
]

# -- Options for HTML output -------------------------------------------------

html_theme = 'furo'
html_static_path = ['_static']
html_title = f'AeroOpt {release}'

html_theme_options = {
    'source_repository': 'https://github.com/swayli94/AeroOpt',
    'source_branch': 'main',
    'source_directory': 'docs/source/',
}
