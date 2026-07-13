# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# =============================================================================
# Project information
# =============================================================================
project = 'OpenDrug'
copyright = '2024–2026, OpenDrug Contributors'
author = 'OpenDrug Contributors'
release = '0.1.0'
version = '0.1'

# =============================================================================
# General configuration
# =============================================================================
import os
import sys
sys.path.insert(0, os.path.abspath('../opendrug'))

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
    'sphinx.ext.doctest',
    'sphinx_autodoc_typehints',
    'sphinx_copybutton',
    'myst_parser',
] 

# Napoleon settings (NumPy-style docstrings)
napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_use_param = True
napoleon_use_rtype = True

# Stub out the custom substitution references used inside model docstrings.
# These are short mathematical expressions written with bare pipes (e.g. "|z1 - z2|")
# which docutils otherwise treats as undefined substitution references.  Defining
# them here keeps the documentation build green without having to edit every
# source file's docstring.
rst_prolog = r"""
.. |z1 - z2|       replace:: :math:`|z_1| - |z_2|`
.. |emb1 - emb2|   replace:: :math:`|\text{emb}_1| - |\text{emb}_2|`
.. |x_i - x_j|     replace:: :math:`|x_i| - |x_j|`
.. |p1 - p2|       replace:: :math:`|p_1| - |p_2|`
.. |proj1 - proj2| replace:: :math:`|\text{proj}_1| - |\text{proj}_2|`
.. |z_dif|         replace:: z_dif
.. |z_mul|         replace:: z_mul
"""

# Treat auto-generated API summaries as documentation pages.
# Set to True so Sphinx generates the stubs under docs/api/generated/ during
# the build; the docs/conf.py serves as the single source of truth for this
# flag so neither the CI workflow nor .readthedocs.yaml needs to override it.
autosummary_generate = True

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'torch':  ('https://pytorch.org/docs/stable', None),
    'numpy':  ('https://numpy.org/devdocs', None),
    'sklearn': ('https://scikit-learn.org/stable', None),
}

# Auto settings
# Mock modules that the framework imports but are not installed in our
# documentation environment.  Sphinx will treat them as empty placeholders so
# autodoc / autosummary can still parse class / function signatures without
# executing the modules.
autodoc_mock_imports = [
    # Third-party heavy dependencies that are not installed in the docs build
    # environment.  Sphinx will treat them as empty placeholders so autodoc can
    # still parse class / function signatures without executing the modules.
    # NOTE: 'opendrug' itself is intentionally NOT mocked so that autodoc can
    # introspect the real module hierarchy and generate API documentation.
    'torch',
    'torch.nn',
    'torch.nn.functional',
    'torch.cuda',
    'torch.cuda.amp',
    'torch.optim',
    'torch.utils.data',
    'torch_geometric',
    'torch_geometric.data',
    'torch_geometric.nn',
    'torch_scatter',
    'torch_sparse',
    'dgl',
    'dgl.nn',
    'dgl.function',
    'rdkit',
    'rdkit.Chem',
    'rdkit.Chem.AllChem',
    'rdkit.Chem.rdchem',
    'subword_nmt',
    'subword_nmt.apply_bpe',
    'gensim',
    'matplotlib',
    'matplotlib.pyplot',
    'sklearn',
    'sklearn.metrics',
    'tqdm',
    'numpy',
    'scipy',
    'scipy.stats',
    'pandas',
    'networkx',
    'yaml',
    'logging',
]

autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'undoc-members': False,
    'show-inheritance': True,
}
autodoc_typehints = 'description'
autodoc_typehints_format = 'short'

# sys.path was already set above.

# =============================================================================
# Options for HTML output
# =============================================================================
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'canonical_url': '',
    'analytics_id': '',
    'logo_only': False,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'vcs_pageview_mode': '',
    'style_nav_header_background': '#2c3e50',
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
}

# Static files
html_static_path = []
# html_favicon = '_static/favicon.ico'  # uncomment after adding a favicon

# =============================================================================
# Options for LaTeX / PDF output
# =============================================================================
latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '11pt',
    'preamble': (
        r'\usepackage{amsmath,amssymb}'
        r'\usepackage{booktabs}'
        r'\usepackage{makecell}'
    ),
}
latex_documents = [
    (('index'), 'opendrug.tex'), 'OpenDrug Documentation', 'OpenDrug Contributors',
]

# =============================================================================
# Source suffix
# =============================================================================
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
myst_enable_extensions = [
    'colon_fence',
    'deflist',
    'dollarmath',
    'fieldlist',
    'html_admonition',
    'html_image',
    'linkify',
    'strikethrough',
    'substitution',
    'tasklist',
] 
myst_heading_anchors = 3

# =============================================================================
# Exclude patterns
# =============================================================================
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# =============================================================================
# Todo / experimental feature flags (shown with toggle)
# =============================================================================
todo_include_todos = True
