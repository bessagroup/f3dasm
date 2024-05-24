# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html


import os
import sys

from sphinx_gallery.sorting import FileNameSortKey

# -- Search path for extensions and modules -----------------------------------
# If extensions or Python modules are in a different directory than this file,
# then add these directories to sys.path so that Sphinx can search for them
# Source: https://docs.python.org/3/library/sys.html#sys.path

sys.path.insert(0, os.path.abspath('../..'))
sys.path.insert(0, os.path.abspath('../../src'))
src_dir = [f.path for f in os.scandir(
    os.path.abspath('../../src/')) if f.is_dir()]
for path in src_dir:
    sys.path.insert(0, path)

# -- Project information ------------------------------------------------------
# Source: https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'f3dasm'
author = 'Martin van der Schelling'
copyright = '2024, Martin van der Schelling'
version = '1.5.0'
release = '1.5.0'


# -- General configuration ----------------------------------------------------


# Add any Sphinx extension module names here, as strings. They can be extensions
# coming with Sphinx (named 'sphinx.ext.*') or your custom ones.

extensions = ['sphinx.ext.duration',
              'sphinx_rtd_theme',
              'sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              'sphinx.ext.autosummary',
              'sphinx.ext.intersphinx',
              'sphinx.ext.viewcode',
              'sphinx_autodoc_typehints',
              'sphinx_tabs.tabs',
              'sphinx_gallery.gen_gallery',]

sphinx_gallery_conf = {
    'examples_dirs': ['../../examples'],   # path to your example scripts
    'gallery_dirs': ['auto_examples'],
    'reference_url': {'sphinx_gallery': None, },
    'backreferences_dir': 'gen_modules/backreferences',
    'doc_module': ('f3dasm',),
    "filename_pattern": r"/*\.py",
    "within_subsection_order": FileNameSortKey,
}

# Source: https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-source_suffix
source_suffix = {'.rst': 'restructuredtext', }

# Source: https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-root_doc
root_doc = 'index'

# Source: https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-exclude_patterns
exclude_patterns = ['_build', ]

# Source: https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-templates_path
templates_path = ['_templates', ]

# Source: https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-pygments_style
pygments_style = 'sphinx'

# Source: https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-add_module_names
add_module_names = False

# Source: https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-toc_object_entries
toc_object_entries = False

# Source: https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-toc_object_entries_show_parents
toc_object_entries_show_parents = 'hide'

# Source: https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-show_authors
show_authors = True

# Source: https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#confval-autodoc_default_options
autodoc_default_options = {'private-members': False}


# -- Extensions configuration -------------------------------------------------

# autodoc: https://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#module-sphinx.ext.autodoc
autoclass_content = 'both'

# napoleon: https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True

# autosummary: https://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html#module-sphinx.ext.autosummary
autosummary_generate = True
autosummary_generate_overwrite = True
autosummary_imported_members = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/reference/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "pandas": ('https://pandas.pydata.org/docs/', None),
    "xarray": ('https://docs.xarray.dev/en/stable/', None),
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

# Source: https://docs.readthedocs.io/en/stable/faq.html#how-do-i-change-behavior-when-building-with-read-the-docs
# Source: https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-html_theme
# Source: https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-html_theme_path
on_rtd = os.environ.get('READTHEDOCS', None) == 'True'
if on_rtd:
    import sphinx_rtd_theme
    html_theme = 'sphinx_rtd_theme'
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]
else:
    # Source: https://sphinx-rtd-theme.readthedocs.io/en/stable/index.html
    # Requires installation of Python package 'sphinx_rtd_theme'
    import sphinx_rtd_theme
    html_theme = 'sphinx_rtd_theme'
    html_theme_path = [sphinx_rtd_theme.get_html_theme_path()]

# Source: https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-html_static_path
html_static_path = ['_static', ]

# Source: https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-html_css_files
html_css_files = ['readthedocs-custom.css', 'theme_overrides.css', ]

# Source: https://www.sphinx-doc.org/en/master/usage/configuration.html#confval-html_show_sourcelink
html_show_sourcelink = True
