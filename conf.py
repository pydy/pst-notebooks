import os
import multiprocessing

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Peter's Mechanics Examples"
copyright = '2025, Peter Stahlecker'
author = 'Peter Stahlecker'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.mathjax',
    'sphinx_gallery.gen_gallery',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

# -- Sphinx Gallery configuration --------------------------------------------
sphinx_gallery_conf = {
    'copyfile_regex': r'.*\.(npy|csv|yml|txt)',
    'examples_dirs': ['gallery'],
    'gallery_dirs': ['autogallery'],
    'matplotlib_animations': True,
    'parallel': multiprocessing.cpu_count(),
    'remove_config_comments': True,
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
