# pylint: disable=redefined-builtin,invalid-name
import os
from importlib.metadata import metadata
from pathlib import Path

# -- Project information

_metadata = metadata("arviz-base")

project = _metadata["Name"]
author = _metadata["Author-email"].split("<", 1)[0].strip()
copyright = f"2022, {author}"

version = _metadata["Version"]
if os.environ.get("READTHEDOCS", False):
    rtd_version = os.environ.get("READTHEDOCS_VERSION", "")
    if "." not in rtd_version and rtd_version.lower() != "stable":
        version = "dev"
else:
    branch_name = os.environ.get("BUILD_SOURCEBRANCHNAME", "")
    if branch_name == "main":
        version = "dev"
release = version


# -- General configuration

extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.autosummary",
    "sphinx.ext.extlinks",
    "numpydoc",
    "myst_nb",
    "sphinx_copybutton",
    "sphinx_design",
    "jupyter_sphinx",
]

templates_path = ["_templates"]

exclude_patterns = [
    "Thumbs.db",
    ".DS_Store",
    ".ipynb_checkpoints",
    "**/*.template.rst",
    "**/*.part.rst",
]
suppress_warnings = ["mystnb.unknown_mime_type"]

backend_modules = {
    "index": "arviz_plots.backend",
    "matplotlib": "arviz_plots.backend.matplotlib",
    "bokeh": "arviz_plots.backend.bokeh",
}
api_backend_dir = Path(__file__).parent.resolve() / "api" / "backend"
with open(api_backend_dir / "interface.template.rst", "r", encoding="utf-8") as f:
    interface_template = f.read()
for file, module in backend_modules.items():
    with open(api_backend_dir / f"{file}.part.rst", "r", encoding="utf-8") as f:
        intro = f.read()
    with open(api_backend_dir / f"{file}.rst", "w", encoding="utf-8") as f:
        f.write(f"{intro}\n\n.. automodule:: {module}\n\n" + interface_template)

# The reST default role (used for this markup: `text`) to use for all documents.
default_role = "autolink"

# If true, '()' will be appended to :func: etc. cross-reference text.
add_function_parentheses = False

# -- Options for extensions

extlinks = {
    "issue": ("https://github.com/arviz-devs/arviz-base/issues/%s", "GH#%s"),
    "pull": ("https://github.com/arviz-devs/arviz-base/pull/%s", "PR#%s"),
}

nb_execution_mode = "auto"
nb_execution_excludepatterns = ["*.ipynb"]
nb_kernel_rgx_aliases = {".*": "python3"}
myst_enable_extensions = ["colon_fence", "deflist", "dollarmath", "amsmath", "linkify"]

autosummary_generate = True
autodoc_typehints = "none"
autodoc_default_options = {
    "members": False,
}

numpydoc_show_class_members = False
numpydoc_xref_param_type = True
numpydoc_xref_ignore = {"of", "or", "optional", "scalar"}
singulars = ("int", "list", "dict", "float")
numpydoc_xref_aliases = {
    "DataArray": ":class:`xarray.DataArray`",
    "Dataset": ":class:`xarray.Dataset`",
    "DataTree": ":class:`datatree.DataTree`",
    **{f"{singular}s": f":any:`{singular}s <{singular}>`" for singular in singulars},
}

intersphinx_mapping = {
    "arviz_org": ("https://www.arviz.org/en/latest/", None),
    "dask": ("https://docs.dask.org/en/latest/", None),
    "datatree": ("https://xarray-datatree.readthedocs.io/en/latest/", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "python": ("https://docs.python.org/3/", None),
    "xarray": ("https://docs.xarray.dev/en/stable/", None),
}

# -- Options for HTML output

html_theme = "furo"
# html_static_path = ["_static"]
