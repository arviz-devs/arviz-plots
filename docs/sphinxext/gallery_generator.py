# pylint: disable=invalid-name
"""Generate images and full gallery pages from python scripts."""
import json
import os
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
from docutils import statemachine
from docutils.parsers.rst import Directive
from sphinx.util import logging

logger = logging.getLogger(__name__)

dir_title_map = {
    "mixed": "Mixed plots",
    "distribution": "Distribution visualization",
    "distribution_comparison": "Distribution comparison",
    "inference_diagnostics": "Inference diagnostics",
    "model_criticism": "Model criticism",
    "model_comparison": "Model comparison",
}

toctree_template = """
## {title}

:::{{toctree}}
:hidden:
:caption: {title}

{files}
:::
"""

grid_item_template = """
::::{{grid-item-card}}
:link: {basename}
:link-type: doc
:text-align: center
:shadow: none
:class-card: example-gallery

:::{{div}} example-img-plot-overlay
{description}
:::

:::{{image}} /gallery/_images/{basename}.png
:alt:

:::

+++
{title}
::::
"""

minigallery_item_template_rst = """
.. grid-item-card::
   :link: {refname}
   :link-type: ref
   :text-align: center
   :shadow: none
   :class-card: example-gallery

   .. div:: example-img-plot-overlay

      {description}

   .. image:: /gallery/_images/{basename}.png
      :alt:

   +++
   {title}
"""

minigallery_in_example = """
## Other examples with `{fun}`

```{{eval-rst}}
.. minigallery:: {fun}
```
"""


def main(app):
    """Generate thumbnail images with matplotlib backend and put together the full gallery pages."""
    working_dir = Path.cwd()
    os.chdir(app.builder.srcdir)
    gallery_dir = Path(app.builder.srcdir).resolve() / "gallery"
    images_dir = gallery_dir / "_images"
    scripts_dir = gallery_dir / "_scripts"
    site_url = "https://arviz-plots.readthedocs.io/en/latest/"

    if not images_dir.is_dir():
        os.makedirs(images_dir)

    if not scripts_dir.is_dir():
        os.makedirs(scripts_dir)

    index_page = ["(example_gallery)=\n# Example gallery"]
    backreferences = defaultdict(list)
    api_regex = re.compile(r"azp\.(plot_[a-z]+)\(")

    for folder, title in dir_title_map.items():
        category_dir = gallery_dir / folder
        files = [filename.stem for filename in sorted(category_dir.glob("*.py"))]
        index_page.append(toctree_template.format(title=title, files="\n".join(files)))
        index_page.append(":::::{grid} 1 2 3 3\n:gutter: 2 2 3 3\n")
        for basename in files:
            logger.info(f"Processing gallery example {basename}")
            # first step: run scripts with matplotlib backend and save png files
            with open(category_dir / f"{basename}.py", "r", encoding="utf-8") as fp:
                text = fp.read()
            _, doc_text, code_text = text.split('"""')
            code_text = code_text.strip("\n")

            backend_line_emphasis = ""
            emph_lines = []
            api_funs = []
            for i, line in enumerate(code_text.splitlines()):
                if 'backend="none"' in line:
                    emph_lines.append(str(i + 1))
                match = api_regex.search(line)
                if match is not None:
                    api_funs.append(match.groups()[0])

            if emph_lines:
                backend_line_emphasis = f":emphasize-lines: {','.join(emph_lines)}"

            with open(scripts_dir / f"{basename}.py", "w", encoding="utf-8") as fc:
                fc.write(code_text)

            head_text, foot_text = doc_text.split("---")

            head_lines = head_text.splitlines()
            for i, line in enumerate(head_lines):
                if line.startswith("# "):
                    break
            else:
                raise ValueError(f"No title found for {basename} example")
            example_title = head_lines[i]
            example_description = "\n".join(head_lines[i + 1 :])
            entry = {
                "basename": basename,
                "refname": basename.replace("plot_", "gallery_"),
                "title": example_title.strip("# "),
                "description": example_description.strip(" \n").replace("\n", " "),
            }
            for fun in api_funs:
                backreferences[fun].append(entry)

            index_page.append(grid_item_template.format(**entry))

            mpl_noshow_code = code_text.replace('backend="none"', 'backend="matplotlib"').replace(
                "pc.show()", ""
            )
            exec(compile(mpl_noshow_code, basename, "exec"))  # pylint: disable=exec-used
            fig = plt.gcf()
            fig.canvas.draw()
            fig.savefig(images_dir / f"{basename}.png", dpi=75)
            plt.close("all")

            minigalleries = "\n".join(minigallery_in_example.format(fun=fun) for fun in api_funs)

            myst_text = f"""
            ({basename.replace("plot_", "gallery_")})=
            {head_text}

            ::::::{{tab-set}}
            :class: full-width
            :sync-group: backend

            :::::{{tab-item}} Matplotlib
            :sync: matplotlib

            ![Matplotlib version of {basename}](_images/{basename}.png)

            :::::

            :::::{{tab-item}} Bokeh
            :sync: bokeh

            ```{{bokeh-plot}}
            :source-position: none

            from bokeh.plotting import show

            {code_text.replace('backend="none"', 'backend="bokeh"').replace("pc.show()", "")}

            # for some reason the bokeh plot extension needs explicit use of show
            show(pc.viz["chart"].item() if pc.viz["chart"].item() is not None else pc.viz["plot"].item())
            ```

            Link to this page with the [bokeh tab selected]({site_url}/gallery/{basename}.html?backend=bokeh#synchronised-tabs)
            :::::

            :::::{{tab-item}} Plotly
            :sync: plotly

            ```{{jupyter-execute}}
            :hide-code:

            {code_text.replace('backend="none"', 'backend="plotly"')}
            ```

            Link to this page with the [plotly tab selected]({site_url}/gallery/{basename}.html?backend=plotly#synchronised-tabs)
            :::::
            ::::::

            ```{{literalinclude}} _scripts/{basename}.py
            {backend_line_emphasis}
            ```

            {foot_text}

            {minigalleries}

            :::{{div}} example-plot-download
            {{download}}`Download Python Source Code: {basename}.py<_scripts/{basename}.py>`
            :::
            """
            myst_text = "\n".join((line.strip(" ") for line in myst_text.strip("\n").splitlines()))

            with open(gallery_dir / f"{basename}.md", "w", encoding="utf-8") as fm:
                fm.write(myst_text)

        index_page.append("\n:::::\n")

    with open(gallery_dir / "backreferences.json", "w", encoding="utf-8") as f:
        json.dump(backreferences, f)

    with open(gallery_dir / "index.md", "w", encoding="utf-8") as fi:
        fi.write("\n".join(index_page))

    os.chdir(working_dir)


class MiniGallery(Directive):
    """Custom directive to insert a mini-gallery.

    The required argument is one or more of the following:

    * fully qualified names of objects
    * pathlike strings to example Python files
    * glob-style pathlike strings to example Python files

    The string list of arguments is separated by spaces.

    The mini-gallery will be the subset of gallery
    examples that make use of that object from that specific namespace

    Options:

    * `add-heading` adds a heading to the mini-gallery.  If an argument is
      provided, it uses that text for the heading.  Otherwise, it uses
      default text.
    * `heading-level` specifies the heading level of the heading as a single
      character.  If omitted, the default heading level is `'^'`.
    """

    required_arguments = 1
    has_content = False
    optional_arguments = 0
    final_argument_whitespace = True

    def run(self):
        """Generate mini-gallery from backreference and example files."""
        gallery_dir = Path(self.state.document.settings.env.srcdir).resolve() / "gallery"
        docname = self.state.document.settings.env.docname
        with open(gallery_dir / "backreferences.json", "r", encoding="utf-8") as f:
            backreferences = json.load(f)

        # Parse the argument into the individual object
        target_obj = self.arguments[0].strip()

        lines = []

        entry_elements = [
            entry
            for entry in backreferences[target_obj]
            if f"gallery/{entry['basename']}" != docname
        ]

        if len(entry_elements) >= 1:
            lines.append(".. grid:: 1 2 3 3\n   :gutter: 2 2 3 3\n\n")

        for entry in entry_elements:
            if f"gallery/{entry['basename']}" == docname:
                continue
            lines.extend(
                [
                    f"   {line}"
                    for line in minigallery_item_template_rst.format(**entry).splitlines()
                ]
            )

        text = "\n".join(lines)
        include_lines = statemachine.string2lines(text, convert_whitespace=True)
        self.state_machine.insert_input(include_lines, self.state_machine.get_source_and_line()[0])

        return []


def setup(app):
    """Connect the extension to sphinx so it is executed when the builder is initialized."""
    app.add_directive("minigallery", MiniGallery)
    app.connect("builder-inited", main)
