"""Generate images and full gallery pages from python scripts."""
import os
from pathlib import Path

import matplotlib.pyplot as plt
from sphinx.util import logging

logger = logging.getLogger(__name__)


def main(app):
    """Generate images with matplotlib backend and put together the full gallery pages."""
    working_dir = Path.cwd()
    os.chdir(app.builder.srcdir)
    gallery_dir = Path(app.builder.srcdir).resolve() / "gallery"
    script_dir = gallery_dir / "scripts"
    images_dir = gallery_dir / "_images"

    if not images_dir.is_dir():
        os.makedirs(images_dir)

    files = sorted(script_dir.glob("*.py"))
    for filename in files:
        basename = filename.stem
        logger.info(f"Processing gallery example {basename}")
        # first step: run scripts with matplotlib backend and save png files
        with open(filename, "r", encoding="utf-8") as fp:
            code_text = fp.read()
        mpl_noshow_code = code_text.replace('backend="none"', 'backend="matplotlib"').replace(
            "pc.show()", ""
        )
        exec(compile(mpl_noshow_code, basename, "exec"))  # pylint: disable=exec-used
        fig = plt.gcf()
        fig.canvas.draw()
        fig.savefig(images_dir / f"{basename}.png", dpi=75)
        plt.close("all")

        # generate the md/rst files corresponding to the tabbed content
        with open(gallery_dir / f"{basename}.part.md", "r", encoding="utf-8") as fm:
            page_start = fm.read()

        backend_tabs = f"""
        ::::::{{tab-set}}
        :class: full-width

        :::::{{tab-item}} Matplotlib
        ![Matplotlib version of {basename}](_images/{basename}.png)
        :::::

        :::::{{tab-item}} Bokeh
        ```{{bokeh-plot}}
        :source-position: none

        from bokeh.plotting import show

        {code_text.replace('backend="none"', 'backend="bokeh"').replace("pc.show()", "")}

        # for some reason the bokeh plot extension needs explicit use of show
        show(pc.viz["chart"].item())
        ```
        :::::

        :::::{{tab-item}} Plotly
        ```{{jupyter-execute}}
        :hide-code:

        {code_text.replace('backend="none"', 'backend="plotly"')}
        ```
        :::::
        ::::::

        ```{{literalinclude}} scripts/{basename}.py
        ```
        """
        backend_tabs = "\n".join((line.strip(" ") for line in backend_tabs.splitlines()))

        with open(gallery_dir / f"{basename}.md", "w", encoding="utf-8") as fm:
            fm.write(page_start)
            fm.write("\n")
            fm.write(backend_tabs)

    os.chdir(working_dir)


def setup(app):
    """Connect the extension to sphinx so it is executed when the builder is initialized."""
    app.connect("builder-inited", main)
