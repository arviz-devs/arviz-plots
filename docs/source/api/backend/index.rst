.. _api_backends:

==============================
Interface to plotting backends
==============================

------------------
Available backends
------------------

.. grid:: 1 1 2 2
    :gutter: 2

    .. grid-item-card::
        :img-alt: Matplotlib
        :img-background: ../../_static/matplotlib-logo-light.svg
        :class-img-bottom: dark-light

    .. grid-item-card::
        :img-alt: Bokeh
        :img-background: ../../_static/bokeh-logo-light.svg
        :class-img-bottom: dark-light

    .. grid-item-card::
        :img-alt: Plotly
        :img-background: ../../_static/plotly-logo-light.png
        :class-img-bottom: dark-light

    .. grid-item-card::
        :img-alt: None (only processing, no plotting)
        :img-background: ../../_static/none-logo-light.png
        :class-img-bottom: dark-light


--------------------------
Common interface arguments
--------------------------

.. automodule:: arviz_plots.backend

.. dropdown:: Keyword arguments
    :name: backend_interface_arguments
    :open:

    The argument names are defined here to have a comprehensive list of all possibilities.
    If relevant, a keyword argument present here should be present in the function,
    and converted in each backend to its corresponding argument in that backend.

    This set of arguments doesn't aim to be complete, only to cover basic properties
    so the plotting functions can work on multiple backends without duplication.
    Advanced customization will be backend specific through ``**kwargs``.

    target
        This module is designed mainly in a functional way. Thus, all functions
        should take a ``target`` argument which indicates on which object should
        the function be applied to.

    color
        Color of the visual element. Should also be present whenever ``facecolor``
        and ``edgecolor`` are present, setting the default value for both.

    facecolor
        Color for filling the visual element.

    edgecolor
        Color for the edges of the visual element.

    alpha
        Transparency of the visual element.

    width
        Width of the visual element itself or of its edges, whichever applies.

    size
        Size of the visual element.

    linestyle
        Style of the line plotted.

    marker
        Marker to be added to the plot. Needs to accept `|` as a valid value.

    rotation
        Rotation of axis labels in degrees.

    vertical_align
        Vertical alignment between the visual element and the data coordinates provided.

    horizontal_align
        Horizontal alignment between the visual element and the data coordinates provided.

    axis
        Data axis (x, y or both) on which to apply the function.

--------------------------
Common interface functions
--------------------------

Object creation and I/O
.......................

.. autosummary::
   :toctree: generated/

   create_plotting_grid
   savefig
   show

Geoms
.....

.. autosummary::
   :toctree: generated/

   ciliney
   fill_between_y
   hist
   hline
   hspan
   line
   multiple_lines
   scatter
   step
   text
   vline
   vspan

Plot appearance
................

.. autosummary::
   :toctree: generated/

   grid
   remove_ticks
   remove_axis
   set_ticklabel_visibility
   set_y_scale
   ticklabel_props
   title
   xlabel
   xticks
   ylabel
   yticks

Legend
......

.. autosummary::
   :toctree: generated/

   legend

Helper functions
................

.. autosummary::
   :toctree: generated/

   get_background_color
   get_default_aes
   scale_fig_size
