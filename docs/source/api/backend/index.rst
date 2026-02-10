.. _api_backends:

==============================
Interface to plotting backends
==============================

------------------
Available backends
------------------

.. grid:: 1 1 2 2

    .. grid-item-card::
        :link: matplotlib
        :link-type: doc
        :link-alt: Matplotlib
        :img-background: ../../_static/matplotlib-logo-light.svg
        :class-img-bottom: dark-light

    .. grid-item-card::
        :link: bokeh
        :link-type: doc
        :link-alt: Bokeh
        :img-background: ../../_static/bokeh-logo-light.svg
        :class-img-bottom: dark-light

    .. grid-item-card::
        :link: plotly
        :link-type: doc
        :link-alt: Plotly
        :img-background: ../../_static/plotly-logo-light.png
        :class-img-bottom: dark-light

    .. grid-item-card::
        :link: none
        :link-type: doc
        :link-alt: None (no plotting, only data processing)
        :img-background: ../../_static/none-logo-light.png
        :class-img-bottom: dark-light


.. toctree::
   :maxdepth: 1
   :hidden:

   Matplotlib <matplotlib>
   Bokeh <bokeh>
   Plotly <plotly>
   None (only processing, no plotting) <none>

---------------------------
Common interface definition
---------------------------

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
