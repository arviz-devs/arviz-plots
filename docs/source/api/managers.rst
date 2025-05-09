=============================================
Managers for faceting and aesthetics mapping
=============================================
The classes in this module lay at the core of the library,
and are consequently available at the ``arviz_plots`` top level namespace.

They abstract all information regarding :term:`faceting` and :term:`aesthetic mapping`
in our :term:`chart` to prevent duplication and ensure coherence between
the different functions.

.. currentmodule:: arviz_plots

PlotCollection
==============

Object creation
...............

.. autosummary::
   :toctree: generated/

   PlotCollection
   PlotCollection.grid
   PlotCollection.wrap

Plotting
........

.. autosummary::
   :toctree: generated/

   PlotCollection.add_legend
   PlotCollection.map
   PlotCollection.plot_iterator

Attributes
..........

.. autosummary::
   :toctree: generated/

   PlotCollection.aes
   PlotCollection.viz
   PlotCollection.aes_set
   PlotCollection.facet_dims
   PlotCollection.data

Faceting and aesthetics mapping
................................

.. autosummary::
   :toctree: generated/

   PlotCollection.generate_aes_dt
   PlotCollection.get_aes_as_dataset
   PlotCollection.get_aes_kwargs
   PlotCollection.update_aes
   PlotCollection.update_aes_from_dataset

Other
.....

.. autosummary::
   :toctree: generated/

   PlotCollection.allocate_artist
   PlotCollection.get_viz
   PlotCollection.get_target
   PlotCollection.show
   PlotCollection.savefig

PlotMatrix
==========

Object creation
...............

.. autosummary::
   :toctree: generated/

   PlotMatrix

Plotting
........

.. autosummary::
   :toctree: generated/

   PlotMatrix.map
   PlotMatrix.map_triangle
   PlotMatrix.map_lower
   PlotMatrix.map_upper

Attributes
..........

.. autosummary::
   :toctree: generated/

   PlotMatrix.aes
   PlotMatrix.viz
   PlotMatrix.aes_set
   PlotMatrix.base_loop_dims
   PlotMatrix.data
