=============================================
Managers for facetting and aesthetics mapping
=============================================
The classes in this module lay at the core of the library,
and are consequently available at the ``arviz_plots`` top level namespace.

They abstract all information regarding :term:`facetting` and :term:`aesthetic mapping`
in our :term:`chart` to prevent duplication and ensure coherence between
the different functions.

.. currentmodule:: arviz_plots

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
   PlotCollection.show

Facetting and aesthetics mapping
................................

.. autosummary::
   :toctree: generated/

   PlotCollection.generate_aes_dt
   PlotCollection.get_aes_kwargs
   PlotCollection.update_aes

Other
.....

.. autosummary::
   :toctree: generated/

   PlotCollection.allocate_artist
   PlotCollection.get_viz
   PlotCollection.get_target
