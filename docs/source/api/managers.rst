=============================================
Managers for facetting and aesthetics mapping
=============================================
The classes in this module lay at the core of the library,
and are consequently available at the ``arviz_plots`` top level namespace.

They abstract all information regarding :term:`facetting` and :term:`aesthetic mapping`
in our :term:`chart` to prevent duplication and ensure coherence between
the different functions.

.. currentmodule:: arviz_plots

.. autosummary::
   :toctree: generated/

   PlotCollection
   PlotCollection.grid
   PlotCollection.map
   PlotCollection.show
   PlotCollection.wrap
