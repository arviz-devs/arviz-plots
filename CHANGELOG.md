<a id="v0.5.0"></a>
# [v0.5.0](https://github.com/arviz-devs/arviz-plots/releases/tag/v0.5.0) - 2025-03-21

## What's Changed
* Randomized pit by [@aloctavodia](https://github.com/aloctavodia) in [#167](https://github.com/arviz-devs/arviz-plots/pull/167)
* Add missing visuals to docs by [@aloctavodia](https://github.com/aloctavodia) in [#168](https://github.com/arviz-devs/arviz-plots/pull/168)
* Make ecdf_line a step line by [@aloctavodia](https://github.com/aloctavodia) in [#169](https://github.com/arviz-devs/arviz-plots/pull/169)
* Properly wrap columns and reduce duplicated code by [@aloctavodia](https://github.com/aloctavodia) in [#170](https://github.com/arviz-devs/arviz-plots/pull/170)
* Added plot_bf() function for bayes_factor in arviz-plots by [@PiyushPanwarFST](https://github.com/PiyushPanwarFST) in [#158](https://github.com/arviz-devs/arviz-plots/pull/158)
* Update utils.py by [@aloctavodia](https://github.com/aloctavodia) in [#171](https://github.com/arviz-devs/arviz-plots/pull/171)
* Add plot_rank by [@aloctavodia](https://github.com/aloctavodia) in [#172](https://github.com/arviz-devs/arviz-plots/pull/172)
* Add plot ecdf pit plot by [@aloctavodia](https://github.com/aloctavodia) in [#173](https://github.com/arviz-devs/arviz-plots/pull/173)
* fix regression bug plot_psense_quantities by [@aloctavodia](https://github.com/aloctavodia) in [#175](https://github.com/arviz-devs/arviz-plots/pull/175)
* Improve Data Generation & Plot Tests: by [@PiyushPanwarFST](https://github.com/PiyushPanwarFST) in [#177](https://github.com/arviz-devs/arviz-plots/pull/177)
* Adds square root scale for bokeh and plotly by [@The-Broken-Keyboard](https://github.com/The-Broken-Keyboard) in [#178](https://github.com/arviz-devs/arviz-plots/pull/178)
* Update plot_compare by [@aloctavodia](https://github.com/aloctavodia) in [#181](https://github.com/arviz-devs/arviz-plots/pull/181)
* plot_converge_dist: Add grouped argument by [@aloctavodia](https://github.com/aloctavodia) in [#182](https://github.com/arviz-devs/arviz-plots/pull/182)
* adds sqrt scale for yaxis in plotly by [@The-Broken-Keyboard](https://github.com/The-Broken-Keyboard) in [#183](https://github.com/arviz-devs/arviz-plots/pull/183)
* Add coverage argument to pit plots by [@aloctavodia](https://github.com/aloctavodia) in [#185](https://github.com/arviz-devs/arviz-plots/pull/185)


## New Contributors
* [@github-actions](https://github.com/github-actions) made their first contribution in [#165](https://github.com/arviz-devs/arviz-plots/pull/165)
* [@PiyushPanwarFST](https://github.com/PiyushPanwarFST) made their first contribution in [#158](https://github.com/arviz-devs/arviz-plots/pull/158)

**Full Changelog**: https://github.com/arviz-devs/arviz-plots/compare/v0.4.0...v0.5.0

[Changes][v0.5.0]


<a id="v0.4.0"></a>
# [v0.4.0](https://github.com/arviz-devs/arviz-plots/releases/tag/v0.4.0) - 2025-03-05

## What's Changed
* move out new_ds to arviz-stats by [@aloctavodia](https://github.com/aloctavodia) in [#102](https://github.com/arviz-devs/arviz-plots/pull/102)
* update version, dependencies and CI by [@OriolAbril](https://github.com/OriolAbril) in [#110](https://github.com/arviz-devs/arviz-plots/pull/110)
* Use DataTree class from xarray by [@OriolAbril](https://github.com/OriolAbril) in [#111](https://github.com/arviz-devs/arviz-plots/pull/111)
* Update pyproject.toml by [@OriolAbril](https://github.com/OriolAbril) in [#113](https://github.com/arviz-devs/arviz-plots/pull/113)
* Add energy plot by [@aloctavodia](https://github.com/aloctavodia) in [#108](https://github.com/arviz-devs/arviz-plots/pull/108)
* Add plot for distribution of convergence diagnostics by [@aloctavodia](https://github.com/aloctavodia) in [#105](https://github.com/arviz-devs/arviz-plots/pull/105)
* Add separated prior and likelihood groups by [@aloctavodia](https://github.com/aloctavodia) in [#117](https://github.com/arviz-devs/arviz-plots/pull/117)
* Add psense_quantities plot by [@aloctavodia](https://github.com/aloctavodia) in [#119](https://github.com/arviz-devs/arviz-plots/pull/119)
* Rename arviz-clean to arviz-variat by [@aloctavodia](https://github.com/aloctavodia) in [#120](https://github.com/arviz-devs/arviz-plots/pull/120)
* add cetrino and vibrant styles for plotly by [@aloctavodia](https://github.com/aloctavodia) in [#121](https://github.com/arviz-devs/arviz-plots/pull/121)
* psense: fix facetting and add xlabel by [@aloctavodia](https://github.com/aloctavodia) in [#123](https://github.com/arviz-devs/arviz-plots/pull/123)
* Add summary dictionary arguments by [@aloctavodia](https://github.com/aloctavodia) in [#125](https://github.com/arviz-devs/arviz-plots/pull/125)
* plotly: change format of title update in backend by [@The-Broken-Keyboard](https://github.com/The-Broken-Keyboard) in [#124](https://github.com/arviz-devs/arviz-plots/pull/124)
* Update glossary.md by [@aloctavodia](https://github.com/aloctavodia) in [#126](https://github.com/arviz-devs/arviz-plots/pull/126)
* Add PAV-adjusted calibration plot by [@aloctavodia](https://github.com/aloctavodia) in [#127](https://github.com/arviz-devs/arviz-plots/pull/127)
* Upper bound plotly by [@aloctavodia](https://github.com/aloctavodia) in [#128](https://github.com/arviz-devs/arviz-plots/pull/128)
* Use isotonic function that works with datatrees by [@aloctavodia](https://github.com/aloctavodia) in [#131](https://github.com/arviz-devs/arviz-plots/pull/131)
* plot_pava_calibrarion: Add reference, fix xlabel by [@aloctavodia](https://github.com/aloctavodia) in [#132](https://github.com/arviz-devs/arviz-plots/pull/132)
* Fix bug when setting some plot_kwargs to false by [@aloctavodia](https://github.com/aloctavodia) in [#134](https://github.com/arviz-devs/arviz-plots/pull/134)
* Add citations by [@aloctavodia](https://github.com/aloctavodia) in [#135](https://github.com/arviz-devs/arviz-plots/pull/135)
* use <6 version of plotly for documentation and use latest for other purposes by [@The-Broken-Keyboard](https://github.com/The-Broken-Keyboard) in [#136](https://github.com/arviz-devs/arviz-plots/pull/136)
* fix see also pava gallery by [@aloctavodia](https://github.com/aloctavodia) in [#137](https://github.com/arviz-devs/arviz-plots/pull/137)
* Add plot_ppc_dist by [@aloctavodia](https://github.com/aloctavodia) in [#138](https://github.com/arviz-devs/arviz-plots/pull/138)
* Add warning message for discrete data by [@aloctavodia](https://github.com/aloctavodia) in [#139](https://github.com/arviz-devs/arviz-plots/pull/139)
* rename plot_pava and minor fixes by [@aloctavodia](https://github.com/aloctavodia) in [#140](https://github.com/arviz-devs/arviz-plots/pull/140)
* Plotly: Fix excesive margins by [@aloctavodia](https://github.com/aloctavodia) in [#141](https://github.com/arviz-devs/arviz-plots/pull/141)
* add arviz-style for bokeh by [@aloctavodia](https://github.com/aloctavodia) in [#122](https://github.com/arviz-devs/arviz-plots/pull/122)
* Add Plot ppc rootogram by [@aloctavodia](https://github.com/aloctavodia) in [#142](https://github.com/arviz-devs/arviz-plots/pull/142)
* plot_ppc_rootogram: fix examples by [@aloctavodia](https://github.com/aloctavodia) in [#144](https://github.com/arviz-devs/arviz-plots/pull/144)
* Reorganize categories in the gallery by [@aloctavodia](https://github.com/aloctavodia) in [#145](https://github.com/arviz-devs/arviz-plots/pull/145)
* remove plots from titles by [@aloctavodia](https://github.com/aloctavodia) in [#146](https://github.com/arviz-devs/arviz-plots/pull/146)
* Consistence use of data_pairs, remove default markers from pava by [@aloctavodia](https://github.com/aloctavodia) in [#152](https://github.com/arviz-devs/arviz-plots/pull/152)
* added functionality of step histogram for all three backends by [@The-Broken-Keyboard](https://github.com/The-Broken-Keyboard) in [#147](https://github.com/arviz-devs/arviz-plots/pull/147)
* Use continuous outcome for plot_ppc_dist example by [@aloctavodia](https://github.com/aloctavodia) in [#154](https://github.com/arviz-devs/arviz-plots/pull/154)
* add grid visual by [@aloctavodia](https://github.com/aloctavodia) in [#155](https://github.com/arviz-devs/arviz-plots/pull/155)
* Add plot_ppc_pit by [@aloctavodia](https://github.com/aloctavodia) in [#159](https://github.com/arviz-devs/arviz-plots/pull/159)
* plot_ppc_pava: Change default xlabel by [@aloctavodia](https://github.com/aloctavodia) in [#161](https://github.com/arviz-devs/arviz-plots/pull/161)

## New Contributors
* [@The-Broken-Keyboard](https://github.com/The-Broken-Keyboard) made their first contribution in [#124](https://github.com/arviz-devs/arviz-plots/pull/124)

**Full Changelog**: https://github.com/arviz-devs/arviz-plots/compare/v0.3.0...v0.4.0

[Changes][v0.4.0]


[v0.5.0]: https://github.com/arviz-devs/arviz-plots/compare/v0.4.0...v0.5.0
[v0.4.0]: https://github.com/arviz-devs/arviz-plots/tree/v0.4.0

<!-- Generated by https://github.com/rhysd/changelog-from-release v3.9.0 -->
