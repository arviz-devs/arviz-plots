<a id="v0.7.0"></a>
# [v0.7.0](https://github.com/arviz-devs/arviz-plots/releases/tag/v0.7.0) - 2025-11-11

## What's Changed

### New features
* Adds Pair plot by [@The-Broken-Keyboard](https://github.com/The-Broken-Keyboard) in [#287](https://github.com/arviz-devs/arviz-plots/pull/287)  
* Add parallel plot by [@The-Broken-Keyboard](https://github.com/The-Broken-Keyboard) in [#300](https://github.com/arviz-devs/arviz-plots/pull/300)  
* Add face in dist_plot by [@The-Broken-Keyboard](https://github.com/The-Broken-Keyboard) in [#305](https://github.com/arviz-devs/arviz-plots/pull/305)  
* Add dark theme by [@The-Broken-Keyboard](https://github.com/The-Broken-Keyboard) in [#309](https://github.com/arviz-devs/arviz-plots/pull/309)  
* Add muted style/palette ("arviz-tenui") by [@aloctavodia](https://github.com/aloctavodia) in [#315](https://github.com/arviz-devs/arviz-plots/pull/315)  
* Add legend to visuals by [@aloctavodia](https://github.com/aloctavodia) in [#321](https://github.com/arviz-devs/arviz-plots/pull/321)  
* plot_ppc_pava: extend support for categorical and ordinal data by [@aloctavodia](https://github.com/aloctavodia) in [#316](https://github.com/arviz-devs/arviz-plots/pull/316)  
* Add mode as a valid option for `point_estimate` in plot_forest by [@aloctavodia](https://github.com/aloctavodia) in [#337](https://github.com/arviz-devs/arviz-plots/pull/337)  
* Add plot_lm by [@The-Broken-Keyboard](https://github.com/The-Broken-Keyboard) in [#328](https://github.com/arviz-devs/arviz-plots/pull/328)  
* Add plot_ppc_intervals function by [@PiyushPanwarFST](https://github.com/PiyushPanwarFST) in [#334](https://github.com/arviz-devs/arviz-plots/pull/334)  
* Add plot_ppc_censored by [@aloctavodia](https://github.com/aloctavodia) in [#346](https://github.com/arviz-devs/arviz-plots/pull/346)  
* Allow using "CX" notation for aliases to default aes cycle by [@OriolAbril](https://github.com/OriolAbril) in [#311](https://github.com/arviz-devs/arviz-plots/pull/311)  
* Allow values in visuals to be True by [@aloctavodia](https://github.com/aloctavodia) in [#326](https://github.com/arviz-devs/arviz-plots/pull/326)  
* Allow non-shared variables for multiple models by [@aloctavodia](https://github.com/aloctavodia) in [#331](https://github.com/arviz-devs/arviz-plots/pull/331)  
* Add warning message when group is prior_predictive by [@aloctavodia](https://github.com/aloctavodia) in [#352](https://github.com/arviz-devs/arviz-plots/pull/352)  

### Maintenance and bug fixes

* Update API plot list by [@aloctavodia](https://github.com/aloctavodia) in [#293](https://github.com/arviz-devs/arviz-plots/pull/293)  
* Update given upcoming changes in arviz-stats by [@aloctavodia](https://github.com/aloctavodia) in [#294](https://github.com/arviz-devs/arviz-plots/pull/294)  
* PR for plot_ridge missing from example gallery by [@aminskey](https://github.com/aminskey) in [#303](https://github.com/arviz-devs/arviz-plots/pull/303)  
* fix position point_estimate_text by [@aloctavodia](https://github.com/aloctavodia) in [#313](https://github.com/arviz-devs/arviz-plots/pull/313)  
* Restrict palette to eight colors by [@aloctavodia](https://github.com/aloctavodia) in [#312](https://github.com/arviz-devs/arviz-plots/pull/312)  
* Set custom tooltip for plotly and other fixes by [@aloctavodia](https://github.com/aloctavodia) in [#323](https://github.com/arviz-devs/arviz-plots/pull/323)  
* Improve default legend for bokeh by [@aloctavodia](https://github.com/aloctavodia) in [#324](https://github.com/arviz-devs/arviz-plots/pull/324)  
* plot_compare: improve docstring and fix plotly overlap by [@aloctavodia](https://github.com/aloctavodia) in [#325](https://github.com/arviz-devs/arviz-plots/pull/325)  
* Generalize plot_compare by [@aloctavodia](https://github.com/aloctavodia) in [#329](https://github.com/arviz-devs/arviz-plots/pull/329)  
* Adapt to changes in ecdf computation by [@aloctavodia](https://github.com/aloctavodia) in [#335](https://github.com/arviz-devs/arviz-plots/pull/335)  
* Add missing check for non-labeller by [@aloctavodia](https://github.com/aloctavodia) in [#341](https://github.com/arviz-devs/arviz-plots/pull/341)  
* Update default ci_prob for ecdf-pit plots by [@aloctavodia](https://github.com/aloctavodia) in [#345](https://github.com/arviz-devs/arviz-plots/pull/345)  
* Add Python 3.13 to the tests by [@aloctavodia](https://github.com/aloctavodia) in [#348](https://github.com/arviz-devs/arviz-plots/pull/348)  
* Small tweaks to tumma by [@aloctavodia](https://github.com/aloctavodia) in [#320](https://github.com/arviz-devs/arviz-plots/pull/320)  
* improve default figsize for plot_ridge by [@OriolAbril](https://github.com/OriolAbril) in [#360](https://github.com/arviz-devs/arviz-plots/pull/360)  
* Change truncation_factor name to avoid possible confusion by [@aloctavodia](https://github.com/aloctavodia) in [#363](https://github.com/arviz-devs/arviz-plots/pull/363)  
* Refactor plot_lm by [@aloctavodia](https://github.com/aloctavodia) in [#343](https://github.com/arviz-devs/arviz-plots/pull/343)  
* Remove deprecated GitHub action by [@OriolAbril](https://github.com/OriolAbril) in [#362](https://github.com/arviz-devs/arviz-plots/pull/362)  
* Bump actions/download-artifact from 4 to 5 by [@dependabot](https://github.com/dependabot)[bot] in [#319](https://github.com/arviz-devs/arviz-plots/pull/319)  
* Bump actions/checkout from 4 to 5 by [@dependabot](https://github.com/dependabot)[bot] in [#327](https://github.com/arviz-devs/arviz-plots/pull/327)  
* Bump actions/setup-python from 5 to 6 by [@dependabot](https://github.com/dependabot)[bot] in [#339](https://github.com/arviz-devs/arviz-plots/pull/339)  
* Bump peter-evans/create-or-update-comment from 4 to 5 by [@dependabot](https://github.com/dependabot)[bot] in [#351](https://github.com/arviz-devs/arviz-plots/pull/351)  
* Bump actions/download-artifact from 5 to 6 by [@dependabot](https://github.com/dependabot)[bot] in [#359](https://github.com/arviz-devs/arviz-plots/pull/359)  
* Bump release 0.7.0 by [@aloctavodia](https://github.com/aloctavodia) in [#366](https://github.com/arviz-devs/arviz-plots/pull/366)  

### Documentation

* Add ref target to installation section by [@OriolAbril](https://github.com/OriolAbril) in [#295](https://github.com/arviz-devs/arviz-plots/pull/295)  
* Clean docstrings by [@aloctavodia](https://github.com/aloctavodia) in [#336](https://github.com/arviz-devs/arviz-plots/pull/336)  
* Correct a typo: exising -> existing by [@star1327p](https://github.com/star1327p) in [#342](https://github.com/arviz-devs/arviz-plots/pull/342)  
* Correct a typo: appeareance -> appearance by [@star1327p](https://github.com/star1327p) in [#349](https://github.com/arviz-devs/arviz-plots/pull/349)  
* Correct more typos in arviz-plots documentation by [@star1327p](https://github.com/star1327p) in [#350](https://github.com/arviz-devs/arviz-plots/pull/350)  
* Add link to `tests/conftest.py` file by [@star1327p](https://github.com/star1327p) in [#355](https://github.com/arviz-devs/arviz-plots/pull/355)  
* Add plot_ppc_interval to API docs by [@satwiksps](https://github.com/satwiksps) in [#365](https://github.com/arviz-devs/arviz-plots/pull/365)  
* Improve usage of quotation marks in documentation by [@star1327p](https://github.com/star1327p) in [#361](https://github.com/arviz-devs/arviz-plots/pull/361)  


## New Contributors
* [@aminskey](https://github.com/aminskey) made their first contribution in [#303](https://github.com/arviz-devs/arviz-plots/pull/303)
* [@satwiksps](https://github.com/satwiksps) made their first contribution in [#365](https://github.com/arviz-devs/arviz-plots/pull/365)

**Full Changelog**: https://github.com/arviz-devs/arviz-plots/compare/v0.6.0...v0.7.0

[Changes][v0.7.0]


<a id="v0.6.0"></a>
# [v0.6.0](https://github.com/arviz-devs/arviz-plots/releases/tag/v0.6.0) - 2025-06-18

## What's Changed

## New features
* Add plot_loo_pit by [@aloctavodia](https://github.com/aloctavodia) in [#190](https://github.com/arviz-devs/arviz-plots/pull/190)
* Added rug to lot_dist() by [@PiyushPanwarFST](https://github.com/PiyushPanwarFST) in [#192](https://github.com/arviz-devs/arviz-plots/pull/192)
* Add plot_prior_posterior and rewrite plot_bf by [@aloctavodia](https://github.com/aloctavodia) in [#198](https://github.com/arviz-devs/arviz-plots/pull/198)
* Add plot_autocorr by [@suhaani-agarwal](https://github.com/suhaani-agarwal) in [#153](https://github.com/arviz-devs/arviz-plots/pull/153)
* Add plot_mcse by [@aloctavodia](https://github.com/aloctavodia) in [#211](https://github.com/arviz-devs/arviz-plots/pull/211)
* Add combine_plots, allowing to arrange a set of existing plots into a single chart by [@aloctavodia](https://github.com/aloctavodia) in [#207](https://github.com/arviz-devs/arviz-plots/pull/207)
* Add plot_ppc_tstat by [@rohanbabbar04](https://github.com/rohanbabbar04) in [#210](https://github.com/arviz-devs/arviz-plots/pull/210)
* Add more tstats to plot_ppc_tstat by [@aloctavodia](https://github.com/aloctavodia) in [#217](https://github.com/arviz-devs/arviz-plots/pull/217)
* Add references to plot_ppc_dist by [@rohanbabbar04](https://github.com/rohanbabbar04) in [#223](https://github.com/arviz-devs/arviz-plots/pull/223)
* Add PlotMatrix plotting manager class by [@OriolAbril](https://github.com/OriolAbril) in [#221](https://github.com/arviz-devs/arviz-plots/pull/221)
* Add savefig method to PlotCollection by [@aloctavodia](https://github.com/aloctavodia) in [#246](https://github.com/arviz-devs/arviz-plots/pull/246)
* Add plot_rank_dist by [@aloctavodia](https://github.com/aloctavodia) in [#253](https://github.com/arviz-devs/arviz-plots/pull/253)
* Add support multiple variables in plot_bf by [@aloctavodia](https://github.com/aloctavodia) in [#256](https://github.com/arviz-devs/arviz-plots/pull/256)
* Add function for plotting reference bands by [@rohanbabbar04](https://github.com/rohanbabbar04) in [#258](https://github.com/arviz-devs/arviz-plots/pull/258)
* Add plot_pair_focus by [@The-Broken-Keyboard](https://github.com/The-Broken-Keyboard) in [#276](https://github.com/arviz-devs/arviz-plots/pull/276)

## Maintenance and bug fixes
* Clean pit functions by [@aloctavodia](https://github.com/aloctavodia) in [#191](https://github.com/arviz-devs/arviz-plots/pull/191)
* Adds rule to select fewer ticks in sqrt scale for bokeh and plotly by [@The-Broken-Keyboard](https://github.com/The-Broken-Keyboard) in [#189](https://github.com/arviz-devs/arviz-plots/pull/189)
* Show bf for null by [@aloctavodia](https://github.com/aloctavodia) in [#196](https://github.com/arviz-devs/arviz-plots/pull/196)
* Make rug defaults to False by [@aloctavodia](https://github.com/aloctavodia) in [#197](https://github.com/arviz-devs/arviz-plots/pull/197)
* Rootogram: do not plot observation by default for prior_predictive by [@aloctavodia](https://github.com/aloctavodia) in [#202](https://github.com/arviz-devs/arviz-plots/pull/202)
* Add group argument to plot_ppc_pava by [@aloctavodia](https://github.com/aloctavodia) in [#203](https://github.com/arviz-devs/arviz-plots/pull/203)
* Handles title spacing during creation of plot by [@The-Broken-Keyboard](https://github.com/The-Broken-Keyboard) in [#199](https://github.com/arviz-devs/arviz-plots/pull/199)
* Rename plot modules by [@aloctavodia](https://github.com/aloctavodia) in [#205](https://github.com/arviz-devs/arviz-plots/pull/205)
* Add group argument loo_pit ppc_pit by [@aloctavodia](https://github.com/aloctavodia) in [#206](https://github.com/arviz-devs/arviz-plots/pull/206)
* Fixes shared x property for appropriate plots ( like rank plot, trace plot etc ) by [@The-Broken-Keyboard](https://github.com/The-Broken-Keyboard) in [#208](https://github.com/arviz-devs/arviz-plots/pull/208)
* Add group argument to plot_prior_posterior by [@aloctavodia](https://github.com/aloctavodia) in [#209](https://github.com/arviz-devs/arviz-plots/pull/209)
* Small fixes to plot_prior_posterior by [@OriolAbril](https://github.com/OriolAbril) in [#216](https://github.com/arviz-devs/arviz-plots/pull/216)
* Add horizontal_align to annotate_xy by [@rohanbabbar04](https://github.com/rohanbabbar04) in [#214](https://github.com/arviz-devs/arviz-plots/pull/214)
* Fix error plot_ppc_dist, generalize title ppc galery by [@aloctavodia](https://github.com/aloctavodia) in [#220](https://github.com/arviz-devs/arviz-plots/pull/220)
* Update labelled title to include text by [@rohanbabbar04](https://github.com/rohanbabbar04) in [#222](https://github.com/arviz-devs/arviz-plots/pull/222)
* Allow prior predictive checks when observed_data group is not present by [@aloctavodia](https://github.com/aloctavodia) in [#226](https://github.com/arviz-devs/arviz-plots/pull/226)
* Fix bug plot_ppc_dist when passing references and add test by [@aloctavodia](https://github.com/aloctavodia) in [#228](https://github.com/arviz-devs/arviz-plots/pull/228)
* use scalar prob for quantile/tail ess by [@OriolAbril](https://github.com/OriolAbril) in [#240](https://github.com/arviz-devs/arviz-plots/pull/240)
* Use fill_between for  matplotlib histogram, instead of bar by [@aloctavodia](https://github.com/aloctavodia) in [#247](https://github.com/arviz-devs/arviz-plots/pull/247)
* Update add_reference_lines as a helper function by [@rohanbabbar04](https://github.com/rohanbabbar04) in [#230](https://github.com/arviz-devs/arviz-plots/pull/230)
* Backend should be none by [@aloctavodia](https://github.com/aloctavodia) in [#251](https://github.com/arviz-devs/arviz-plots/pull/251)
* Improve publish workflow by [@OriolAbril](https://github.com/OriolAbril) in [#252](https://github.com/arviz-devs/arviz-plots/pull/252)
* Update ref_dim in add_reference_lines by [@rohanbabbar04](https://github.com/rohanbabbar04) in [#254](https://github.com/arviz-devs/arviz-plots/pull/254)
* Explicit index support in PlotCollection.map by [@OriolAbril](https://github.com/OriolAbril) in [#241](https://github.com/arviz-devs/arviz-plots/pull/241)
* Add consistent legend support across backends by [@Advaitgaur004](https://github.com/Advaitgaur004) in [#115](https://github.com/arviz-devs/arviz-plots/pull/115)
* Fix cols definition plot_autocorr by [@aloctavodia](https://github.com/aloctavodia) in [#263](https://github.com/arviz-devs/arviz-plots/pull/263)
* rename add_reference_lines and reference_bands by [@aloctavodia](https://github.com/aloctavodia) in [#264](https://github.com/arviz-devs/arviz-plots/pull/264)
* psense_dist: Change order point-interval by [@aloctavodia](https://github.com/aloctavodia) in [#265](https://github.com/arviz-devs/arviz-plots/pull/265)
* Unify default value for `col_wrap` to 4 by [@aloctavodia](https://github.com/aloctavodia) in [#268](https://github.com/arviz-devs/arviz-plots/pull/268)
* Change batteries-included plots' call signature by [@aloctavodia](https://github.com/aloctavodia) in [#275](https://github.com/arviz-devs/arviz-plots/pull/275)
* update dim+sample_dims by [@aloctavodia](https://github.com/aloctavodia) in [#277](https://github.com/arviz-devs/arviz-plots/pull/277)
* Add tests plot_ppc_* by [@aloctavodia](https://github.com/aloctavodia) in [#280](https://github.com/arviz-devs/arviz-plots/pull/280)
* Add more tests by [@aloctavodia](https://github.com/aloctavodia) in [#281](https://github.com/arviz-devs/arviz-plots/pull/281)
* raise when extra keyword arguments make it to aes generation by [@OriolAbril](https://github.com/OriolAbril) in [#285](https://github.com/arviz-devs/arviz-plots/pull/285)
* More corrections in pair_focus_plot by [@The-Broken-Keyboard](https://github.com/The-Broken-Keyboard) in [#282](https://github.com/arviz-devs/arviz-plots/pull/282)

## Documentation
* Improve documentation by [@aloctavodia](https://github.com/aloctavodia) in [#193](https://github.com/arviz-devs/arviz-plots/pull/193)
* Improve docs for PIT plots by [@AlexAndorra](https://github.com/AlexAndorra) in [#195](https://github.com/arviz-devs/arviz-plots/pull/195)
* Several documentation improvements by [@OriolAbril](https://github.com/OriolAbril) in [#215](https://github.com/arviz-devs/arviz-plots/pull/215)
* Docstring update in add_reference_lines by [@rohanbabbar04](https://github.com/rohanbabbar04) in [#225](https://github.com/arviz-devs/arviz-plots/pull/225)
* Correct a typo: currenly -> currently by [@star1327p](https://github.com/star1327p) in [#233](https://github.com/arviz-devs/arviz-plots/pull/233)
* Clean gallery description for ppc plots by [@aloctavodia](https://github.com/aloctavodia) in [#232](https://github.com/arviz-devs/arviz-plots/pull/232)
* Make example less cluncky by [@aloctavodia](https://github.com/aloctavodia) in [#234](https://github.com/arviz-devs/arviz-plots/pull/234) 
* Add link to John K. Kruschke's Doing Bayesian Data Analysis book by [@star1327p](https://github.com/star1327p) in [#238](https://github.com/arviz-devs/arviz-plots/pull/238)
* Add more variation to distribution examples by [@aloctavodia](https://github.com/aloctavodia) in [#242](https://github.com/arviz-devs/arviz-plots/pull/242)
* Add style to docs by [@aloctavodia](https://github.com/aloctavodia) in [#248](https://github.com/arviz-devs/arviz-plots/pull/248)
* Link "batteries-included plots intro" page to Example Gallery by [@rohanbabbar04](https://github.com/rohanbabbar04) in [#249](https://github.com/arviz-devs/arviz-plots/pull/249)
* Correct a typo: instatiates -> instantiates by [@star1327p](https://github.com/star1327p) in [#255](https://github.com/arviz-devs/arviz-plots/pull/255)
* Revise a cross-ref in docs.md by [@star1327p](https://github.com/star1327p) in [#260](https://github.com/arviz-devs/arviz-plots/pull/260)
* Fix references by [@aloctavodia](https://github.com/aloctavodia) in [#266](https://github.com/arviz-devs/arviz-plots/pull/266)
* fixes docstring and error statement in compare function by [@The-Broken-Keyboard](https://github.com/The-Broken-Keyboard) in [#267](https://github.com/arviz-devs/arviz-plots/pull/267)
* Add overview/getting started tutorial by [@aloctavodia](https://github.com/aloctavodia) in [#269](https://github.com/arviz-devs/arviz-plots/pull/269)
* Improve tutorial notebooks by [@OriolAbril](https://github.com/OriolAbril) in [#261](https://github.com/arviz-devs/arviz-plots/pull/261)
* Proof of concept of explicit type hints for visuals, stats and aes_by_visuals by [@OriolAbril](https://github.com/OriolAbril) in [#283](https://github.com/arviz-devs/arviz-plots/pull/283)


## New Contributors
* [@AlexAndorra](https://github.com/AlexAndorra) made their first contribution in [#195](https://github.com/arviz-devs/arviz-plots/pull/195)
* [@suhaani-agarwal](https://github.com/suhaani-agarwal) made their first contribution in [#153](https://github.com/arviz-devs/arviz-plots/pull/153)
* [@rohanbabbar04](https://github.com/rohanbabbar04) made their first contribution in [#210](https://github.com/arviz-devs/arviz-plots/pull/210)
* [@star1327p](https://github.com/star1327p) made their first contribution in [#233](https://github.com/arviz-devs/arviz-plots/pull/233)
* [@Advaitgaur004](https://github.com/Advaitgaur004) made their first contribution in [#115](https://github.com/arviz-devs/arviz-plots/pull/115)

**Full Changelog**: https://github.com/arviz-devs/arviz-plots/compare/v0.5.0...v0.6.0

[Changes][v0.6.0]


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


[v0.7.0]: https://github.com/arviz-devs/arviz-plots/compare/v0.6.0...v0.7.0
[v0.6.0]: https://github.com/arviz-devs/arviz-plots/compare/v0.5.0...v0.6.0
[v0.5.0]: https://github.com/arviz-devs/arviz-plots/compare/v0.4.0...v0.5.0
[v0.4.0]: https://github.com/arviz-devs/arviz-plots/tree/v0.4.0

<!-- Generated by https://github.com/rhysd/changelog-from-release v3.9.1 -->
