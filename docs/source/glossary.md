# Glossary


:::{glossary}
aesthetic
aesthetics
  When used as a noun, we use _an aesthetic_ as a graphical property that is
  being used to encode data. 

  Moreover, within `arviz_plots` _aesthetics_ can actually be any arbitrary
  keyword argument accepted by the plotting function being used.

aesthetic mapping
  We use _aesthetic mapping_ to indicate the relation between the {term}`aesthetics`
  in our plot and properties in our dataset.

chart
  Highest level data visualization structure. All plotted elements
  are contained within a chart or its children.

plot
plots
  Area (or areas) where the data will be plotted into. A {term}`chart`
  can contain multiple {term}`facetted` plots.

artist
artists
  Visual element added by `arviz-plots`

facetting
facetted
  Generate multiple similar {term}`plot` elements with each of them
  referring to a specific property or value of the data.
:::

## Equivalences with library specific objects

| arviz-plots name | matplotlib   | bokeh   |
|------------------|--------------|---------|
| chart            | figure       | layout  |
| plot             | axes/subplot | figure  |
| artist           | artist       | glyph   |
