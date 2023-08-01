# Glossary


:::{glossary}
chart
  Highest level data visualization structure. All plotted elements
  are contained within a chart or its children.

plot
  Area (or areas) where the data will be plotted into. A {term}`chart`
  can contain multiple {term}`facetted` plots.

artist
  Visual element added by `arviz-base`

facetting
facetted
  Generate multiple similar {term}`plot` elements with each of them
  referring to a specific property or value of the data.
:::

## Equivalences with library specific objects

| arviz-base name | matplotlib   | bokeh   |
|-----------------|--------------|---------|
| chart           | figure       | layout  |
| plot/target     | axes/subplot | figure  |
| artist          | artist       | glyph   |
