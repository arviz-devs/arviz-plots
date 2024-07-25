"""Plotly templates for ArviZ styles."""
import plotly.graph_objects as go

arviz_clean_template = go.layout.Template()

arviz_clean_template.layout.paper_bgcolor = "white"
arviz_clean_template.layout.plot_bgcolor = "white"
arviz_clean_template.layout.polar.bgcolor = "white"
arviz_clean_template.layout.ternary.bgcolor = "white"
axis_common = {"showgrid": False, "ticks": "outside", "showline": True, "zeroline": False}
arviz_clean_template.layout.xaxis = axis_common
arviz_clean_template.layout.yaxis = axis_common

arviz_clean_template.layout.colorway = [
    "#1a6587",
    "#a6ccfe",
    "#f98d74",
    "#f5e257",
    "#e5441a",
    "#5b8073",
    "#b66edd",
    "#9b403d",
    "#969bab",
    "#c1c1c1",
]
