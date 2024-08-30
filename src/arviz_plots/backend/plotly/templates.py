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
    "#36acc6",
    "#f66d7f",
    "#fac364",
    "#7c2695",
    "#228306",
    "#a252f4",
    "#63f0ea",
    "#000000",
    "#6f6f6f",
    "#b7b7b7",
]
