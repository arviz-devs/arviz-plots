"""Plotly templates for ArviZ styles."""
import plotly.graph_objects as go

axis_common = {"showgrid": False, "ticks": "outside", "showline": True, "zeroline": False}

arviz_cetrino_template = go.layout.Template()
arviz_cetrino_template.layout.paper_bgcolor = "white"
arviz_cetrino_template.layout.plot_bgcolor = "white"
arviz_cetrino_template.layout.polar.bgcolor = "white"
arviz_cetrino_template.layout.ternary.bgcolor = "white"
arviz_cetrino_template.layout.margin = {"l": 50, "r": 10, "t": 40, "b": 45}
arviz_cetrino_template.layout.xaxis = axis_common
arviz_cetrino_template.layout.yaxis = axis_common
arviz_cetrino_template.layout.colorway = [
    "#3cd186",
    "#fcd026",
    "#ec8f26",
    "#c969eb",
    "#2b8c92",
    "#a91a32",
    "#2f2cec",
    "#958152",
]

arviz_tenui_template = go.layout.Template()
arviz_tenui_template.layout.paper_bgcolor = "white"
arviz_tenui_template.layout.plot_bgcolor = "white"
arviz_tenui_template.layout.polar.bgcolor = "white"
arviz_tenui_template.layout.ternary.bgcolor = "white"
arviz_tenui_template.layout.margin = {"l": 50, "r": 10, "t": 40, "b": 45}
arviz_tenui_template.layout.xaxis = axis_common
arviz_tenui_template.layout.yaxis = axis_common
arviz_tenui_template.layout.colorway = [
    "#84d3ee",
    "#e7b08b",
    "#f7ea8d",
    "#b6e4d0",
    "#c88bb8",
    "#5bb2ed",
    "#7b9091",
    "#c7b4c0",
]


arviz_variat_template = go.layout.Template()
arviz_variat_template.layout.paper_bgcolor = "white"
arviz_variat_template.layout.plot_bgcolor = "white"
arviz_variat_template.layout.polar.bgcolor = "white"
arviz_variat_template.layout.ternary.bgcolor = "white"
arviz_variat_template.layout.margin = {"l": 50, "r": 10, "t": 40, "b": 45}
arviz_variat_template.layout.xaxis = axis_common
arviz_variat_template.layout.yaxis = axis_common
arviz_variat_template.layout.colorway = [
    "#36acc6",
    "#f66d7f",
    "#fac364",
    "#7c2695",
    "#228306",
    "#a252f4",
    "#63f0ea",
    "#a77e4f",
]


arviz_vibrant_template = go.layout.Template()
arviz_vibrant_template.layout.paper_bgcolor = "white"
arviz_vibrant_template.layout.plot_bgcolor = "white"
arviz_vibrant_template.layout.polar.bgcolor = "white"
arviz_vibrant_template.layout.ternary.bgcolor = "white"
arviz_vibrant_template.layout.margin = {"l": 50, "r": 10, "t": 40, "b": 45}
arviz_vibrant_template.layout.xaxis = axis_common
arviz_vibrant_template.layout.yaxis = axis_common
arviz_vibrant_template.layout.colorway = [
    "#008b92",
    "#f15c58",
    "#48cdef",
    "#bc56f0",
    "#8ddb17",
    "#f9cf9c",
    "#c90a4e",
    "#b0aaa2",
]
