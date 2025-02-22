"""Plotly templates for ArviZ styles."""
import plotly.graph_objects as go

arviz_variat_template = go.layout.Template()

arviz_variat_template.layout.paper_bgcolor = "white"
arviz_variat_template.layout.plot_bgcolor = "white"
arviz_variat_template.layout.polar.bgcolor = "white"
arviz_variat_template.layout.ternary.bgcolor = "white"
arviz_variat_template.layout.margin = {"l": 50, "r": 10, "t": 40, "b": 45}
axis_common = {"showgrid": False, "ticks": "outside", "showline": True, "zeroline": False}
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
    "#000000",
    "#6f6f6f",
    "#b7b7b7",
]


arviz_cetrino_template = go.layout.Template()

arviz_cetrino_template.layout.paper_bgcolor = "white"
arviz_cetrino_template.layout.plot_bgcolor = "white"
arviz_cetrino_template.layout.polar.bgcolor = "white"
arviz_cetrino_template.layout.ternary.bgcolor = "white"
arviz_cetrino_template.layout.margin = {"l": 50, "r": 10, "t": 40, "b": 45}
axis_common = {"showgrid": False, "ticks": "outside", "showline": True, "zeroline": False}
arviz_cetrino_template.layout.xaxis = axis_common
arviz_cetrino_template.layout.yaxis = axis_common

arviz_cetrino_template.layout.colorway = [
    "#009988",
    "#9238b2",
    "#d2225f",
    "#ec8f26",
    "#fcd026",
    "#3cd186",
    "#a57119",
    "#2f5e14",
    "#f225f4",
    "#8f9fbf",
]


arviz_vibrant_template = go.layout.Template()

arviz_vibrant_template.layout.paper_bgcolor = "white"
arviz_vibrant_template.layout.plot_bgcolor = "white"
arviz_vibrant_template.layout.polar.bgcolor = "white"
arviz_vibrant_template.layout.ternary.bgcolor = "white"
arviz_vibrant_template.layout.margin = {"l": 50, "r": 10, "t": 40, "b": 45}
axis_common = {"showgrid": False, "ticks": "outside", "showline": True, "zeroline": False}
arviz_vibrant_template.layout.xaxis = axis_common
arviz_vibrant_template.layout.yaxis = axis_common

arviz_vibrant_template.layout.colorway = [
    "#008b92",
    "#f15c58",
    "#48cdef",
    "#98d81a",
    "#997ee5",
    "#f5dc9d",
    "#c90a4e",
    "#145393",
    "#323232",
    "#616161",
]
