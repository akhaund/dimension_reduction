#!/usr/bin/env python3

# Author: Anshuman Khaund <ansh.khaund@gmail.com>

import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from statistics import median


def explained_variance_plot(df):
    """ Pareto-Chart of the explained variance
    """
    trace1 = dict(
        type="bar",
        x=df.index,
        y=df["var_exp"]
    )
    trace2 = dict(
        type="scatter",
        x=df.index,
        y=df["cumul_var_exp"],
        line=dict(color="#dadbb2")
    )
    traces = [trace1, trace2]
    layout = go.Layout(
        showlegend=False,
        template="plotly_dark",
        xaxis=dict(title="Principal Component rank"),
        yaxis=dict(title="Variance Explained",
                   tickformat=",.1%",
                   gridcolor="#828994"),
        title="Variance explained by Principal Components"
    )
    fig = go.Figure(traces, layout)
    return fig


def scree_plot(df):
    trace = dict(
        type="scatter",
        x=df.index,
        y=df["eig_val"],
        line=dict(color="#dadbb2")
    )
    layout = go.Layout(
        showlegend=False,
        template="plotly_dark",
        xaxis=dict(title="Principal Component rank",
                   tickformat="d"),
        yaxis=dict(title="Eigen Value",
                   tickformat=",.3f",
                   gridcolor="#828994"),
        title="Eigen Values by Principal Components"
    )
    fig = go.Figure(trace, layout)
    return fig


def low_dimensional_projection(n_comp, components, transforms,
                               project_features):
    """ 2d/3d projections from PCA/MCA
    """
    if n_comp == 2:
        plotter = px.scatter
        axes = dict(zip(("x", "y"), components.columns))
    elif n_comp == 3:
        plotter = px.scatter_3d
        axes = dict(zip(("x", "y", "z"), components.columns))
    # edit hover data
    hover_data = dict.fromkeys(components.columns, False)
    hover_data.update(dict(label=True, idx=True))
    fig = plotter(  # ? What is this warning from Pylance
        transforms,
        **axes,
        color="label",
        hover_data=hover_data,
        title="Principal Component Analysis",
        template="plotly_dark"
    )

    # Projections of features
    # todo: Alternate ways for projections & annotations, they don't look good
    if n_comp == 2 and project_features:
        # scale feature prokections
        cols = "PC1 PC2".split()
        transforms["norm"] = transforms[cols].apply(np.linalg.norm, axis=1)
        components["norm"] = components[cols].apply(np.linalg.norm, axis=1)
        scale = (median(transforms["norm"]/median(components["norm"])))
        components = components * scale * 0.75
        for i, val in enumerate(components.index):
            fig.add_shape(
                type="line",
                x0=0,
                y0=0,
                x1=components.iloc[i, 0],
                y1=components.iloc[i, 1],
                line=dict(color="#dadbb2", width=1)
            )
            fig.add_annotation(
                x=components.iloc[i, 0],
                y=components.iloc[i, 1],
                text=val,
                showarrow=True,
                arrowsize=2,
                arrowhead=2
            )
    return fig
