#!/usr/bin/env python3

# Author: Anshuman Khaund <ansh.khaund@gmail.com>

# %% Imports
import pandas as pd
from pca import OutputPCA

# %% Test PCA with the Iris data set
iris = pd.read_csv("../data/iris.csv")
data, labels = (iris.drop(columns="labels"),
                iris["labels"])
# 2D visualization
OutputPCA(data).get_projections(
    labels=labels,
).show()
# 3D visualization
OutputPCA(data).get_projections(
    labels=labels,
    n_components=3,
).show()
# Scree Plot
OutputPCA(data).get_scree_plot().show()
# Explained variance
OutputPCA(data).get_explained_variance().show()
# Get components
OutputPCA(data).get_components(n_components=2)
