#!/usr/bin/env python3

# Author: Anshuman Khaund <ansh.khaund@gmail.com>

import sys
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import plots


class OutputPCA:
    """ Returns,
        1. Pareto-chart of the explained variance, or
        2. Low-dimensional (2D or 3D) projection of the data, or
        3. The principal components on which to project the data.
    """

    def __init__(self,
                 df: pd.DataFrame,
                 ) -> None:
        # Data columns being standardized prior to PCA
        df = pd.DataFrame(
            data=StandardScaler().fit_transform(df.values),
            columns=df.columns,
            index=df.index,
        )
        self._df = df
        self._pca = PCA().fit(df.values)

    def get_explained_variance(self):
        """ Pareto-Chart of the explained variance
        """
        explained_variance = pd.DataFrame({
            "var_exp": self._pca.explained_variance_ratio_,
            "cumul_var_exp": self._pca.explained_variance_ratio_.cumsum(),
        })
        explained_variance.index += 1
        fig = plots.explained_variance_plot(
            df=explained_variance,
            x_title="Principal Component Rank",
            y_title="Variance Explained",
            title="Variance explained by Principal Components",
        )
        return fig

    def get_scree_plot(self):
        eigen_values = pd.DataFrame({
            "eigen_values": self._pca.explained_variance_,
        })
        eigen_values.index += 1
        fig = plots.scree_plot(
            df=eigen_values,
            x_title="Principal Component rank",
            y_title="Eigen Value",
            title="Eigen Values by Principal Components",
        )
        return fig

    def get_projections(self,
                        project_features: bool = True,
                        labels=None,
                        n_components: int = 2,
                        ):
        """ Low-dimensional (2D or 3D) projection of the data
        """
        features, indeces = self._df.columns, self._df.index.values
        # input checks
        if n_components not in {2, 3}:
            print(
                "\033[1m"
                "Input Error for 'visualize' function. \n"
                "\033[0m"
                f"Given: {n_components=}. It must be 2 or 3. \n",
                file=sys.stderr,
            )
            return None
        pca_components = pd.DataFrame(
            data=self._pca.components_[:n_components, :].T,  # comps as rows
            index=features,
            columns=["PC" + str(i + 1) for i in range(n_components)],
        )
        pca_transformed = pd.DataFrame(
            data=np.concatenate(
                (self._pca.transform(self._df.values)[:, :n_components],
                 indeces.reshape(len(indeces), 1)),
                axis=1,
            ),
            columns=list(pca_components.columns) + ["idx"],
        )
        if labels is not None:
            pca_transformed["label"] = labels
        fig = plots.low_dimensional_projection(
            n_comp=n_components,
            components=pca_components,
            transforms=pca_transformed,
            project_features=project_features,
            title="Principal Component Analysis",
        )
        return fig

    def get_components(self,
                       n_components: int,
                       ):
        """ The principal components on which to project the data
        """
        print("Principal components returned as 'rows'.")
        return self._pca.components_[:n_components, :]


if __name__ == "__main__":

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
