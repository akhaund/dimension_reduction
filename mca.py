#!/usr/bin/env python3

# WIP

# Author: Anshuman Khaund <ansh.khaund@gmail.com>
# Date: 11/10/2020
# Acknowledgements:
#      https://github.com/esafak/mca
#      https://personal.utdallas.edu/~herve/

import numpy as np
import pandas as pd

import scipy.sparse as sp
import scipy.linalg as sl
import scipy.sparse.linalg as spl

from collections import namedtuple

import plots


class OutputMCA:
    """ Multiple Correspondence Analysis (under construction) """

    def __init__(self, df) -> None:
        if self._check_sparcity():
            self._sparse_type = sp.csr_matrix
            self._dat = self._sparse_type(df).values
            self._sum = self._sparse_type.sum
            self._diag = sp.diags
            self._svd = spl.svds
            self._outer = self._sparse_outer
            self._inv = spl.inv
        else:
            self._dat = df.values
            self._sum = np.sum
            self._diag = np.diag
            self._svd = sl.svd
            self._outer = np.outer
            self._inv = sl.inv
        self._mca = self._fit_mca()

    def _check_sparcity(self):
        """ Check of the input matrix is sparse.
        """
        is_sparse = False
        return is_sparse

    def _sparse_outer(self, a, b):  # todo: Find simpler alternative
        a, b = map(self._sparse_type.toarray, (a, b))
        outer_product = np.outer(a, b)
        sparse_outer_product = self._sparse_type(outer_product)
        return sparse_outer_product

    def _fit_mca(self):
        X = self._dat
        sum = self._sum
        diag = self._diag
        svd = self._svd
        outer = self._outer
        inv = self._inv
        mca_out = namedtuple("mca_attrs",
                             ["inertia", "latent_rows", "latent_cols"])
        Z = X / sum(X)  # probability matrix
        sum_row = sum(Z, axis=1)
        sum_col = sum(Z, axis=0)
        Z_centered = Z - outer(sum_row, sum_col)
        # weighted SVD (GSVD)
        weight_row = np.sqrt(inv(diag(sum_row)))
        weight_col = np.sqrt(inv(diag(sum_col)))
        Z_weighted = weight_row @ Z_centered @ weight_col
        P, S, Qh = svd(Z_weighted, full_matrices=False)
        Q = Qh.T
        F = weight_row @ P @ diag(S)  # row factors
        G = weight_col @ Q @ diag(S)  # column factors

        return mca_out(S ** 2, F, G)

    def explained_variance_plot(self):
        """ Pareto chart of explained variance
        """
        inertia = getattr(self._mca, "inertia")
        expl_var_ratio = (inertia/inertia.sum())
        expl_var = pd.DataFrame({
            "var_exp": expl_var_ratio,
            "cumul_var_exp": expl_var_ratio.cumsum()
        })
        expl_var.index += 1

        # Plotting
        fig = plots.explained_variance_plot(
            df=expl_var,
            x_title="Latent Variable Rank",
            y_title="Variance Explained",
            title="Variance explained by Latent Variables"
        )
        return fig

    def scree_plot(self):
        eig_vals = pd.DataFrame({
            "eig_val": getattr(self._mca, "inertia")
        })
        eig_vals.index += 1
        fig = plots.scree_plot(
            df=eig_vals,
            x_title="Latent Variable Rank",
            y_title="Inertia (Eigen values)",
            title="Inertia by Latent Variables"
        )
        return fig

    def get_projections():
        return

    def get_components():
        return


# Test
if __name__ == "__main__":
    df = pd.read_csv("data/burgundies.csv",
                     skiprows=1, index_col=0, header=0)

    OutputMCA(df).explained_variance_plot()
    OutputMCA(df).scree_plot()
