#!/usr/bin/env python3

# WIP

# Author: Anshuman Khaund <ansh.khaund@gmail.com>
# Date: 11/10/2020
# Acknowledgements:
#      https://github.com/esafak/mca
#      https://personal.utdallas.edu/~herve/Abdi-MCA2007-pretty.pdf

import numpy as np
import pandas as pd

import scipy.sparse as sp

from scipy.linalg import svd

import plots


class OutputMCA:
    """ Multiple Correspondence Analysis (under construction) """

    def __init__(self, df) -> None:
        if self._check_sparcity():
            self._sparse_type = sp.csr_matrix
            self._dat = df.values
            self._sum = self._sparse_type.sum
            self._diag = sp.diags
            self._svd = sp.linalg.svds
            self._outer = self._sparse_outer
        else:
            self._dat = df.values
            self._sum = np.sum
            self._diag = np.diag
            self._svd = svd
            self._outer = np.outer

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

    def fit_mca(self):
        X = self._dat
        sum = self._sum
        diag = self._diag
        svd = self._svd
        outer = self._outer

        eps = np.finfo(float).eps
        N = sum(X)
        Z = X / N

        sum_r = sum(Z, axis=1)
        sum_c = sum(Z, axis=0)

        Z_expected = outer(sum_r, sum_c)
        Z_residual = Z - Z_expected

        D_r_sqrt = np.sqrt(diag(1/(sum_r + eps)))
        D_c_sqrt = np.sqrt(diag(1/(sum_c + eps)))

        mca_mat = D_r_sqrt @ Z_residual @ D_c_sqrt
        _, S, Qh = svd(mca_mat, full_matrices=False)
        Q = Qh.T

        # G = D_c_sqrt @ Q @ np.diag(S)

        # eig_vals = S ** 2
        # expl_var_ratio = eig_vals / eig_vals.sum()
        # expl_var = pd.DataFrame({
        #     "var_exp": expl_var_ratio,
        #     "cumul_var_exp": expl_var_ratio.cumsum()})

        # # Plotting
        # fig = plots.explained_variance_plot(expl_var)
        return


# Test
if __name__ == "__main__":
    df = pd.read_csv("data/burgundies.csv",
                     skiprows=1, index_col=0, header=0)

OutputMCA(df).do_mca()
