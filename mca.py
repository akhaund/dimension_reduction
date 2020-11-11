#!/usr/bin/env python3

# Author: Anshuman Khaund <ansh.khaund@gmail.com>
# Date: 11/10/2020
# Acknowledgements:
#      https://github.com/esafak/mca
#      https://personal.utdallas.edu/~herve/Abdi-MCA2007-pretty.pdf

import numpy as np
import pandas as pd


from scipy.linalg import svd

import plots


class OutputMCA:
    """ Multiple Correspondence Analysis (under construction) """

    def __init__(self, df) -> None:
        self._df = df
        pass

    def check_sparcity(self):
        is_sparse = False
        return is_sparse

    def do_mca(self):
        """"""
        # x = df.values
        # N = np.sum(x).sum()
        # Z = x / N

        # sum_r = np.sum(Z, axis=1)
        # sum_c = np.sum(Z, axis=0)

        # Z_expected = np.outer(sum_r, sum_c)
        # Z_residual = Z - Z_expected

        # D_r_sqrt = np.sqrt(np.diag(sum_r ** -1))
        # D_c_sqrt = np.sqrt(np.diag(sum_c ** -1))

        # mca_mat = D_r_sqrt @ Z_residual @ D_c_sqrt
        # _, S, Qh = svd(mca_mat, full_matrices=False)
        # Q = Qh.T

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
