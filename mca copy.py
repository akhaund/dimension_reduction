#!/usr/bin/env python3

# Author: Anshuman Khaund <ansh.khaund@gmail.com>

import numpy as np
import pandas as pd

from scipy.linalg import svd

from plots import Plotters as plots


class OutputMCA:
    """ Multiple Correspondence Analysis (under construction) """

    def __init__(self) -> None:
        pass

    def do_mca(df):
        """"""
        x = df.values
        N = np.sum(x)
        Z = x / N

        sum_r = np.sum(Z, axis=1)
        sum_c = np.sum(Z, axis=0)

        Z_expected = np.outer(sum_r, sum_c)
        Z_residual = Z - Z_expected

        D_r_sqrt = np.sqrt(np.diag(sum_r ** -1))
        D_c_sqrt = np.sqrt(np.diag(sum_c ** -1))

        mca_mat = D_r_sqrt @ Z_residual @ D_c_sqrt
        _, S, Qh = svd(mca_mat, full_matrices=False)
        Q = Qh.T

        G = D_c_sqrt @ Q @ np.diag(S)

        eig_vals = S ** 2
        expl_var_ratio = eig_vals / eig_vals.sum()
        expl_var = pd.DataFrame({
            "var_exp": expl_var_ratio,
            "cumul_var_exp": expl_var_ratio.cumsum()})

        # Plotting
        fig = plots.explained_variance_plot(expl_var)
        return fig, G


# Test
if __name__ == "__main__":
    pass
