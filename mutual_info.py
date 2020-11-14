#!/usr/bin/env python3

# Author: Anshuman Khaund <ansh.khaund@gmail.com>

import pandas as pd
from sklearn.feature_selection import mutual_info_classif


def get_mutual_info(df, labels, discretes,
                    save_plot=None,
                    show_plot: bool = True):
    """ Mutual Information
    """
    mi = pd.Series(mutual_info_classif(df,
                                       labels,
                                       discrete_features=discretes),
                   index=df.columns).sort_values(ascending=False)
    fig = px.line(y=mi)
    fig.update_layout(
        xaxis={"title": "feature rank"},
        yaxis={"title": "mutual_info"},
        title="Mutual info. for all features (descending)")
    if show_plot:
        fig.show()
    if save_plot is not None:
        save_plot(fig, "mutual_info")
    return mi


# Test
if __name__ == "__main__":
    pass
