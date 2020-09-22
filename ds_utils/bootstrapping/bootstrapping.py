import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def percentile_25(x):  # dont use lambda function because need __name__

    return np.percentile(x, q=25, axis=0)


def percentile_75(x):

    return np.percentile(x, q=75, axis=0)


def explain_series(
    series: pd.Series, num_samples=50, standardize=True, plot=False,
):
    if standardize:
        series = (series - series.mean()) / series.std()
    agg_functions = [
        "mean",
        "min",
        "max",
        "median",
        "std",
        percentile_25,
        percentile_75,
    ]
    cols = [
        "mean",
        "min",
        "max",
        "median",
        "std",
        percentile_25.__name__,
        percentile_75.__name__,
    ]
    df = pd.DataFrame(columns=cols)

    for i in range(num_samples):
        cur_series = series.sample(series.size, replace=True).agg(agg_functions)
        df = df.append(cur_series, ignore_index=True)

    if plot:
        for col in cols:
            fig, axes = plt.subplots()
            sns.distplot(df[col], kde=False, ax=axes)

    return df


dff = explain_series(pd.Series(np.random.randn(500)))

