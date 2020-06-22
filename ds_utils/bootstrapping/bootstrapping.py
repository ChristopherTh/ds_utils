import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from functools import partial


def explain_series(
    series: pd.Series,
    num_samples=500,
    agg_functions=["mean", "min", "max", "median", "std", percentile_25],
):
    # currently supports str methods

    cols = agg_functions
    for i, col in enumerate(cols):
        if not isinstance(col, str):
            cols[i] = col.__name__

    df = pd.DataFrame(columns=agg_functions)

    for i in range(num_samples):
        cur_series = series.sample(series.size, replace=True).agg(agg_functions)
        df = df.append(cur_series, ignore_index=True)

    for col in agg_functions:
        fig, axes = plt.subplots()
        sns.distplot(df[col], kde=False, ax=axes)

    return df


percentile_25 = lambda x: np.percentile(x, q=25)
percentile_25.__name__ = "lala"
dff = explain_series(pd.Series(np.random.randn(500)))
tt = pd.DataFrame(columns=["mean", "std"])

ss = pd.Series(np.random.randn(500)).agg(np.percentile, **{"q": 25})
ss1 = pd.Series([1, 2, 4]).agg(["mean", np.percentile])

pd.concat([tt, ss], axis=1)


tt.append(ss, ignore_index=True)


pd.concat([tt, pd.Series([1, 3])], ignore_index=True, axis=1)

