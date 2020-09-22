import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

def t_test(data: pd.Series, hypo: int, bootstrap_sample = False, alpha = 0.05):
    "Performs simple t_test" ""
    # plot standardized data
    # power function
    # 

    n = data.size
    df = n - 1 # degrees of freedom of t-distribution
    alpha = alpha
    mu_0 = hypo
    mean = data.mean()
    std = data.std()
    alpha_ = 1 - (alpha / 2)
    test_statistic = np.sqrt(n) * (mean - mu_0) / std
    p_val = 2 * np.min([stats.t.cdf(test_statistic, df), 1 - stats.t.cdf(test_statistic,df)])
    #c1 = stats.norm.ppf(alpha_, 0, 1)
    #c0 = -c1
    #1 - stats.norm(0, 1).cdf(c1 - np.sqrt(n) * (mu - mu_0) / s) + stats.norm(0, 1).cdf(c0 - np.sqrt(n) * (mu - mu_0) / s)

    return p_val, test_statistic

ser = pd.Series(np.random.randn(100))

p, t = t_test(ser, 0)
def rank_test():

    return p_val, t_statistic


def signed_rank_test():

    return p_val


def anova():

    return


def manova():

    return

