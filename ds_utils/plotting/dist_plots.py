import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from itertools import combinations

df = sns.load_dataset("titanic")

def dist_plots(dependent_variable, df):

    cwd = Path.cwd()
    plot_path = cwd.joinpath("plotting")
    dist_plots_path = plot_path.joinpath("dist_plots")

    if not plot_path.exists():
        plot_path.mkdir()
        dist_plots_path.mkdir()
        
    if not dist_plots_path:
        dist_plots_path.mkdir()

    
    cat_cols = df.select_dtypes('category').columns.to_list()

    for col in df.columns.to_list():

        if str(df[col].dtype) in 'category':

            cat_plot = sns.countplot(x = col, data = df)
            plt.savefig(dist_plots_path.joinpath(f"{col}.png"))
            plt.clf()

        if str(df[col].dtype) == 'float64':

            num_plot = sns.distplot(df[col])
            plt.savefig(dist_plots_path.joinpath(f"{col}.png"))
            plt.clf()


                
def interaction_plots(dependent_variable, df):

    cwd = Path.cwd()
    plot_path = cwd.joinpath("plotting")
    inter_plot = plot_path.joinpath("interaction_plots")

    if not inter_plot.exists():
            inter_plot.mkdir()
    cat_cols = df.select_dtypes('category').columns.to_list()
    cat_cols.insert(0, dependent_variable)
    for a, b, c in combinations(cat_cols, 3):

        print(a, b, c)

        if a == dependent_variable:

            g = sns.FacetGrid(df, col = c, col_wrap=3)
            g.map(sns.pointplot, b,dependent_variable)
            plt.savefig(inter_plot.joinpath(f"{b + c}.png"))
            plt.clf()

interaction_plots('tip', df)


def scatter_plots(dependent_variable, df):

    cwd = Path.cwd()
    plot_path = cwd.joinpath("plotting")
    scatter_path = plot_path.joinpath("scatter_plots")

    if not scatter_path.exists():
        scatter_path.mkdir()

    columns = df.columns.to_list()
    columns.remove(dependent_variable)

    if str(df[dependent_variable].dtype) == 'int64':

        for col in columns:

            if str(df[col].dtype) == 'float64':

                rel_plot = sns.lmplot(x = col, y = dependent_variable, logistic=True,y_jitter=.03, data = df, ci = True)
                plt.savefig(scatter_path.joinpath(f"{col}.png"))
                plt.clf()

            if str(df[col].dtype) == 'category':

                rel_plot = sns.stripplot(x = col, y = dependent_variable, data = df)
                plt.savefig(scatter_path.joinpath(f"{col}.png"))
                plt.clf()

    else:

        for col in columns:

            if str(df[col].dtype) == 'float64':

                rel_plot = sns.regplot(x = col, y = dependent_variable, lowess=True, data = df)
                plt.savefig(scatter_path.joinpath(f"{col}.png"))
                plt.clf()

            if str(df[col].dtype) == 'category':

                rel_plot = sns.stripplot(x = col, y = dependent_variable, data = df)
                plt.savefig(scatter_path.joinpath(f"{col}.png"))
                plt.clf()

scatter_plots('survived', df)

sns.lmplot(x = 'age', y = 'survived', logistic=True,y_jitter=.03, data = df, ci = False)

scatter_plots('age', df)


def conditional_scatter_plots(dependent_variable, df):

    cwd = Path.cwd()
    plot_path = cwd.joinpath("plotting")
    cond_scatter_path = plot_path.joinpath("cond. scatter plots")

    if not cond_scatter_path.exists():
            cond_scatter_path.mkdir()
    num_cols = df.select_dtypes('float64').columns.to_list()
    num_cols.remove(dependent_variable)
    cat_cols = df.select_dtypes('category').columns.to_list()

    if len(cat_cols) == 0:

        print('No category columns in df.')
        return

    for num_col in num_cols:

        for cat_col in cat_cols:

            rel_plot = sns.lmplot(x = num_col, y = dependent_variable, data = df, col = cat_col, col_wrap = 3, lowess=True)
            plt.savefig(cond_scatter_path.joinpath(f"{cat_col}.png"))
            plt.clf()


conditional_scatter_plots('tip', df)


df.head()

       

dist_plots('total', df)

