import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import statsmodels.api as sm
import matplotlib.dates as mdates
import pandas_datareader.data as web
import streamlit as st
gs = web.DataReader("GS", data_source="yahoo", start="2006-01-01", end="2009-01-01")

# next steps -> more than one ts
# seasonal decomposition
# currently supports yyyy-mm-dd format what about only yyyy or yyyy-mm
# add first difference plot
class ts_eda:
    def __init__(self, ts, freq="d", lags=4):

        self.ts = ts
        self.cwd = Path.cwd()
        self.plot_folder = self.cwd.joinpath("ts_plots")
        if not self.plot_folder.exists():
            self.plot_folder.mkdir()
        self.ts_df = self.make_df()

        self.lags = lags
        self.create_lags(self.ts_df, lags)

        # make quick plot to infer additive or multiplicative seasonality
        sns.lineplot(x=ts.index, y=ts)

    def make_df(self):
        month_map = {
            1: "Jan",
            2: "Feb",
            3: "Mar",
            4: "Apr",
            5: "May",
            6: "Jun",
            7: "Jul",
            8: "Aug",
            9: "Sep",
            10: "Oct",
            11: "Nov",
            12: "Dec",
        }
        week_map = {
            1: "Monday",
            2: "Tuesday",
            3: "Wednesday",
            4: "Thursday",
            5: "Friday",
            6: "Saturday",
            7: "Sunday",
        }

        ts_df = pd.DataFrame(self.ts)
        ts_df.columns = ["y"]
        ts_df["month"] = self.ts.index.month
        ts_df.month = ts_df.month.map(month_map)
        ts_df["year"] = self.ts.index.year
        ts_df["week"] = self.ts.index.weekday + 1
        ts_df.week = ts_df.week.map(week_map)

        return ts_df

    def create_lags(self, df: pd.DataFrame, fill_value=np.nan):
        for i in np.arange(1, self.lags + 1):
            df[f"lag_{i}"] = np.append(df["y"].iloc[i:].values, [fill_value] * (i))

    def timeplot(self):
        fig, axs = plt.subplots()

        sns.lineplot(x=self.ts.index, y=self.ts, ax=axs)
        fig.autofmt_xdate()

        plt.savefig(self.plot_folder / "timeplot.png")
        plt.clf()

    def season_plot(self,):
        fig, axs = plt.subplots()

        sns.lineplot(x="month", y="y", hue="year", data=self.ts_df, ci=False, ax=axs)
        plt.savefig(self.plot_folder / "season1.png")
        st.pyplot(fig)
        plt.clf()

        def plot_mena(x, **kwargs):

            plt.hlines(x.mean(), self.ts_df.year.min(), self.ts_df.year.max())

        g = sns.FacetGrid(self.ts_df, col="month", sharex=True, sharey=False)
        g.map(sns.lineplot, "year", "y", ci=False)
        g.map(plot_mena, "y")
        plt.savefig(self.plot_folder / "season2.png")
        plt.clf()

    def acf(self,):
        fig, axes = plt.subplots()
        sm.graphics.tsa.plot_acf(self.ts_df["y"], lags=40, ax=axes)
        plt.savefig(self.plot_folder / "auto_correlation_function.png")
        plt.clf()

    def pacf(self,):
        fig, axes = plt.subplots()

        sm.graphics.tsa.plot_pacf(self.ts_df["y"], lags=40, ax=axes)
        plt.savefig(self.plot_folder / "partial_auto_correlation_function.png")
        plt.clf()

    def polar_plot(self,):
        fig, axes = plt.subplots(figsize=(20, 20))

        axes = plt.subplot(projection="polar")
        axes.set_theta_direction(-1)
        axes.set_theta_zero_location("N")
        lines, labels = plt.thetagrids(
            (0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330),
            labels=(
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "Mai",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ),
            fmt=None,
        )

        for year in self.ts_df.index.year.unique().to_list():
            times = pd.date_range(
                self.ts_df[self.ts_df.index.year == year].index.min().to_pydatetime(),
                self.ts_df[self.ts_df.index.year == year].index.max().to_pydatetime(),
            )
            
            t = mdates.date2num(
                (self.ts_df.loc[self.ts_df.index.year == year].index).to_pydatetime()
            )

            y = self.ts_df[self.ts_df.index.year == year]["y"]
            tnorm = (t - t.min()) / (t.max() - t.min()) * 2.0 * np.pi
            axes.plot(tnorm, y, linewidth=0.8, label=year)
            axes.legend()
        plt.savefig(self.plot_folder / "polar.png")
        plt.clf()

    def lag_plot(self,):

        if self.lags % 2 == 0:
            fig, axs = plt.subplots(
                int(self.lags / 2), 2, figsize=(10, 10), tight_layout=True
            )
        else:
            fig, axs = plt.subplots(
                int((self.lags + 1) / 2), 2, figsize=(10, 10), tight_layout=True
            )
        print(self.ts_df.columns)
        for i, axis in enumerate(axs.flat):
            sns.scatterplot(
                x=f"lag_{i + 1}", y="y", data=self.ts_df, ax=axis, hue="month"
            )
            handles, labels = axis.get_legend_handles_labels()
            fig.legend(handles, labels)
            axis.get_legend().remove()
        plt.savefig(self.plot_folder / "lag_plot.png")
        plt.clf()

    def plot_all(self,):

        self.season_plot()
        self.lag_plot()
        self.polar_plot()
        self.pacf()
        self.acf()
        self.timeplot()


#aa = ts_eda(gs.Volume)
#aa.plot_all()
