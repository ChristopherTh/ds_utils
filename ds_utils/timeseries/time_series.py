import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import scipy.stats
import statsmodels.api as sm
import matplotlib.dates as mdates
from pandas.plotting import lag_plot

def acf():
	sm.graphics.tsa.plot_acf(ts_df[0], lags=40)
	return
	
def pacf():
	sm.graphics.tsa.plot_pacf(ts_df[0], lags=40)
	return
	
def polar_plot():
	ax = plt.subplot(projection='polar')
	ax.set_theta_direction(-1)
	ax.set_theta_zero_location("N")
	lines, labels = plt.thetagrids((0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330), labels=('Jan', 'Feb', 'Mar', 'Apr', 'Mai', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'), fmt=None)

	for  year in [2012, 2020]:
		times = pd.date_range(ts_df[ts_df.index.year == year].index.min().to_pydatetime(), ts_df[ts_df.index.year == year].index.max().to_pydatetime())
		rand_nums = ts_df[ts_df.index.year == year][0].values
		df = pd.DataFrame(index=times, data=rand_nums, columns=['A'])
		t = mdates.date2num(df.index.to_pydatetime())
		y = df['A']
		tnorm = (t-t.min())/(t.max()-t.min())*2.*np.pi
		#ax.fill_between(tnorm,y ,0, alpha=0.4)
		ax.plot(tnorm,y , linewidth=0.8,  label = year)
		ax.legend('right')
	plt.show()
	return
	
def season_plot():
	sns.lineplot(x = 'month', y = 0, hue = 'year',data = ts_df,ci = False)
	
	def plot_mena(x, **kwargs):
    
    plt.hlines(x.mean(), ts_df.year.min(), ts_df.year.max())
    


	g = sns.FacetGrid(ts_df, col = 'month', sharex = True, sharey = False)
	g.map(sns.lineplot, 'year', 0, ci = False)
	g.map(plot_mena, 0)
	return
	
def timeplot(series):
	fig, axs = plt.subplots()


	axs = sns.lineplot(x = series.index, y = series)
	fig.autofmt_xdate()

	return fig
	


if __name__ == 'main':
	# generate some toy timeseries
	rng = pd.date_range('1/1/2012', periods=3000, freq='d')
	ts = pd.Series(np.random.randint(0, 500, len(rng)), index=rng)
	
	month_map = {1 : 'Jan', 2:'Feb', 3: 'Mar', 4:'Apr', 5:'May', 6:'Jun',7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov', 12:'Dec'}
	week_map = {1 : 'Monday', 2:'Tuesday', 3: 'Wednesday', 4:'Thursday', 5:'Friday', 6:'Saturfay',7:'Sunday'}

	ts_df = pd.DataFrame(ts)
	ts_df['month'] = ts.index.month
	ts_df.month = ts_df.month.map(month_map)
	ts_df['year'] = ts.index.year
	ts_df['week'] = ts.index.week
	ts_df.week = ts_df.week.map(week_map)
	
	def create_lags(df: pd.DataFrame, lags = 1, fill_value = np.nan):
		for i in np.arange(1, lags + 1):
		    ts_df[f'lag_{i }'] = np.append(ts_df[0][(i ):].values, [fill_value] * (i))
		
	create_lags(ts_df, 9)
	
	fig , axs = plt.subplots(3, 3, figsize = (10,10), tight_layout = True)
	for i, axis in enumerate(axs.flat):
		sns.scatterplot(x = f'lag_{i + 1 }', y = 0, data = ts_df, ax = axis, hue = 'month')
		handles, labels = axis.get_legend_handles_labels()
		fig.legend(handles,labels,loc = 'right')
		axis.get_legend().remove()


