import logging
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedShuffleSplit
import pandas as pd
from sklearn.model_selection import ShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, median_absolute_error, mean_squared_log_error
from sklearn.metrics import mean_absolute_error
from sklearn.dummy import DummyRegressor
import seaborn as sns
module_logger = logging.getLogger(__name__)
import matplotlib.pyplot as plt

module_logger.info('Found sample column in df, stopping function without any changes to')
df = sns.load_dataset('tips')


def dummy_regressor(df, dependent_variable, splits = 30,  glm_type = None):


	splitter = ShuffleSplit(n_splits = splits, test_size = 0.2)
	
	eval_results = dict()
	metric_names = ['mse', 'mean ae', 'median ae', 'mean s log e']
	metric_list = [mean_squared_error, mean_absolute_error, median_absolute_error, mean_squared_log_error]
	strategy_list = ['mean', 'median']
	for name, metric in zip(metric_names, metric_list):
	
		eval_results[name] = dict()
		
		
		for strat in strategy_list:
		
			eval_results[name][strat] = dict()
		
			eval_results[name][strat]['train'] = []
			eval_results[name][strat]['test'] = []
		
			for train_idx, test_idx in splitter.split(df, df[dependent_variable]):
		


				estimator = DummyRegressor(strategy = strat)
				estimator.fit(df.loc[train_idx], df.loc[train_idx, dependent_variable])

				

				train_error = metric(df.loc[train_idx, dependent_variable],estimator.predict(df.loc[train_idx, dependent_variable]))
				test_error = metric(df.loc[test_idx, dependent_variable],estimator.predict(df.loc[test_idx, dependent_variable]))
				
				eval_results[name][strat]['train'].append(train_error)
				eval_results[name][strat]['test'].append(test_error)
			
	eval_results_df = pd.DataFrame.from_dict({(i,j, k): eval_results[i][j][k] for i in eval_results.keys() for j in eval_results[i].keys() for k in eval_results[i][j]}, orient = 'index')
	eval_results_df.reset_index(inplace = True)
	dd = eval_results_df.reset_index()['index'].apply(pd.Series)
	dd.columns = ['metric', 'strategy', 'sample']
	final = pd.concat([eval_results_df, dd] , axis = 1)
	final.drop(columns = 'index', inplace = True)
	final = pd.melt(final, id_vars = ['metric', 'strategy', 'sample'])

	g = sns.FacetGrid(final, col = 'metric', row = 'strategy', hue = 'sample', sharex = False, sharey = False, legend_out = True)
	g.map(sns.distplot, 'value', kde = False)
	g.add_legend()
	plt.show()
dummy_regressor(df, 'tip', 100)
