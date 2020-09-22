import logging
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
module_logger = logging.getLogger(__name__)
module_logger.info('Found sample column in df, stopping function without any changes')
from sklearn.linear_model import LinearRegression
from ds_utils.helper_functions.helper_functions import get_feature_names
from ds_utils.model_selection.split import split, generator
import seaborn as sns
from sklearn.model_selection import cross_validate
import numpy as np
def root_mean_squared_error(y_true, y_pred):

	return np.sqrt(mean_squared_error(y_true, y_pred))

def eval_regression_estimator(estimator, df,dependent_variable, features):

	# check if estimator has predict method

	metrics = dict()

	metric_names = ['mse', 'mae', 'rmse']
	metric_list = [mean_squared_error, mean_absolute_error, root_mean_squared_error]



	for name, metric in zip(metric_names, metric_list):

		metrics[name] = dict()
		
		for set in ['train', 'test']:
		
			y_true = df.loc[df['sample'] == set, dependent_variable]
			
			y_pred = estimator.predict(df.loc[df['sample'] == set, features])
		
			metrics[name][set] = metric(y_true, y_pred)
			
	metrics_df = pd.DataFrame.from_dict({(i,j): metrics[i][j] for i in metrics.keys() for j in metrics[i].keys()}, orient = 'index')

	metrics_df.reset_index(inplace = True)
	dd = metrics_df.reset_index()['index'].apply(pd.Series)
	dd.columns = ['metric', 'sample']
	final = pd.concat([metrics_df, dd] , axis = 1)
	final.drop(columns = ['index'], inplace = True)
	final = pd.melt(final, id_vars = ['metric', 'sample'])
	final.drop(columns = 'variable', inplace = True)
	return final.pivot(columns = 'sample', index = 'metric', values = 'value')
	
   # create df, styling
   
   # create fold df
   
if __name__ == '__main__':

	df = sns.load_dataset('tips')
	
	df = split(df)

	
	model = LinearRegression()
	
	model.fit(df[['size', 'total_bill']], df.tip)
	
	
	print(eval_regression_estimator(model, df[['size', 'total_bill', 'tip', 'sample']] , 'tip', ['size', 'total_bill']))
	
	
    
    
