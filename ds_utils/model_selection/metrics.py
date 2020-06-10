import logging
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
module_logger = logging.getLogger(__name__)
module_logger.info('Found sample column in df, stopping function without any changes')

from ds_utils.helper_functions.helper_functions import get_feature_names

def eval_regression_estimator(estimator, df,dependent_variable, type = 'regression'):

	# check if estimator has predict method

	metrics = dict()

	metric_names = ['mse', 'mae']
	metric_list = [mean_squared_error, mean_absolute_error]
	
	features = get_feature_names(dependent_variable, df)

	for name, metric in zip(metric_names, metric_list):

		metrics[name] = dict()
		
		for set in ['train', 'test']:
		
			y_true = df.loc[df['sample'] == set, dependent_variable]
			
			y_pred = estimator.predict(df.loc[df['sample'] == set, features])
		
			metrics[name][set] = metric(y_true, y_pred)
    	    
	metrics_df = pd.DataFrame.from_dict({(i,j): metrics[i][j] for i in metrics.keys() for j in metrics[i].keys()}, orient = 'index')
    	    
	return metrics_df
	

	

   # create df, styling
   
   # create fold df
    
    
