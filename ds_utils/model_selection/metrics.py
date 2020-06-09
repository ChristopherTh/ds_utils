import logging
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedShuffleSplit
import pandas as pd

module_logger = logging.getLogger(__name__)


module_logger.info('Found sample column in df, stopping function without any changes 


def eval_regression_estimator(estimator, df, type = 'regression'):

    # check if estimator has predict method
    
    metrics = dict()
    
    metrics_list = ['mse', 'mae']
    
    for metric in metrics list:
    
    	metrics[metric] = dict{}
    	
    	for set in ['train', 'test]:
    	
    	    y_train = df.loc[df['sample'] == set, dependent_variable]
    	    
    	    y_pred = estimator.predict(df.loc[df['sample'] == set, features]
    	
    	    metrics[metric][set] = mse(y_true, y_pred)
    	    
    	 
   # create df, styling
   
   # create fold df
    
    
