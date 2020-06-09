import logging
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedShuffleSplit
import pandas as pd

module_logger = logging.getLogger(__name__)


module_logger.info('Found sample column in df, stopping function without any changes 


def get_feature_names(dependent_variable, df):

    feature_names = df.columns.to_list()
    
    feature_names.remove(dependent_variable)
    
    non_feature = ['fold', 'offset', 'fold', 'sample_weight', 'sample']
    
    for i in non_feature:
    
        if i in feature_names:
            feature_names.remove(i)
            module_logger.info(f'Removing {i} for further analysis.')
        else:
            module_logger.info(f'Didnt find {i} in df, hence not removing it.')
            
    return feature_names
            
    
    
        
    
    
    
