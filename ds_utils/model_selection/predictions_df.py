import logging
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedShuffleSplit
import pandas as pd
import seaborn as sns

module_logger = logging.getLogger(__name__)




def predictions(model, df):

	pred_train = model.fit(df[dependent_variable], df[features])



if __name__ == '__main__':
	df = sns.load_dataset('tips')
	print(split(df))
	
	
	
	
