import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from ds_utils.model_selection.split import split
from sklearn.datasets import load_boston
import logging
module_logger = logging.getLogger(__name__)


class DataClass(object):

	available_datasets = ['car_crashes', 'boston_housing', 'penguins']

	def __init__(self, test_ratio, fold: int = 0):
	
		self.test_ratio = test_ratio
		self.fold = fold
		
	def load_data(self, name):
		print(name)
	
		module_logger.info(f'Attempting to load data {name}')
		
		# seaborn datasets
		if name == 'car_crashes':
			df = sns.load_dataset('car_crashes')
		
		# sklearn datasets
		if name == 'boston_housing':
			data_dict = load_boston()
			
			df_X = pd.DataFrame(data_dict['data'], columns = data_dict['feature_names'])
			df_y = pd.Series(data_dict['target'], name = 'house_price')
			
			df = pd.concat([df_y, df_X ], axis = 1)
			
		# R datasets

		# Misc.
		if name == 'penguins':
			df = pd.read_csv(r"https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/data-raw/penguins_raw.csv")
			
		
		
		module_logger.info(f'Adding sample and fold columns if specified.')
		df= self.train_test_split(df)
	
		return df
		
	# add function which has a dict which maps available data to description files stored somewhere ?
		
		
	def train_test_split(self, df):
	
		return split(df, 
            train_test = True,
            fold = None,
            random_state = None,
            stratified = False)
            
if __name__ == '__main__':

	test = DataClass(.4)
	df = test.load_data('boston_housing')
	
	print(test.available_datasets)
           
            
           
	
		
	
		
		
	
	

	



     

	
	
	
	

