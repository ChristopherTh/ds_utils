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

	available_datasets = { 	'regression' : ['car_crashes', 'boston_housing', 'tips'],
							'classification' : ['penguins'],
							'NLP': [],
							'image': [],
							'time_series' : ['flights']}

	def __init__(self, test_ratio = None, fold: int = 0):
	
		self.test_ratio = test_ratio
		self.fold = fold
		
	def load_data(self, name):

	
		module_logger.info(f'Attempting to load data {name}')
		
		# seaborn datasets
		if name == 'car_crashes':
			df = sns.load_dataset('car_crashes')
			
		if name == 'tips':
			df = sns.load_dataset('tips')
			
		if name == 'flights':
			df = sns.load_dataset('flights')
		# sklearn datasets
		if name == 'boston_housing':
			data_dict = load_boston()
			
			df_X = pd.DataFrame(data_dict['data'], columns = data_dict['feature_names'])
			df_y = pd.Series(data_dict['target'], name = 'house_price')
			
			df = pd.concat([df_y, df_X ], axis = 1)
			
		# R datasets
		
		# local datasets

		# Misc.
		if name == 'penguins':
			df = pd.read_csv(r"https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/data-raw/penguins_raw.csv")
			df['Species'] = df.Species.str.split().str.get(0)
			
		
		
		module_logger.info(f'Adding sample and fold columns if specified.')
		if self.test_ratio:
			print(self.test_ratio != None)
			df = self.train_test_split(df)
	
		return df
		
	# add function which has a dict which maps available data to description files stored somewhere ?
		
		
	def train_test_split(self, df):
	
		return split(df, 
            train_test = True,
            fold = None,
            random_state = None,
            stratified = False)
            
if __name__ == '__main__':

	test = DataClass()
	df = test.load_data('penguins')
	
	print(df.Species.str.split().str.get(0))
           
            
           
	
		
	
		
		
	
	

	



     

	
	
	
	

