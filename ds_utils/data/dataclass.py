import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from ds_utils.model_selection.split import split
import logging
module_logger = logging.getLogger(__name__)


class DataClass(object):

	def __init__(self, test_ratio, fold: int = 0):
	
		self.test_ratio = test_ratio
		self.fold = fold
		
	def load_data(self, name):
	
		module_logger.info(f'Attempting to load data {name}')
		
		if name == 'car_crashes':
			df = sns.load_dataset('car_crashes')


		if name == 'penguins':
			df = pd.read_csv(r"https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/data-raw/penguins_raw.csv")
		
		module_logger.info(f'Adding sample and fold columns if specified.')
		df= self.train_test_split(df)
	
		return df
		
		
	def train_test_split(self, df):
	
		return split(df, 
            train_test = True,
            fold = None,
            random_state = None,
            stratified = False)
            
           
            
           
	
		
	
		
		
	
	

	



     

	
	
	
	

