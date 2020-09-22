import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from ds_utils.model_selection.split import split
from sklearn.datasets import load_boston
import logging
import statsmodels.api as sm
import pyreadr
import streamlit as st
from ds_utils.data.dataclass import DataClass

st.title("Dataclass")

aa = DataClass()


data_type = st.selectbox("What type of application are you looking for ?", [x for x in aa.available_datasets.keys()])

chosen_data = st.selectbox("The available atasets are", aa.available_datasets[data_type])

if st.checkbox("Add a train/test split to the data as an extra column."):
	
	test_ratio = st.selectbox('ratio', [x / 10 for x in range(0, 10)])
else:
	test_ratio = None
	

if st.checkbox("Add a cross validation split"):

	fold = st.selectbox('select fold size', [x for x in range(0, 10)])
else:
	fold = 0
	


df = aa.load_data(chosen_data, test_ratio = test_ratio, fold = fold)
	
st.dataframe(df.astype('object').head())


if aa.doc != 'empty':

	if st.checkbox("Show documentation of data"):
		st.write(aa.doc)


            


	
		
	
		
		
	
	

	



     

	
	
	
	

