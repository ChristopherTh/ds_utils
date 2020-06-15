import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path




def residuals(df):
	
	
	g = sns.FacetGrid(df, hue = 'sample', col = 'residual_type', sharex = False, sharey = False, col_wrap = 3)
	g.map(sns.distplot, 'residuals')
	
	plt.savefig("residuals.png")
	plt.clf()
	



     

	
	
	
	

