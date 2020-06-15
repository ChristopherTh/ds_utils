import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from itertools import combinations

def pva(df, dependent_variable)
	
	pred_max = max(df.dependent_variable)
	pred_min = min(df.dependent_variable)
	g = sns.FacetGrid(df, col = 'sample')
	g.map(sns.scatterplot, dependent_variable, f'{dependent_variable}_pred')
	for ax in g.axes[0]:
		ax.plot([pred_min, pred_max], [pred_min, pred_max])
	plt.savefig("pva}.png")
	plt.clf()

     

	
	
	
	

