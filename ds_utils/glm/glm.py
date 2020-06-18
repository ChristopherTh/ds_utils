from pathlib import Path
from datetime import datetime
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from scipy import stats
import webbrowser
from ds_utils.model_selection.split import split
from ds_utils.helper_functions.helper_functions import get_feature_names
import seaborn as sns
from patsy import dmatrices
import pandas as pd
from ds_utils.plotting.residuals import residuals
from statsmodels.genmod.families import links, family, varfuncs
import numpy as np
import scipy.stats
from statsmodels.graphics.regressionplots import plot_partregress_grid
df = sns.load_dataset("car_crashes")
a, b = dmatrices('total ~ speeding + alcohol', df , return_type = 'dataframe')
df = pd.concat([a, b] , axis = 1)

df = split(df,random_state =3)
def glm_func(dependent_variable, data, glm_type):

	# scripting
	file_path = Path.cwd()

	exp = file_path.joinpath("experiments")

	if not exp.exists():
		exp.mkdir()
		
	
	
	new_folder = exp.joinpath(datetime.today().strftime("%d.%m.%Y um %H:%M:%S Uhr"))

	new_folder.mkdir()

	# selecting model
	if glm_type == 'gamma':
		fam = family.Gamma(links.log())
		var_func = lambda mu: mu ** 2
	elif glm_type == 'poisson':
		fam = sm.families.Poisson(sm.families.links.log)
	elif glm_type == 'gaussian':
		fam = sm.families.Gaussian()
		
	features = get_feature_names(dependent_variable, data)
	train = df['sample'] == 'train'
	test = df['sample'] == 'test'


	model = sm.GLM(data.loc[train, dependent_variable], data.loc[train, features], family=fam).fit()

	with open(new_folder.joinpath("Output.txt"), "w") as output:
		output.write(str(model.summary()))
		
	df_test = df[[dependent_variable, 'sample']].copy()
	df_test['nu'] = fam.link(model.predict(data[features])) # explicit with link -> no case destinction for different predict implmentations
	df_test['pred'] = model.predict(data[features])
	df_test['raw_residuals'] = df_test[dependent_variable] - df_test['pred']
	df_test['pearson_residuals'] = (df[dependent_variable] - df_test['pred']) / np.sqrt(var_func(df_test['pred']))
	aaa = stats.gamma.cdf(df[dependent_variable], df_test['pred'] / model.scale, scale = model.scale)
	df_test['quantile_residuals'] = stats.norm.ppf(aaa, loc=0, scale=1)
	df_test['quantile_residuals'].clip(-10, 10, inplace = True)
	df_test['deviance_residuals'] = fam._resid_dev(df[dependent_variable], df_test['pred'])
	df_test['anscombe'] = fam.resid_anscombe(df[dependent_variable], df_test['pred'])
	df_test['working_responses'] = model.predict(data.loc[:, features], linear = True) + fam.link.deriv(df_test['raw_residuals'])
	df_test['working_residual'] = df_test['working_responses'] - df_test['nu']
	
	
	




	



	
	resid_df = pd.melt(df_test, var_name = 'residual_type',value_name = 'residuals' ,id_vars = [dependent_variable, 'sample', 'pred'] ,value_vars = ['quantile_residuals','deviance_residuals', 'pearson_residuals', 'anscombe'])

	residuals(resid_df)
	
	gg = sns.FacetGrid(resid_df, col = 'residual_type', hue = 'sample', sharey = True, sharex = False)
	gg.map(plt.scatter,  'residuals','pred', alpha = .7)
	gg.add_legend()
	plt.show()

	fig = plt.figure(figsize=(8, 6))
	
	plot_partregress_grid(model, fig=fig)
	plt.savefig("test2.png")
	

	


glm_func('total', data = df , glm_type = 'gamma')

stats.norm.ppf(0.975, loc= 0, scale = 1)


