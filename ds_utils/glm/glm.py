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



	y_pred_test = model.predict(df.loc[test, features])
	y_pred_train = model.predict(df.loc[train, features])

	y_train = df.loc[train, dependent_variable]
	y_test = df.loc[test, dependent_variable]
	
	
	# pearson residuals
	pearson_resid_train = (y_train - y_pred_train) / np.sqrt(var_func(y_pred_train))
	pearson_resid_test = (y_test - y_pred_test) / np.sqrt(var_func(y_pred_test))
	
	# quantile residuals
	bbb = stats.gamma.cdf(y_test, y_pred_test / model.scale, scale = model.scale)
	quantile_resid_test = stats.norm.ppf(bbb, loc=0, scale=1)

	
	
	aaa = stats.gamma.cdf(y_train, y_pred_train / model.scale, scale = model.scale)
	quantile_resid_train = stats.norm.ppf(aaa, loc=0, scale=1)
	quantile_resid_train[quantile_resid_train == np.inf] = 0
	print(quantile_resid_train, quantile_resid_test)

	resid_df_train = pd.DataFrame({'deviance' : fam._resid_dev(y_train, y_pred_train),'resid_pearson': pearson_resid_train, 'resid_anscombe' : fam.resid_anscombe(y_train, y_pred_train), 'sample': 'train', 'quantile_resid' : quantile_resid_train})
	
	resid_df_test = pd.DataFrame({'deviance' : fam._resid_dev(y_test, y_pred_test),'resid_pearson': pearson_resid_test, 'resid_anscombe' : fam.resid_anscombe(y_test, y_pred_test), 'sample': 'test', 'quantile_resid' : quantile_resid_test})
	
	resid_df = pd.concat([resid_df_train, resid_df_test])
	
	resid_df = pd.melt(resid_df, var_name = 'residual_type',value_name = 'residuals' ,id_vars = 'sample' ,value_vars = ['quantile_resid','deviance', 'resid_pearson', 'resid_anscombe'])

	residuals(resid_df)
	

	


glm_func('total', data = df , glm_type = 'gamma')

stats.norm.ppf(0.975, loc= 0, scale = 1)


