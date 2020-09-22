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
import streamlit as st
from statsmodels.graphics.regressionplots import plot_partregress_grid
from ds_utils.model_selection.metrics import eval_regression_estimator
from itertools import product
from sklearn.inspection import plot_partial_dependence
import numpy as np

# source: http://davmre.github.io/blog/python/2013/12/15/orthogonal_poly
def ortho_poly_fit(x, degree = 1):
    n = degree + 1
    x = np.asarray(x).flatten()
    if(degree >= len(np.unique(x))):
            stop("'degree' must be less than number of unique points")
    xbar = np.mean(x)
    x = x - xbar
    X = np.fliplr(np.vander(x, n))
    q,r = np.linalg.qr(X)

    z = np.diag(np.diag(r))
    raw = np.dot(q, z)

    norm2 = np.sum(raw**2, axis=0)
    alpha = (np.sum((raw**2)*np.reshape(x,(-1,1)), axis=0)/norm2 + xbar)[:degree]
    Z = raw / np.sqrt(norm2)
    return Z, norm2, alpha

def ortho_poly_predict(x, alpha, norm2, degree = 1):
    x = np.asarray(x).flatten()
    n = degree + 1
    Z = np.empty((len(x), n))
    Z[:,0] = 1
    if degree > 0:
        Z[:, 1] = x - alpha[0]
    if degree > 1:
      for i in np.arange(1,degree):
          Z[:, i+1] = (x - alpha[i]) * Z[:, i] - (norm2[i] / norm2[i-1]) * Z[:, i-1]
    Z /= np.sqrt(norm2)
    return Z

st.title("GLM")


st.write("## Load Data")

df = pd.read_csv("boston_housing.csv")




#df.drop(columns = "Unnamed: 0", inplace = True)
df_cols = df.columns.to_list()
st.dataframe(df.head())

st.write("## Select variables")


if 'sample' in df_cols:
	st.info("Detected sample column in the dataframe")
else:
	st.warning("Didnt found fold column.")
	
if 'fold' in df_cols:
	st.info("Detected fold column in the dataframe")
else:
	st.warning("Didnt found fold column.")
	
if 'offset' in df_cols:
	st.info("Detected offset column in the dataframe")
else:
	st.warning("Didnt found offset column.")
	
if 'sample_weight' in df_cols:
	st.info("Detected sample_weight column in the dataframe")
else:
	st.warning("Didnt found sample_weight column.")
	
dependent_variable = st.selectbox("Select the dependent variable", [x for x in df_cols if x not in ['sample', 'offset', 'fold']])

features = st.multiselect("Select the explanatory variables for the further analysis", [x for x in df_cols if x not in ['sample', 'offset', dependent_variable, 'fold']])


	
if st.checkbox("Add interaction terms?"):
	
	inter_cols = [col for col in features if df[col].value_counts().index.isin([0,1]).all()]
	
	inter_cols = st.multiselect("Choose binary columns to build interactions from", inter_cols)
	
	inter_cols = st.multiselect("Choose the interactions you want to include", list(product(inter_cols, [col for col in features if col not in inter_cols])))
	
	for interactions in inter_cols:
		
		a, b = interactions
		
		df[f"int: {a, b}"] = df[a] * df[b]
		
		features = features + [f"int: {a, b}"]
		
if st.checkbox("Add non linearities"):
	
	num_cols = st.multiselect("asd", df[features].astype('float64').columns.to_list())
	
	degrees = []
	for col in num_cols:
	
		deg = st.selectbox(f"Please select the degree for {col}", [0, 1, 2 ,3],key = col)
		
		Z, norm2, alpha =  ortho_poly_fit(df[col], degree = deg)
		
		#df.drop(columns = col, inplace = True)
		
		df_pol = pd.DataFrame(Z[:, 1:], columns = [col + str(x + 1) for x in range(deg )])
		
		df = pd.concat([df, df_pol], axis = 1)
		
		features = features + df_pol.columns.to_list()
		features.remove(col)
		
	st.dataframe(df)
		
	
	
		
	
	
if st.checkbox("Add constant ?"):

	df["constant"] = 1
	features = features + ["constant"]

#if st.checkbox("Include interaction terms"):



	#inter1 = st.selectbox("Select the dependent variable", df_cols)
	
	

st.write("## Select Distribution of dependen variable and link function")

glm_type = st.selectbox("Here select distribution of the dependen variable", ['gamma', 'poisson', 'gaussian', 'binomial'])

link_function = st.selectbox("Here select the link function", ['identity', 'log', 'logit'])

# select link
if link_function == 'identity':
	link = sm.families.links.identity
elif link_function == 'log':
	link = sm.families.links.log
elif link_function == 'logit':
	link = sm.families.links.logit

# selecting model
if glm_type == 'gamma':
	fam = family.Gamma(link)
	var_func = lambda mu: mu ** 2
elif glm_type == 'poisson':
	fam = sm.families.Poisson(link)
elif glm_type == 'gaussian':
	fam = sm.families.Gaussian()
	var_func = lambda mu: mu
train = df['sample'] == 'train'
test = df['sample'] == 'test'




model = sm.GLM(df.loc[train, dependent_variable], df.loc[train, features], family=fam).fit()



st.text(str(model.summary()))


df_test = df[[dependent_variable, 'sample']].copy()
df_test['nu'] = fam.link(model.predict(df[features])) # explicit with link -> no case destinction for different predict implmentations
df_test['pred'] = model.predict(df[features])
df_test['raw_residuals'] = df_test[dependent_variable] - df_test['pred']
df_test['pearson_residuals'] = (df[dependent_variable] - df_test['pred']) / np.sqrt(var_func(df_test['pred']))
aaa = stats.gamma.cdf(df[dependent_variable], df_test['pred'] / model.scale, scale = model.scale)
df_test['quantile_residuals'] = stats.norm.ppf(aaa, loc=0, scale=1)
df_test['quantile_residuals'].clip(-10, 10, inplace = True)
df_test['deviance_residuals'] = fam._resid_dev(df[dependent_variable], df_test['pred'])
df_test['anscombe'] = fam.resid_anscombe(df[dependent_variable], df_test['pred'])
df_test['working_responses'] = model.predict(df.loc[:, features], linear = True) + fam.link.deriv(df_test['raw_residuals'])
df_test['working_residual'] = df_test['working_responses'] - df_test['nu']
resid_df = pd.melt(df_test, var_name = 'residual_type',value_name = 'residuals' ,id_vars = [dependent_variable, 'sample', 'pred'] ,value_vars = ['quantile_residuals','deviance_residuals', 'pearson_residuals', 'anscombe'])


st.write("Predictions against Residuals")
res_plot = st.selectbox("Select one of the available residual types", ['quantile_residuals','deviance_residuals', 'pearson_residuals', 'anscombe'])
fig, axes = plt.subplots(2, 1, tight_layout = True)

sns.scatterplot(y = 'pred', x = res_plot, data = df_test.loc[train], ax = axes[0])
sns.scatterplot(y = 'pred', x = res_plot, data = df_test.loc[test], ax = axes[1])

st.pyplot(fig)

st.write("## Explanatory variables against Residuals")
res_ex = st.selectbox("Select one of the available residual types", ['quantile_residuals','deviance_residuals', 'pearson_residuals', 'anscombe'], key = 'ex_res1')
ex_res = st.selectbox("Select one of the available explanatory variables", features, key = 'ex_res')
fig, axes = plt.subplots(2, 1, tight_layout = True)
axes[0].scatter(x = df_test.loc[train, res_ex], y = df.loc[train, ex_res])
axes[1].scatter(x = df_test.loc[test, res_ex], y = df.loc[test, ex_res])
st.pyplot(fig)


st.write("## QQ Plots of residuals")

fig, axes = plt.subplots(2,2,tight_layout = True)

from statsmodels import graphics
res_ex = st.selectbox("Select one of the available residual types", ['quantile_residuals','deviance_residuals', 'pearson_residuals', 'anscombe'], key = 'qq plot')
sns.distplot(df_test.loc[train, res_ex], ax = axes[0,0])
sns.distplot(df_test.loc[test, res_ex], ax = axes[1,0])
graphics.gofplots.qqplot(df_test.loc[train, res_ex], line='r', ax = axes[0,1])
graphics.gofplots.qqplot(df_test.loc[test, res_ex], line='r', ax = axes[1,1])
st.pyplot(fig)
st.write("## Predictions against actuals")
fig, axes = plt.subplots(2, 1, tight_layout = True)

pred_max = max(df[dependent_variable])
pred_min = min(df[dependent_variable])

axes[0].scatter(x = df_test.loc[train, 'pred'], y = df.loc[train, dependent_variable])
axes[0].set_xlabel("pred")
axes[0].set_ylabel("actuals")
axes[0].set_title("train")
axes[1].scatter(x = df_test.loc[test, 'pred'], y = df.loc[test, dependent_variable])
axes[1].set_xlabel("pred")
axes[1].set_ylabel("actuals")
axes[1].set_title("test")
axes[1].plot([pred_min, pred_max], [pred_min, pred_max])
axes[0].plot([pred_min, pred_max], [pred_min, pred_max])
st.pyplot(fig)

st.write("## Influence Plots")

#fig , axes = plt.subplots()

#fig = sm.graphics.influence_plot(model, ax = axes, criterion = 'cooks')

#st.pyplot(fig)

st.write("## Interpretation plots")


#fig = plt.figure(figsize=(8, 6))
	
#plot_partregress_grid(model, fig=fig)

#st.pyplot(fig)

st.write(features)
st.write("## Metrics")
res = eval_regression_estimator(model, df, dependent_variable, features)

st.dataframe(res)





    



