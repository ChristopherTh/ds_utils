import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import umap
from sklearn.preprocessing import StandardScaler
from ds_utils.data.dataclass import DataClass
from mpl_toolkits.mplot3d import Axes3D
from ds_utils.helper_functions.helper_functions import get_feature_names
import umap.plot
def pva(df, dependent_variable):
	
	pred_max = max(df.dependent_variable)
	pred_min = min(df.dependent_variable)
	g = sns.FacetGrid(df, col = 'sample')
	g.map(sns.scatterplot, dependent_variable, f'{dependent_variable}_pred')
	for ax in g.axes[0]:
		ax.plot([pred_min, pred_max], [pred_min, pred_max])
	plt.savefig("pva}.png")
	plt.clf()
	
# standadize ?
def plot_umap(df, dependent_variable = None):

	df.select_dtypes('float64').apply(lambda x: (x - x.mean()) / x.std(), axis = 1)

	if dependent_variable is not None:
		features = get_feature_names(dependent_variable, df)
		
		
		
		if str(df[dependent_variable].dtype) == 'float64':
			umap.plot.points(reducer, values = df[dependent_variable], theme='fire')
			reducer = umap.UMAP(n_components=2)
			reducer.fit(df, y = df[dependent_variable])
			embedding = reducer.transform(df)
			
		elif str(df[dependent_variable].dtype) == 'category':
			reducer = umap.UMAP(n_components=2)
			reducer.fit(df[features], y = df[dependent_variable].cat.codes)
			embedding = reducer.transform(df[features])
			umap.plot.points(reducer, labels = df[dependent_variable], theme='fire')
			
			
			hover_data = pd.DataFrame({'index':df.index,
                           'label':df[dependent_variable]})
			#hover_data['item'] = hover_data.label


			p = umap.plot.interactive(reducer, labels=df[dependent_variable], hover_data=hover_data, point_size=5)
			umap.plot.show(p)

		# binary class case labeled 0, 1	
		elif (str(df[dependent_variable].dtype) == 'int64') and (df[dependent_variable].unique().size == 2):
			umap.plot.points(reducer, labels = df[dependent_variable], theme='fire')
		
		
	else:
		features = df.columns.to_list()

		
		reducer = umap.UMAP()
		reducer.fit(df)
		embedding = reducer.transform(df)
		umap.plot.points(reducer, theme='fire')
		
	# various plots
	
	umap.plot.diagnostic(reducer, diagnostic_type='all')
	plt.savefig("diagnostics.png")
				
	# expensive to compute
	if df.shape[0] < 5000:
		umap.plot.connectivity(reducer, edge_bundling='hammer')


	#umap.plot.diagnostic(reducer, diagnostic_type='neighborhood')
	#plt.savefig("diagnostic_neigborhood.png")
	
	umap.plot.connectivity(reducer, show_points=True)
	plt.savefig("connectivity.png")
if __name__ == '__main__':




	#df = DataClass().load_data('tips')
	#df = df.dropna()
	
	#metr = [('num','euclidean' ,df.select_dtypes('float64').columns.to_list()), ('cat', 'dice',df.select_dtypes('category').columns.to_list())]
	
	#ll = umap.umap_.DataFrameUMAP(metr)
	
	#ll.fit(df)

	
	
	df = pd.DataFrame(np.random.randn(100, 4))
	df['aa'] = ['apple', 'b', 'c', 'd', 'e'] * 20
	#df['aa'] = [1,2,3,4,5] * 20
	df.aa = df.aa.astype('category')
	#plot_umap(df, 'aa')
	
	#print(pd.Series([1,0,0,1]).unique().size)

	import scipy.sparse
	#aa =  umap.UMAP().fit(np.random.randn(100,4))
	#aa.graph_
	numeric_data = np.random.randn(10,4)
	categorical_data = np.array([[1,0], [1,0],[0,0], [1,1],[1,0], [1,1],[1,0], [0,1],[1,0], [1,0]])
	fit1 = umap.UMAP().fit(numeric_data)
	fit2 = umap.UMAP(metric='dice').fit(categorical_data)
	prod_graph = fit1.graph_.multiply(fit2.graph_)
	new_graph = 0.99 * prod_graph + 0.01 * (fit1.graph_ + fit2.graph_ - prod_graph)
	embedding = umap.umap_.simplicial_set_embedding(scipy.sparse.csgraph(new_graph), fit1.n_components, fit1._initial_alpha, fit1._a, fit1._b, fit1.repulsion_strength, fit1.negative_sample_rate, 200, fit1.init, np.random, False, 'euclidean', {})
	plt.show()

	

     

	
	
	
	

