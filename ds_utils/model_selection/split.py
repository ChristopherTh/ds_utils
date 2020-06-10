import logging
from sklearn.model_selection import KFold, ShuffleSplit, StratifiedShuffleSplit
import pandas as pd
import seaborn as sns
df = sns.load_dataset('tips')
module_logger = logging.getLogger(__name__)

def split(  df,
            train_test = True,
            fold = False,
            random_state = None,
            stratified = False):

    
    if 'sample' in df:
        module_logger.info('Found sample column in df, stopping function without any changes to df')
        return

    if 'fold' in df:
        module_logger.info('Found fold column in df, stopping function without any changes to df')
        return

    train_idx = ShuffleSplit(n_splits=1, test_size = 0.2, random_state = random_state)
    train, test = next(train_idx.split(df))
        

    df.loc[train, 'sample'] = 'train'
    df.loc[test, 'sample'] = 'test'
    
    df.sort_values('sample', ascending = False, inplace = True)
    df.reset_index(inplace = True, drop = True)
    


    kf = KFold(n_splits=5, shuffle = False, random_state = random_state)

    kfolds = kf.split(df.loc[df['sample'] == 'train'])

    for i, (train, test) in enumerate(kfolds):

        df.loc[test, 'fold'] = int(i)

    df.loc[df['sample'] == 'test', 'fold'] = -1
    
    

    return df
    



def generator(df):

	tt = list(df.fold.unique())
	
	tt.remove(-1)
	
	for i in tt:
	
		train_idx = df[df['fold'] != i].index
		
		test_idx = df[df['fold'] == i].index
		
		yield train_idx, test_idx

	
	
	
	
	
