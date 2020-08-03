import pandas as pd
import numpy as np
import datetime as dt
from sklearn import linear_model
from sklearn import metrics
from sklearn.decomposition import PCA
import logging


# The list of instruments we want to use
inst_list =['SPX','Oil','10Y_US','EUR','XAU','JPY','VIX']

# List of currencies
fx_list = ['EUR','JPY']

# The mapping between instruments and their yf names
names_dict = {'SPX':'ES=F',
              'EUR':'EUR=X',
              'XAU':'GLD',
              'JPY':'JPY=X',
              'Oil':'USO',
              '10Y_US':'^TNX',
              'VIX':'^VIX'}

# Params for the yf api
yf_list = [names_dict[x] for x in inst_list]
start = '2006-05-01'
end = '2020-05-15'
interval = '1d'

# List of absolute and relative instruments, for calculation of returns
abs_list = ['10Y_US','VIX']
rel_list = [x for x in inst_list if x not in abs_list]

# Clustering method (either 'K-means' or 'Agglo')
clus_method = 'Agglo'
clus_n = 3

# For linear regression intra-cluster prediction
exo = ['SPX','10Y_US','JPY']+['cat_'+str(x) for x in np.arange(clus_n)]
endo = ['Oil','VIX']

# Directory for saving plots
directory = 'DIRECTORY'

# Set random seed for clustering initialisation
np.random.seed(0)

def main():

	logging.basicConfig(format='%(levelname)s:%(message)s',level=logging.INFO)

	# Load market data
	logging.info('Loading market data')
	import yf_load as md

	yf_data = md.load_data(yf_list,start,end,interval)
	logging.info('yf done')
	

	df = md.get_close_data(yf_data)
	df = md.invert_fx(fx_list,names_dict,df)
	df = md.rename_columns(df, inst_list, names_dict)
	logging.info('yf transforms done')

	# Save the raw df down
	df.to_csv(directory+'md.csv',index=True)

	# Calculate returns
	abs_df = df[abs_list].diff(1)
	pct_df = df[rel_list].pct_change(1)
	logging.info('returns calc done')

	# Merge df
	data = pct_df.join(abs_df,on='Date',how='left')
	logging.info('merge returns to df done')

	# Drop NaNs
	data.dropna(inplace=True)

	# Run clustering:
	logging.info('Running clustering')
	import clustering as cl

	clus = cl.generate_cluster_labels(data, clus_method, clus_n)

	# Write clusters to dataframe
	logging.info('Clusters written to df')
	data['cluster']= clus.labels_

	# Save cluster plots
	logging.info('Save cluster plot')
	import saveplots as sv
	sv.save_clusters(data, clus_n, directory=directory)

	# Save regime plot
	logging.info('Save regime plot')
	sv.save_regimes(data,clus_n, directory=directory)

	# Prepare data for stats models
	cluster_dummies = pd.get_dummies(data.cluster,prefix='cat')
	data_enc = pd.concat([data,cluster_dummies],axis=1)

	# output to csv
	logging.info('Save processed data frame')
	data_enc.to_csv(directory+'df_out.csv')

	# Run PCA
	import stats as st
	st.run_PCA(data_enc, inst_list, directory)

	st.run_linear_regression(data_enc, exo, endo)
	
	return

if __name__ == "__main__":
    # execute only if run as a script
    main()

