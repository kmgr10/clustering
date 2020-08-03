from sklearn import preprocessing
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn import metrics
import pandas as pd

def run_PCA(data_enc, inst_list, directory):

	# Need to make this more pythonic
	# Here we assume there are 3 clusters and show the first component for each

	scaler_pca = preprocessing.StandardScaler()
	scaler_pca.fit(data_enc[inst_list])
	cluster_0_scaled = scaler_pca.transform(data_enc[data_enc.cluster==0][inst_list])
	cluster_1_scaled = scaler_pca.transform(data_enc[data_enc.cluster==1][inst_list])
	cluster_2_scaled = scaler_pca.transform(data_enc[data_enc.cluster==2][inst_list])
	#scaler_0 = preprocessing.StandardScaler()
	#scaler_1 = preprocessing.StandardScaler()
	#scaler_2 = preprocessing.StandardScaler(with_mean=False)
	#cluster_0_scaled = scaler_0.fit_transform(data_enc[data_enc.cluster==0][inst_list])
	#cluster_1_scaled = scaler_1.fit_transform(data_enc[data_enc.cluster==1][inst_list])
	#cluster_2_scaled = scaler_2.fit_transform(data_enc[data_enc.cluster==2][inst_list])

	# output to csv
	#data_enc[data_enc.cluster==2][inst_list].to_csv(directory+'cluster_2_scaled_out.csv')
	#data_enc[data_enc.cluster==1][inst_list].to_csv(directory+'cluster_1_scaled_out.csv')
	#pd.DataFrame(cluster_1_scaled).to_csv(directory+'scaled_1.csv')


	pca_0 = PCA(n_components=3).fit(cluster_0_scaled)
	pca_1 = PCA(n_components=3).fit(cluster_1_scaled)
	pca_2 = PCA(n_components=3).fit(cluster_2_scaled)

	print('Variance explained pca_0: ' + str(pca_0.explained_variance_/sum(pca_0.explained_variance_)))
	print('Variance explained pca_1: ' + str(pca_1.explained_variance_/sum(pca_1.explained_variance_)))
	print('Variance explained pca_2: ' + str(pca_2.explained_variance_/sum(pca_2.explained_variance_)))

	# Plot and save PCA 1
	plt.figure(figsize=(14,6))

	tick_range = np.arange(len(inst_list))
	plt.plot(pca_0.components_[0],label='Cluster_0,PCA_1')
	plt.plot(pca_1.components_[0],label='Cluster_1,PCA_1')
	plt.plot(pca_2.components_[0],label='Cluster_2,PCA_1')
	plt.xticks(ticks=tick_range,labels=inst_list)
	plt.legend()
	plt.title('Cluster PCA decomposition')
	plt.ylabel('Component std scaled move')
	plt.savefig(directory+'PCA_1.png')

	# Plot and save PCA 2
	plt.figure(figsize=(14,6))

	tick_range = np.arange(len(inst_list))
	plt.plot(pca_0.components_[1],label='Cluster_0,PCA_2')
	plt.plot(pca_1.components_[1],label='Cluster_1,PCA_2')
	plt.plot(pca_2.components_[1],label='Cluster_2,PCA_2')
	plt.xticks(ticks=tick_range,labels=inst_list)
	plt.legend()
	plt.title('Cluster PCA decomposition')
	plt.ylabel('Component std scaled move')
	plt.savefig(directory+'PCA_2.png')

	return

def run_linear_regression(data_enc, exo, endo):
	''' Fit a multivariate regression model to clusters based on exogenous variables stored in list exo
	And endogenous variables stored in list endo '''

	reg = linear_model.LinearRegression()

	X = data_enc[exo]
	y = data_enc[endo]

	reg.fit(X,y)
	r2 = metrics.r2_score(y,reg.predict(X),multioutput='variance_weighted')
	print(f'R^2 is {r2}')

	# Some examples of predictions using clusters:

	# Predicting Oil and VIX for a given move in cluster 0 (risk-off)
	print('Exogenous variables are: '+str(exo))
	print('Predicting moves in Oil and VIX for [-0.05, -0.1, 0.01, 1, 0, 0] in cluster 0')
	print(reg.predict(np.array([-0.05,-0.1,0.01,0,1,0]).reshape(1,-1)))

	print('Predicting moves in Oil and VIX for [-0.025, -0.1, 0.01, 1, 0, 0] in cluster 0')
	print(reg.predict(np.array([-0.025,-0.1,0.01,0,1,0]).reshape(1,-1)))

	# Predicting Oil and VIX for a given move in cluster 1 (risk-on)
	print('Predicting moves in Oil and VIX for [0.05, 0.1, -0.01, 0, 1, 0] in cluster 1')
	print(reg.predict(np.array([0.05,0.1,-0.01,1,0,0]).reshape(1,-1)))

	print('Predicting moves in Oil and VIX for [0.025, 0.1, -0.01, 0, 1, 0] in cluster 1')
	print(reg.predict(np.array([0.025,0.1,-0.01,1,0,0]).reshape(1,-1)))



	return

