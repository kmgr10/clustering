from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering
import numpy as np

def generate_cluster_labels(data, clus_method, clus_n):

	Scaling = preprocessing.MinMaxScaler().fit(data)
	X_scaled = Scaling.transform(data)

	if clus_method == 'K-means':
		clus = KMeans(n_clusters=clus_n).fit(X_scaled)

	elif clus_method == 'Agglo':
		clus = AgglomerativeClustering(n_clusters=clus_n,affinity='l2',linkage='complete').fit(X_scaled)

	return clus