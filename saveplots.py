# To save plots
import matplotlib.pyplot as plt
import numpy as np

def save_clusters(data, clus_n, directory):
	
	# Save plot of clusters

	for x in np.arange(clus_n):
		plt.figure(figsize=(14,5))
		plt.plot(data['SPX'],linewidth=1)
		cluster_indices = data[data['cluster']==x].index
		for xc in cluster_indices:
	  		plt.axvline(x=xc,color='r',alpha=0.2)
		plt.savefig(directory+ 'clus_' + str(x) + '.png')
		plt.close('all')

	return


def save_regimes(data, clus_n, directory):

	plt.figure(figsize=(14,2))
	plt.plot(data['cluster'],linestyle='None', marker='o', markersize=2,alpha=0.8)
	plt.yticks([x for x in np.arange(clus_n)])
	plt.ylabel('Regime')
	plt.savefig(directory + 'regimes.png')

	return