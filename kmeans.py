import math
data = 'kr-vs-kp.data'
from pprint import pprint
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import argparse
import copy


class KMeans:
	def __init__(self):
		pass
	
	def read_dataset(self, data_path='iris.data'):
		self.dat = pd.read_csv(data_path, header=None).sample(frac=1).reset_index(drop=True)
		self.np_dat = self.dat.iloc[:,:-1].values
		#print(self.np_dat)
	
	def mean(self, df):
		return df.mean(axis=0)
	def cost(self, m, df):
		return np.sum([np.sum(np.linalg.norm(m-df[i,:])) for i in range(df.shape[0])])
		
	def run(self, k=10):
		# randomly choose k objects
		mask = np.random.randint(0, self.np_dat.shape[0], k)
		means = copy.deepcopy(self.np_dat[mask, :])
		while True:
			#print(means)
			cluster = [np.argmin(np.linalg.norm(means-self.np_dat[i,:], axis=1)) for i in range(self.np_dat.shape[0])]
			self.dat['cluster'] = cluster
			curr_cost = np.sum([self.cost(m, self.np_dat[self.dat['cluster']==i]) for m in means for i in range(k)])
			print(curr_cost)
			new_means = self.dat.groupby('cluster').mean().values
			print(np.sum(np.abs(new_means-means)))
			if np.sum(np.abs(new_means-means))<.5:
				break
			means = copy.deepcopy(new_means)
kmeans = KMeans()

kmeans.read_dataset()
kmeans.run()

