import math
data = 'kr-vs-kp.data'
from pprint import pprint
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
		
	def run(self, k=3):
		# randomly choose k objects
		self.k = k
		mask = np.random.randint(0, self.np_dat.shape[0], k)
		means = copy.deepcopy(self.np_dat[mask, :])
		while True:
			#print(means)
			cluster = [np.argmin(np.linalg.norm(means-self.np_dat[i,:], axis=1)) for i in range(self.np_dat.shape[0])]
			self.dat['cluster'] = cluster
			curr_cost = np.sum([self.cost(m, self.np_dat[self.dat['cluster']==i]) for i, m in enumerate(means)])
			self.variance = curr_cost
			#print(curr_cost)
			new_means = self.dat.groupby('cluster').mean().values
			#print(np.sum(np.abs(new_means-means)))
			if np.sum(np.abs(new_means-means))<1.e-5:
				break
			means = copy.deepcopy(new_means)
		
	def correctness(self, o_i, o_j):
		if o_i[-2] == o_j[-2] and o_i[-1]==o_j[-1]:
			return 1
		return 0
	def recall_bcubed(self):
		_sum = 0.
		for o_i in self.dat.values:
			for o_j in self.dat[self.dat.iloc[:, -2]==o_i[-2]].values:
				if not all(o_i==o_j):
					_sum += self.correctness(o_i, o_j)/(self.dat[self.dat.iloc[:, -2]==o_i[-2]].shape[0] -1)
		return _sum/self.dat.shape[0]

	def precision_bcubed(self):
		_sum = 0.
		for o_i in self.dat.values:
			for o_j in self.dat[self.dat.iloc[:, -1]==o_i[-1]].values:
				if not all(o_i==o_j):
					_sum += self.correctness(o_i, o_j)/(self.dat[self.dat.iloc[:, -1]==o_i[-1]].shape[0] -1)
		return _sum/self.dat.shape[0]
	
	def get_variance(self):
		#print(self.variance)
		return self.variance
	
	def silhoutte(self):
		def silhoutte_for_i(m):
			df = self.np_dat[self.dat['cluster']==m['cluster']]
			a = np.sum(self.cost(m.values[:-2], df))/(df.shape[0]-1)
			b = 1e8
			for i in range(self.k):
				if i!=m['cluster']:
					df = self.np_dat[self.dat['cluster']==i]
					b = min(b, np.sum(self.cost(m.values[:-2], df))/(df.shape[0]-1))
			return (b-a)/	max(a,b)
		return max([silhoutte_for_i(self.dat.iloc[m,:]) for m in range(self.dat.shape[0])])
			
kmeans = KMeans()

kmeans.read_dataset()
kmeans.run()
kmeans.recall_bcubed()
kmeans.precision_bcubed()
kmeans.get_variance()
kmeans.silhoutte()
