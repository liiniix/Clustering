import math
data = 'kr-vs-kp.data'
from pprint import pprint
import numpy as np
import pandas as pd
import argparse
import copy
import matplotlib.pyplot as plt



class KMedoids:
	def __init__(self):
		pass
	
	def read_dataset(self, data_path='iris.data'):
		print('# read dataset')
		self.dat = pd.read_csv(data_path, header=None).sample(frac=1).reset_index(drop=True)
		self.np_dat = self.dat.iloc[:,:-1].values
		#print(self.np_dat)
	
	def mean(self, df):
		return df.mean(axis=0)
	
	def cost(self, m, df, var=False):
		if var==False:
			return np.sum([np.sum(np.abs(m-df[i,:])) for i in range(df.shape[0])])
		return np.sum([np.sum(np.linalg.norm(m-df[i,:])) for i in range(df.shape[0])])
	
	def run(self, k=3):
		print('# run')
		# randomly choose k objects
		mask = np.random.randint(0, self.np_dat.shape[0], k)
		means = copy.deepcopy(self.np_dat[mask, :])
		costs = []
		while True:
			#print(means)
			cluster = [np.argmin(np.sum(np.abs(means-self.np_dat[i,:]), axis=1)) for i in range(self.np_dat.shape[0])]
			self.dat['cluster'] = cluster
			
			curr_cost = np.sum([self.cost(m, self.np_dat[self.dat['cluster']==i]) for i, m in enumerate(means)])
			self.variance = np.sum([self.cost(m, self.np_dat[self.dat['cluster']==i], var=True) for i, m in enumerate(means)])
			#print('###', curr_cost)
			costs.append(curr_cost)
			
			new_mean_idx = -1
			new_obj_idx = -1
			min_cost = curr_cost
			
			#print(means)
			for m_idx, m in enumerate(means):
				for o_idx, o in enumerate(self.np_dat):
					swap_temp = copy.deepcopy(means[m_idx])
					means[m_idx] = copy.deepcopy(self.np_dat[o_idx])
					
					new_cost = np.sum([self.cost(me, self.np_dat[self.dat['cluster']==i]) for i, me in enumerate(means)])
					
					
					
					if new_cost < min_cost:
						#print(new_cost, min_cost)
						new_mean_idx = m_idx
						new_obj_idx = o_idx
						min_cost = new_cost
					costs.append(min_cost)
					
					means[m_idx] = copy.deepcopy(swap_temp)
			new_means = copy.deepcopy(means)
			if new_mean_idx!=-1:
				new_means[new_mean_idx] = copy.deepcopy(self.np_dat[new_obj_idx])
			#print(means, new_means)
			if np.abs(np.sum(new_means-means))<1.e-5:
				break
			means = new_means
		#print('#', np.sum([self.cost(me, self.np_dat[self.dat['cluster']==i]) for i, me in enumerate(means)]))
		
	def correctness(self, o_i, o_j):
		if o_i[-2] == o_j[-2] and o_i[-1]==o_j[-1]:
			return 1
		return 0
	def recall_bcubed(self):
		print('# recall bcubed')
		_sum = 0.
		for o_i in self.dat.values:
			for o_j in self.dat[self.dat.iloc[:, -2]==o_i[-2]].values:
				if not all(o_i==o_j):
					_sum += self.correctness(o_i, o_j)/(self.dat[self.dat.iloc[:, -2]==o_i[-2]].shape[0] -1)
		return _sum/self.dat.shape[0]
	
	def precision_bcubed(self):
		print('# precision bcubed')
		_sum = 0.
		for o_i in self.dat.values:
			for o_j in self.dat[self.dat.iloc[:, -1]==o_i[-1]].values:
				if not all(o_i==o_j):
					_sum += self.correctness(o_i, o_j)/(self.dat[self.dat.iloc[:, -1]==o_i[-1]].shape[0] -1)
		return _sum/self.dat.shape[0]
	
	def get_variance(self):
		print('# variance')
		#print(self.variance)
		return self.variance
kmedoids = KMedoids()

kmedoids.read_dataset()
kmedoids.run()
kmedoids.precision_bcubed()
kmedoids.recall_bcubed()
kmedoids.get_variance()
