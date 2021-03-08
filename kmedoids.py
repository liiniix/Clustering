import math
data = 'kr-vs-kp.data'
from pprint import pprint
from sklearn.metrics import confusion_matrix
import numpy as np
import pandas as pd
import argparse
import copy


class KMedoids:
	def __init__(self):
		pass
	
	def read_dataset(self, data_path='iris.data'):
		self.dat = pd.read_csv(data_path, header=None).sample(frac=1).reset_index(drop=True)
		self.np_dat = self.dat.iloc[:,:-1].values
		#print(self.np_dat)
	
	def mean(self, df):
		return df.mean(axis=0)
	
	def cost(self, m, df):
		return np.sum([np.sum(np.abs(m-df[i,:])) for i in range(df.shape[0])])
	
	def run(self, k=3):
		# randomly choose k objects
		mask = np.random.randint(0, self.np_dat.shape[0], k)
		means = copy.deepcopy(self.np_dat[mask, :])
		while True:
			print(means)
			cluster = [np.argmin(np.sum(np.abs(means-self.np_dat[i,:]), axis=1)) for i in range(self.np_dat.shape[0])]
			self.dat['cluster'] = cluster
			
			curr_cost = np.sum([self.cost(m, self.np_dat[self.dat['cluster']==i]) for m in means for i in range(k)])
			print('###', curr_cost)
			
			new_mean_idx = -1
			new_obj_idx = -1
			min_cost = curr_cost
			print(means)
			for m_idx, m in enumerate(means):
				for o_idx, o in enumerate(self.np_dat):
					swap_temp = copy.deepcopy(means[m_idx])
					means[m_idx] = copy.deepcopy(self.np_dat[o_idx])
					print(means)
					new_cost = np.sum([self.cost(me, self.np_dat[self.dat['cluster']==i]) for me in means for i in range(k)])
					print(new_cost)
				
					if new_cost < min_cost:
						print(new_cost, min_cost)
						new_mean_idx = m_idx
						new_obj_idx = o_idx
						min_cost = new_cost
			
					
					means[m_idx] = copy.deepcopy(swap_temp)
			new_means = copy.deepcopy(means)
			if new_mean_idx!=-1:
				new_means[m_idx] = copy.deepcopy(self.np_dat[o_idx])
			if np.abs(np.sum(new_means-means))<.5:
				break
			means = new_means
kmedoids = KMedoids()

kmedoids.read_dataset()
kmedoids.run()

