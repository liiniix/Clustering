import math
data = 'kr-vs-kp.data'
from pprint import pprint
import numpy as np
import pandas as pd
import argparse
import copy
from tqdm import tqdm
import matplotlib.pyplot as plt


class KMeans:
	def __init__(self):
		pass
	
	def read_dataset(self, data_path='iris.data'):
		self.dat = pd.read_csv(data_path, header=None).sample(frac=1).reset_index(drop=True)
		self.np_dat = self.dat.iloc[:,:-1].values
		#print(self.dat.shape)
	
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
			#print(new_means)
			if means.shape[1] == new_means.shape[1]-1:
				new_means = new_means[:,:-1]
			#print(np.sum(np.abs(new_means-means)))
			if new_means.shape == means.shape and np.sum(np.abs(new_means-means))<1.e-5:
				break
			means = copy.deepcopy(new_means)
		
	def correctness(self, o_i, o_j):
		if o_i[-2] == o_j[-2] and o_i[-1]==o_j[-1]:
			return 1
		return 0
	def recall_bcubed(self):
		_sum = 0.
		for o_i in tqdm(self.dat.values):
			for o_j in self.dat[self.dat.iloc[:, -2]==o_i[-2]].values:
				if not all(o_i==o_j):
					_sum += self.correctness(o_i, o_j)/(self.dat[self.dat.iloc[:, -2]==o_i[-2]].shape[0] -1)
		return _sum/self.dat.shape[0]

	def precision_bcubed(self):
		_sum = 0.
		for o_i in tqdm(self.dat.values):
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
		return max([silhoutte_for_i(self.dat.iloc[m,:]) for m in tqdm(range(self.dat.shape[0]))])
			
if __name__ == "__main__":
	my_parser = argparse.ArgumentParser(description='')
	my_parser.add_argument('-p', help='the path to list')
	my_parser.add_argument("-e",action="store_true",help="just a flag argument")
	args = my_parser.parse_args()
	#print(vars(args)['p'])
	
	kmeans = KMeans()

	kmeans.read_dataset(data_path=vars(args)['p'])
	n_classes = len(kmeans.dat.iloc[:,-1].unique())
	if vars(args)['e']:
		costs = []
		for k in range(1, 4*n_classes):
			
			print("k = ", k)
			kmeans.run(k)
			costs.append(kmeans.get_variance())
		plt.plot([i+1 for i in range(len(costs))], costs)
		plt.xlabel("k")
		plt.ylabel("varience")
		plt.title(vars(args)['p'])
		plt.show()
		exit()
	kmeans.run(k=n_classes)
	#abul = []
	#abul.append(kmeans.precision_bcubed())
	#abul.append(kmeans.recall_bcubed())
	#abul.append(kmeans.get_variance())
	#abul.append(kmeans.silhoutte())
	#print(abul)
	print("Precision: ", kmeans.precision_bcubed())
	print("Recall: ", kmeans.recall_bcubed())
	print("Varience: " , kmeans.get_variance())
	print("Silhouette: ", kmeans.silhoutte())