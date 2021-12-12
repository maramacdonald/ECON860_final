import pandas
import numpy
from factor_analyzer import FactorAnalyzer

import matplotlib.pyplot as pyplot
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score

dataset = pandas.read_csv("dataset_final.csv")



dataset.drop(['country'], axis=1, inplace=True)
dataset.drop(['Unnamed: 0'], axis=1, inplace=True)

dataset.dropna(inplace=True)

#machine = FactorAnalyzer(n_factors=40, rotation=None)
#ev, v = machine.get_eigenvalues()
#pyplot.scatter(range(1,41), ev)
#pyplot.savefig("plot.png")
#pyplot.close()

machine = FactorAnalyzer(n_factors=4, rotation='varimax')
machine.fit(dataset)
loadings = machine.loadings_

#print("\nfactor loadings:\n")
#print(loadings)
#print(machine.get_factor_variance())

dataset = dataset.values

result = numpy.dot(dataset, loadings)
print(result)
print(result.shape)

pyplot.scatter(result[:,0], result[:,1])
pyplot.savefig("scatterplot.png")



def run_kmeans(n):
	machine = KMeans(n_clusters=n)
	machine.fit(result)
	results = machine.predict(result)
	centroids = machine.cluster_centers_
	ssd = machine.inertia_
	silhouette = 0 
	if n>1:
		silhouette = silhouette_score(result, machine.labels_, metric = 'euclidean')
	pyplot.scatter(result[:,0], result[:,1], c=results)
	pyplot.scatter(centroids[:,0], centroids[:,1], c='red', marker="*", s=200)
	pyplot.savefig("scatterplot_color_" + str(n) + ".png")
	pyplot.close()
	print( "KMeans Silhouette Score = %f" %silhouette)
	print("KMeans SSD = %f" %ssd)
 

run_kmeans(4)


def run_kmedoids(n):
	machine = KMedoids(n_clusters=n)
	machine.fit(result)
	results = machine.predict(result)
	centroids = machine.cluster_centers_
	ssd = machine.inertia_
	silhouette = 0 
	if n>1:
		silhouette = silhouette_score(result, machine.labels_, metric = 'euclidean')
	pyplot.scatter(result[:,0], result[:,1], c=results)
	pyplot.scatter(centroids[:,0], centroids[:,1], c='red', marker="*", s=200)
	pyplot.savefig("scatterplot_color_" + str(n) + ".png")
	pyplot.close()
	print( "KMedoids Silhouette Score = %f" %silhouette)
	print("KMedoids SSD = %f" %ssd)
 

run_kmedoids(4)






