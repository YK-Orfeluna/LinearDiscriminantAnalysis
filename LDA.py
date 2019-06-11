# coding: utf-8

import numpy as np

class LinearDiscriminantAnalysis :
	def __init__ (self) :
		self.w = None

	def __init_z(self) :
		self.mean = np.mean(self.X, axis=0, dtype=np.float64)
		self.stdev = np.std(self.X, axis=0, ddof=1, dtype=np.float64)

	def __zscore(self, data) :
		return (data - self.mean) / self.stdev

	def __mean_vectors(self) :
		self.mean_vecs = []
		for label in self.labels :
			self.mean_vecs.append(np.mean(self.X_z[self.y==label], axis=0, dtype=np.float64))

	def __within_matrix(self) :
		d = self.X.shape[1]
		self.S_W = np.zeros((d, d), dtype=np.float64)
		for label, mv in zip(self.labels, self.mean_vecs) :
			class_scatter = np.cov(self.X_z[self.y==label].T)
			self.S_W += class_scatter

	def __between_matrix(self) :
		mean_overall = np.mean(self.X_z, axis=0)
		d = self.X.shape[1]
		self.S_B = np.zeros((d, d), dtype=np.float64)
		for i, mean_vec in enumerate(self.mean_vecs) :
			n = self.X[self.y==i+1, :].shape[0]
			mean_vec = mean_vec.reshape(d, 1)
			mean_overall = mean_overall.reshape(d, 1)
			self.S_B += n * (mean_vec - mean_overall).dot((mean_vec - mean_overall).T)

	def __matrix(self) :
		self.eigen_vals, eigen_vecs = np.linalg.eig(np.linalg.inv(self.S_W).dot(self.S_B))
		eigen_pairs = [(np.abs(self.eigen_vals[i]), eigen_vecs[:, i]) for i in range(len(self.eigen_vals))]

		eigen_pairs = sorted(eigen_pairs, key=lambda k: k[0], reverse=True)

		t = []
		for i in range(self.N) :
			t.append(eigen_pairs[i][1][:, np.newaxis].real)
		self.w = np.hstack(t)

	def fit(self, X, y, n_components=None) :
		if type(X)==list :
			self.X = np.array(X)
		else :
			self.X = X

		self.__init_z()
		self.X_z = self.__zscore(X)

		if type(y)==list :
			self.y = np.array(y)
		else :
			self.y = y
		self.labels = np.unique(y)

		if n_components is None or n_components>X.shape[1] :
			self.N = X.shape[1]
		else :
			self.N = n_components

		self.__mean_vectors()
		self.__within_matrix()
		self.__between_matrix()
		self.__matrix()

	def transform(self, data) :
		if self.w is None :
			print("You have to train the data or read the matrix.")
			return None
		else :
			return (self.__zscore(data)).dot(self.w)

	def eigen_value(self, normalization=True) :
		if normalization :
			return [(i/sum(self.eigen_vals.real)) for i in sorted(self.eigen_vals.real, reverse=True)]
		else :
			return [i for i in sorted(self.eigen_vals.real, reverse=True)]

	def save_matrix(self, outname="LDA_matrix") :
		if not self.w is None :
			np.savetxt(outname, self.w, delimiter="\t")
		else :
			print("You have to train the data.")

	def read_matrix(self, filename="LDA_matrix") :
		self.w = np.genfromtxt(filename, delimiter="\t").astype(np.float64)

if __name__=="__main__" :
	import pandas as pd
	from sklearn.model_selection import train_test_split

	df_wine = pd.read_csv('https://raw.githubusercontent.com/rasbt/python-machine-learning-book/master/code/datasets/wine/wine.data', header=None)
	X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

	lda = LinearDiscriminantAnalysis()

	# $1: X, $2: y, $3: Number of LDA's components
	lda.fit(X_train, y_train, 2)

	X_lda = lda.transform(X_train)

	# Using 'save_matrix' >> You can save the matrix transforming to LDA's feature
	# Using `read_matrix' >> You can read the matrxi transforming to LDA's feature
	# both functions' $1: matrix's name
	lda.save_matrix("LDA_matrix")
	lda.read_matrix("LDA_matrix")

	from sklearn.linear_model import LogisticRegression

	clf = LogisticRegression(solver="newton-cg", multi_class="auto", random_state=0, class_weight="balanced")
	clf.fit(X_train, y_train)
	score = clf.score(X_test, y_test)
	print("Before LDA: ACC = %.3f" %score)

	clf = LogisticRegression(solver="newton-cg", multi_class="auto", random_state=0, class_weight="balanced")
	clf.fit(X_lda, y_train)
	score = clf.score(lda.transform(X_test), y_test)
	print("After LDA: ACC = %.3f" %score)

	"""Drawing the LDA-features"""
	import matplotlib.pyplot as plt

	colors = ['r', 'b', 'g']
	markers = ['s', 'x', 'o']

	plt.figure()
	for label, color, marker in zip(np.unique(y_train), colors, markers):
		plt.scatter(X_lda[y_train==label, 0], X_lda[y_train==label, 1], c=color, label=label, marker=marker)

	plt.xlabel("LD1")
	plt.ylabel("LD2")

	plt.title("Scatter of LDA (train)")

	plt.legend(loc="best")

	plt.show()