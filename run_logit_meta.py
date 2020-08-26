file_name = "logit_meta/"
import multiprocessing
from itertools import repeat


#import findspark

#findspark.init()

from tmp import *
#from skdist.distribute.ensemble import (
#    DistRandomForestClassifier,
#    DistExtraTreesClassifier
#    )

#from skdist.distribute.search import RandomizedSearchCV
from sklearn.kernel_approximation import *
import math
import copy
from sklearn.svm import *
import json
from sklearn.metrics import *
import pickle
from helpers import *
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
#from pyspark import SparkContext, SparkConf
from sklearn.model_selection import *
import numpy as np
from imblearn.ensemble import *
from imblearn.over_sampling import *
from sklearn.cluster import *

#from spark_sklearn.grid_search import GridSearchCV

#from sklearn.model_selection import GridSearchCV
#from pyspark import SparkContext, SparkConf

#from sklearn.utils import parallel_backend
#from sklearn.externals import joblib
import joblib
from joblib import parallel_backend

#from pyspark.sql import SparkSession
import xgboost as xgb


#from joblibspark import register_spark
from sklearn.linear_model import *
import random
from sklearn.ensemble import *
from joblib import Parallel, delayed
from sklearn.metrics import *
from sklearn.preprocessing import *
#from sklearn.pipeline import Pipeline
from imblearn.pipeline import Pipeline
import os
from sklearn.ensemble import *
from sklearn.model_selection import train_test_split

chromosomes = "101111110111001100000000110000010010010001100110011000000011001111101001100100010101001000001000010101001010000000011010001100000100000110101110101000010111110110011101110111011101110101001011111001001111000011100111100"

import joblib

#import dask.array as da

#from dask_ml.wrappers import ParallelPostFit
#from xgboost import XGBClassifier
random.seed(2000)
np.random.seed(2000)
import time

def fit_classifier(svc_rbf, param, X_train, y_train, X_test, y_test):
	clas = svc_rbf.set_params(**param)
	clas.fit(X_train, y_train)
	#return param, roc_auc_score(y_test, [a[1]  for a in clas.predict_proba(X_test)] )
	return param, roc_auc_score(y_test, [ a[1] for a in clas.predict_proba(X_test)]  )


def gridsearch(X, y, param_space, svc_rbf, skf, metric=roc_auc_score):
	my_hash = {}
	print(len(X))

	i = 0	
	my_param_scores = Parallel(n_jobs=-1)(delayed(fit_classifier)(svc_rbf, param, X[train], y[train], X[test], y[test] ) for param in ParameterGrid(param_space) for train,test in skf.split(X, y) )
	#for train, test in skf.split(X, y):
	#	print(train)
	#	param_score = Parallel(n_jobs=-1)(delayed(fit_classifier)(svc_rbf, param, X[train], y[train], X[test], y[test] ) for param in ParameterGrid(param_space) )
                #best_param, best_score = max(param_score, key=lambda x: x[1])
                #print('best param this generation is {} with score {}.'.format(best_param, best_score))
	#	for my_val in param_score:
	#		key = json.dumps(my_val[0] )
	#		if key not in my_hash.keys() or my_hash[key]==None:
	#			my_hash[key ] = my_val[1]
	#		else:
	#			my_hash[key ] = my_hash[key ] + my_val[1]
	#	best_param, best_score = max(param_score, key=lambda x: x[1])
	#	print('Best scoring param is {} with score {}.'.format(best_param, best_score))
	#	i+=1
	#print(my_param_scores)
	#for param_score in my_param_scores:
	for my_val in my_param_scores:
		#key = my_val[0]
		#print(my_val)
		#key = my_val#[0]
		#print(key)
		key = json.dumps(my_val[0] )
		#print(key)
		if key not in my_hash.keys() or my_hash[key]==None:
			my_hash[key ] = my_val[1]
		else:
			my_hash[key ] = my_hash[key ] + my_val[1]
	#i+=1
	i = skf.get_n_splits(X, y)
	param_scores = []
	for k, v in my_hash.items():
		param_scores.append( (k, v/i ) )
	best_param, best_score = max(param_scores, key=lambda x: x[1])
	print('Best scoring param for all folds is: {} with score {}.'.format(best_param, best_score))
	return best_param
class Step:
	def __init__(self, classifiers= [], X = [], y= [], meta= None, ):
		self.classifiers = classifiers#[ ParallelPostFit(a) for a in classifiers ]
		#self.X = X
		#self.y = y
		#self.sc = sc
		#vals = np.concatenate( ([[a] for a in y], X ), axis=1)
		#np.random.shuffle(vals)
		#self.X = [a[1:] for a in vals]
		#self.y = [a[0] for a in vals]	
	
		#vals = []
		#for a in range(len(X)):
		#	vals.append([y[a], X[a]] )
		#random.shuffle(vals)
		#X_= []
		#y_=[]
		#for a in vals:
		#	y_.append(vals[0] )
		#	X_.append(vals[0:] )
		#X = X_
		#y = y_
		self.base_X = X#[ int(0.8 * len(X) ): ]
		self.base_y = y#[ int(0.8 * len(y) ): ]
		self.X = np.array( X)#[:int(0.2 * len(X) )] )
		self.y = np.array(  y)#[:int(0.2 * len(y) )] )
		#meta = BalancedBaggingClassifier(random_state=10, n_estimators=100, n_jobs=-1)
		
		self.grid = None
		self.sc = None
		self.skfold = StratifiedKFold(n_splits=5, random_state=200, shuffle=True)
		if meta == None:
			#km_est = MiniBatchKMeans(batch_size=1000, n_clusters=30)
			pipe = Pipeline([('anova', StandardScaler() ), ('lr', LogisticRegression(class_weight='balanced', max_iter=1000))]) 
			#pipe = Pipeline([('scale', StandardScaler() ) , ('sample', Nystroem(random_state=100) ), ('lr', LogisticRegression(class_weight='balanced', max_iter=500) ) ])
			
			#pipe = Pipeline([('anova', StandardScaler() ), ('lr', ExtraTreesClassifier(class_weight='balanced', n_jobs=1))])
			#pipe = Pipeline([('anova', StandardScaler() ), ('lr', RUSBoostClassifier(n_estimators=200) ) ])
			self.grid = {'lr__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000], }#'os__k_neighbors' :[5, 15, 25, 35, 45]  }
			#g_range = np.linspace(-15,3,num=10)
			#gamma_range = [math.pow(2,j) for j in g_range]
			#c_range = np.linspace(-5,15,num=11)
			#C_range = [math.pow(2,i) for i in c_range]
			#self.grid = {
			#	'lr__C': C_range,
			#	'sample__gamma': gamma_range,
			#	'sample__n_components': [2000],
			#}
			#self.grid = {'lr__n_estimators': [500, 1000, 1500, 2050, 3050, 4050,], 'lr__max_depth':[None, 10, 20, 30, 40, 45],
			#'lr__min_samples_split':[0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01  ], #
			#'lr__min_samples_split':[10, 15, 35, 45, 55, 75]# 65, 85, 95, ],
			#}
			#self.grid = { 'lr__n_estimators': [20, 50, 100, 150 ]
			#'lr__learning_rate': [0.1, 0.2, 0.5, 1 ] }
			#self.meta = None
			self.sc = None#SparkContext(master='spark://137.30.125.208:7077', appName='JoblibSparkBackend')
				
			#params = {'lr__class_weight': 'balanced', 'lr__max_depth': 10, 'lr__min_samples_split': 0.001, 'lr__n_estimators': 2000}
			#pipe.set_params(**params)
			self.meta = pipe
			

			#self.meta = GridSearchCV(LogisticRegression(class_weight='balanced', max_iter=1000), param_grid={'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000] }, n_jobs=-1,  cv=5)
			#self.meta = GridSearchCV(pipe, param_grid=grid,  cv=3, n_jobs=-1, scoring="balanced_accuracy")
		else:
			self.meta = meta

		self.arr_length = len(X[0] )
		
		#self.X = self.sc.parallelize(np.array(X).flatten() )
		#self.y = self.sc.parallelize(np.array(y).flatten() )
		#self.X = da.from_array(X)
		#self.y = da.from_array(y)
		#self.__train()
		#self.predictions = self.__generate_predictions()
		self.predictions = []

	def __train(self):
		if self.classifiers is not None:
			#for classifier in self.classifiers:
			#	classifier.fit(self.X, self.y)
			#with parallel_backend('spark',):
			#ts = str(time.time() )
			#joblib.dump(self.X, ts)
			#self.X = joblib.load(ts, mmap_mode='r+')
			
			#ts_1 = str(time.time() )
			#joblib.dump(self.y, ts_1)
			#self.y = joblib.load(ts_1, mmap_mode='r+')
			#with parallel_backend('spark',):
			#with parallel_backend('spark',):
			#dat = np.concatenate(( [[a] for a in self.y ], self.X), axis=1)
			#X = sc.parallelize(np.array(self.base_X).flatten(), int(len(self.X)/3) )
			#y = sc.parallelize(np.array(self.base_y).flatten(), int(len(self.y)/3) )
			#dat = sc.parallelize( np.array(dat), int(len(dat)/5) ) 
			X = np.array(self.base_X)
			#joblib.dump(X, "tmpX")
			#X = joblib.load("tmpX", mmap_mode='r+')
			y = np.array(self.base_y)
			#self.classifiers = Parallel(n_jobs=-1, )(delayed( a.fit ) ( X,y ) for a in self.classifiers)
			#os.remove(ts)			
			#os.remove(ts_1) 
	
	def __generate_predictions(self, X = None, y = None, sc= None):
		if X is None:
			X = self.X

		pred = []
		if y is None:
			y = self.y
		#joblib.dump(X, "tmpX")
		#X = joblib.load("tmpX", mmap_mode='r+')
		#sc = SparkSession.builder.master('spark://137.30.125.208:7077').appName('JoblibSparkBackend').getOrCreate().sparkContext
		#if sc is None:
		#	sc = SparkSession.builder.appName('JoblibSparkBackend').getOrCreate().sparkContext
		#X = sc.parallelize(X_meta.flatten(), int(len(X_meta)/5000))
		#X = sc.parallelize(X, int(len(X)/100))
		#skfold = 10#StratifiedKFold(n_splits=20, random_state=100, shuffle=True)
		#skfold = StratifiedKFold(n_splits=5, random_state=100,)
		
		skfold = KFold(n_splits=5, random_state=100,)
		#if self.classifiers is not None:
		#	with parallel_backend('loky',):
				#predictions = Parallel(n_jobs=-1, )(delayed( a.predict_proba) (X ) for a in self.classifiers) 
				#
		#pred = Parallel(n_jobs=-1, )(delayed( a.predict_proba) (X ) for a in self.classifiers)
		#my_pred = []
		#for a in pred:
		#	my_pred.append([ [m[0]] for m in a ] )
		#pred = my_pred
		#for classifier in self.classifiers:
		#		#	#print(classifier.predict_proba(X))
		#		#	pred.append([ a[0]  for a in classifier.predict_proba(X) ] )
		#			proba = cross_val_predict(classifier, X, self.y, cv=skfold, method='predict_proba', n_jobs=-1)
		#			predictions.append([a for a in proba ] )
					#vals = self.brier_multi( self.y, [ a[0] for a in proba] )
					#to_append = []
					#for val in range(len(vals) ):
					#	my_val = proba[a].append(vals[va] )
					#	to_append.append(my_val)
					#	predictions.append(to_append)
				#	#vals = X.map(lambda x : classifier.predict_proba([x])[0] ).map(lambda elem: list(elem)[0]).collect()
				#	#predictions.append(vals)
		#predictions = np.array([ a.compute() for a in predictions ])
		#print(X)
		#X = np.array( [ list(a) for a in X ] )
		#y = np.array(y)
		#print(X)
		#print(y)
		#from tmp import *
		pred = train_base(np.array(X) , np.array(y), self.classifiers, skfold, batch_size=100000)
		#print(val)
		pred = np.array(pred)
		
		self.classifiers = Parallel(n_jobs=-1, )(delayed( a.fit ) ( X,y ) for a in self.classifiers)
		#if len(pred) > 10:
		#	pred = pred.T
		
		#normal
		#mypred = []
		#for a in pred:
		#	mypred.append([item for sublist in a for item in sublist] )
		#pred = np.array(mypred )
		#print(pred)#[0])#[0])
		
		#for normal one
		return pred#T#[0]#.T
		

		#return np.concatenate(predictions.T, axis=1)
		#return pred.T
	def brier_multi(self,targets, probs):
		return np.mean(np.sum((probs - targets)**2, axis=1))
	def get_base_predictions(self, X):
		predictions_ = []
		
		#parallel prediction
		#pred = Parallel(n_jobs=-1, )(delayed( a.predict_proba) (X ) for a in self.classifiers)
		#my_pred = []
		#for a in range(len(pred[0])):
		#	my_val = []
		#	for b in range(len(pred)):
		#		my_val.append(  [ pred[b][a][0] ]  )
		#	my_pred.append(my_val )
		#predictions_ = my_pred
		
		#serial prediction
		for classifier in self.classifiers:
		#	#predictions.append([ a  for a in classifier.predict_proba(X) ] )
		#	#mp = [ a[0] for a in classifier.predict_proba(X) ]
			predictions_.append([ a[0] for a in classifier.predict_proba(X) ] )
			
		#my_pred = []
		#for a in
		my_pred = np.array(predictions_) 
		
		print(len(my_pred) )
		return np.array(my_pred).T
		#return np.array([ np.ndarray.flatten(np.array(m) ) for m in np.array(predictions).T ] ).T
	def train(self, X, y, sc = None):
		#if sc == None:
		#	sc = SparkSession.builder.appName('JoblibSparkBackend').getOrCreate().sparkContext 
		
		#vals = []
		#for a in range(len(X)):
		#       vals.append([y[a], X[a]] )
		#random.shuffle(vals)
		#X_= []
		#y_=[]
		#for a in vals:
		#       y_.append(vals[0] )
		#       X_.append(vals[0:] )
		#X = X_
		#y = y_
		self.__train()
		pred = self.__generate_predictions(X, y, sc=sc)
		#predictions = generate_predictions()
		#print(self.predictions)
		#pred = np.array(pred.T)
		#if len(pred) > 6:
		#	pred = pred.T
		#print(X)
		#print(pred.T)#np.concatenate(pred, axis=1).T )
		#pred = pred.T#np.concatenate(pred , axis=1).T 
		X_meta = np.concatenate((X, pred), axis=1)
		length = len(X_meta[0] )
		#X_meta = sc.parallelize(X_meta.flatten(), int(len(X_meta)/5))
		y = np.array(y)	
		#pipe = Pipeline([('anova', StandardScaler() ), ('lr', LogisticRegression(class_weight='balanced', max_iter=100))])
		#pipe = Pipeline([('anova', StandardScaler() ), ('lr', ExtraTreesClassifier(class_weight='balanced'))])
		#grid = {'lr__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],  }
		#grid = {'lr__n_estimators': [100, 200, 400, 500, 1000, 1500, 2000], 'lr__max_depth':[None, 5, 10, 15, 20, 40, 50],
		#'lr__min_samples_split':[2, 5, 7, 10], 'lr__n_jobs':[-2, ],
		#}
		#sc = SparkSession.builder.appName('JoblibSparkBackend').getOrCreate().sparkContext
		#if self.meta != None:
		#	#self.meta = GridSearchCV(sc, pipe, param_grid=grid,  cv=5, n_jobs=-2)
		#	self.meta = GridSearchCV(pipe, param_grid=grid,  cv=5, n_jobs=-2)
		#y = sc.parallelize(np.array(self.y).flatten(), int(len(self.y)/5) )
		#with parallel_backend('loky',):
		#self.meta = fit_clas(self.meta, X_meta, y , length)
		#self.meta.fit(X_meta, self.y)
		print(len(X_meta))
		print(len(y ))
		#print(X_meta[0] )
		#print(y[0] )
		
		#for searching parameters
		skfold = StratifiedKFold(n_splits=5, random_state=200, shuffle=True)		
		#self.search.fit(X_meta, y)
		#best_params = self.search.best_params_
		#self.meta.set_params(lr__n_estimators=best_params['lr__n_estimators'], lr__min_samples_split=best_params['lr__min_samples_split'], lr__max_depth=best_params['lr__max_depth']  )	
		
		
		#for lr
		#self.meta.fit(X_meta, y)
		#set the spark context to none, for saving purposes
		#self.sc = None
		#self.grid=None
		#if top layer is none	
		best_param = gridsearch(X_meta, y, self.grid, self.meta, skfold) 
		#print(self.search.best_estimator_ )
		param = json.loads(best_param)
		#param = {'lr__class_weight': 'balanced', 'lr__max_depth': 45, 'lr__min_samples_split': 10, 'lr__n_estimators': 4000}
		self.meta = self.meta.set_params(**param)
		
		pred = cross_val_predict(self.meta, X_meta, y, cv=skfold, method='predict_proba', n_jobs=-1)
		report(y, pred)
		self.meta.fit(X_meta, y)
	def predict_proba(self, X = None):
		pred = []
		#sc = SparkSession.builder.appName('JoblibSparkBackend').getOrCreate().sparkContext 
		if X is None:
			X = self.X
			pred = self.predictions
			
		else:
			pred = self.get_base_predictions(X)
		
		X_meta = np.concatenate((X, pred), axis=1)
		#return self.meta.predict_proba(X_meta)
		#X_meta = sc.parallelize(X_meta, int(len(X_meta)/100))
		with parallel_backend('loky'):
			#predictions = Parallel(n_jobs=-1, )(delayed( self.meta.predict_proba) ([a] ) for a in X_meta)
			return [ a for a in self.meta.predict_proba(X_meta) ]
		#with parallel_backend('spark',):
		#	val = X_meta.map(lambda x: self.meta.predict_proba([x])[0] ).map(lambda elem: list(elem)[0]).collect()
		#	return val
	def get_predictions(self, X=None):
		pred = []
		#sc = SparkSession.builder.appName('JoblibSparkBackend').getOrCreate().sparkContext 
		if X is None:
			X = self.X
			pred = self.predictions
			
		else:
			pred = self.get_base_predictions(X) 
		#predictions = [ 
		#if len(pred) <6:
		#	pred = pred.T
		X_meta = np.concatenate((X, pred), axis=1)
		#X_meta = self.sc.parallelize(np.array(X_meta).flatten(), int(len(X_meta)/5000) )
		#return np.array( [ a for a in self.meta.predict_proba(X_meta) ] )
		with parallel_backend('loky',):
			#predictions = Parallel(n_jobs=-1, )(delayed( self.meta.predict_proba) ([a] ) for a in X_meta)
		#	#return [a.collect() for a in vals ]
			return [a[0] for a in self.meta.predict_proba(X_meta) ]
		#X_meta = sc.parallelize(X_meta)
		#with parallel_backend('spark',):
		#	val= X_meta.map(lambda x: self.meta.predict_proba([x])[0] ).map(lambda elem: list(elem)[0]).collect()
		#	return [ [a] for a in val ]
	def predict(self, X = None):
		pred = []
		#sc = SparkSession.builder.appName('JoblibSparkBackend').getOrCreate().sparkContext 
		if X is None:
			X = self.X
			pred = self.predictions
		else:
			pred = self.get_base_predictions(X)
			
		X_meta = np.concatenate((X, pred), axis=1)
		#return self.meta.predict(X)
		with parallel_backend('loky',):
			#predictions = Parallel(n_jobs=-1, )(delayed( self.meta.predict) ([a] ) for a in X_meta)
			pred = self.meta.predict(X_meta)
			return [a  for a in pred]
		#X_meta = sc.parallelize(X_meta)
		#with parallel_backend('spark',):
		#	val=X_meta.map(lambda x: self.meta.predict_proba(x)[0] ).map(lambda elem: list(elem)[0] ).collect()
		#	return val
def loop_a(fil, step, window_, test):
	X_initial = []
	y_initial = [ ]
	#for file in path_1[: int(0.6 * len(path_1) )]:
	with open("buoy_data/test_step" + str(step) +"/window_"+ str(window_) +"_dataset/" + fil, "r") as f:
		data = [a for a in f.readlines() ]
		if len(data) == 0:
			return [], []#continue
		my_data = data
		#if test != True:
		#	rand_vals = [random.randint(1,len(data) ) for d in range(int(len(data)*0.98 ) )]
		#	my_data = [ data[a] for a in range(len(data)) if data[a][0]==1 or a in rand_vals ]
		for line in my_data:#[:100000]
			#output = int(line.split(",")[0]
			features = [float(x) for x in line.strip().split(",") if x !='']
			inputs = features[1:]
			if len(X_initial) == 0 or len(inputs) == len(X_initial[0] ):
				X_initial.append(inputs  )
				y_initial.append( int(features[0]) )

	return X_initial, y_initial
def loop_b(fil, step, window_, test):
	X_initial = []
	y_initial = [ ]
	#for file in path_1[: int(0.6 * len(path_1) )]:
	with open("buoy_data/step" + str(step) +"/window_"+ str(window_) +"_dataset/" + fil, "r") as f:
		data = [a for a in f.readlines() ]
		if len(data) == 0:
			return [],[]
		my_data = data
		#if test != True:
		#	rand_vals = [random.randint(1,len(data) ) for d in range(int(len(data)*0.98 ) )]
		#	my_data = [ data[a] for a in range(len(data)) if data[a][0]==1 or a in rand_vals ]
		for line in my_data:#[:100000]
			features = [float(x) for x in line.strip().split(",") if x !='']
			inputs = features[1:]
			if len(X_initial) == 0 or len(inputs) == len(X_initial[0] ):
				X_initial.append(inputs  )
				y_initial.append( int(features[0]) )

	return X_initial, y_initial

def get_data(step, window_, test = False):
	#step = sys.argv[1]
	#window_ = sys.argv[2]
	random.seed(200)
	X_initial = []
	y_initial = []
	#if test !=True:
	#	pat = "buoy_data/test_step"
	#	#return [], []
	#else:
	#	pat = "buoy_data/step"
	#	#return [], [] # dont have anything for now, need to run again
	path_1 = os.listdir("buoy_data/test_step" + str(step) +"/window_"+ str(window_) +"_dataset")
	path_2 = os.listdir("buoy_data/step" + str(step) +"/window_"+ str(window_) +"_dataset")
	
	random.shuffle(path_1)
	random.shuffle(path_2)
	#if test != True:
	#	paths = paths#[: int( 0.5 * len(paths) )]
	import glob

	excludes = glob.iglob("buoy_data/nodc/ftp.nodc.noaa.gov/nodc/archive/arc0060/0111915/**", recursive=True)
	excludes = [ a.split("/")[-1] for a in excludes if os.path.isfile(a) and a.split("/")[-1][:2] == 'sp' ]

	#test data is in this folder
	#test_data = glob.iglob("buoy_data/nodc/skipped_data/**", recursive=True)
	test_data = glob.iglob("ftp.nodc.noaa.gov/nodc/archive/arc0098/**", recursive=True)
	test_data = [ a.split("/")[-1] for a in test_data if os.path.isfile(a) and a.split("/")[-1][:2] == 'sp' ]	
	test_data1 = glob.iglob("buoy_data/nodc/archive/archive/arc0087/0145619/10.10/**", recursive=True)
	test_data1 = [ a.split("/")[-1] for a in test_data1 if os.path.isfile(a) and a.split("/")[-1][:2] == 'sp' ]
	
	test_data2 = glob.iglob("buoy_data/nodc/archive/archive/arc0105/0161019/**", recursive=True)
	test_data2 = [ a.split("/")[-1] for a in test_data2 if os.path.isfile(a) and a.split("/")[-1][:2] == 'sp' ]
	test_data = test_data + test_data1 + test_data2
	#excludes_2 = glob.iglob("buoy_data/nodc/archive/arc0069/**", recursive=True)
	#excludes.extend ([ a.split("/")[-1] for a in excludes_2 if os.path.isfile(a) and a.split("/")[-1][:2] == 'sp' ])

	#excludes_3 = glob.iglob("buoy_data/nodc/archive/archive/**", recursive=True)
	#excludes.extend ([ a.split("/")[-1] for a in excludes_3 if os.path.isfile(a) and a.split("/")[-1][:2] == 'sp' ])
	
	path_1 = set([a for a in path_1 if a not in excludes ])
	path_2 = set([a for a in path_2 if a not in excludes ])
	#for test, only give the test data
	if test != True:
		path_1 = [a for a in path_1 if a in test_data]
		path_2 = [a for a in path_2 if a in test_data]
	#for train, skip the test data
	if test == True:
		path_1 = [a for a in path_1 if a not in test_data]
		path_2 = [a for a in path_2 if a not in test_data]
#	
	pool = multiprocessing.Pool(processes = multiprocessing.cpu_count())
	X_initial = []
	y_initial = []
#	X_initial_1, y_intitial_1 = 
	for a, b in pool.starmap(loop_a, [(a, step, window_, test) for a in path_1  ]  ):
		X_initial.extend(a)
		y_initial.extend(b)
	for a, b in pool.starmap(loop_b, [(a, step, window_, test) for a in path_2  ] ):
		X_initial.extend(a)
		y_initial.extend(b)
	indices = []
	for x in range(len(chromosomes)):
		if chromosomes[x] == '1':
			indices.append(x )
	
	X_ = []
	for x in X_initial:
		X_.append( np.array( [ x[b] for b in indices ] )  )
	X_initial = X_
	X_initial, _, y_initial, _ = train_test_split( X_initial, y_initial, test_size=0.2, random_state=42, shuffle=True, stratify=y_initial)
	print(len (X_initial ) )
	if test == False:
		return X_initial, y_initial
		#return X_initial[:40000], y_initial[:40000]
	else:
		return X_initial, y_initial
		#return X_initial[:10000], y_initial[:10000]
		
def report(y, predicted):
	print("\n")
	r_score = roc_auc_score(y, [a[0] for a in predicted ] )
	print("ROC AUC:" + str(r_score) )
	print( 1 - r_score )
	predicted = [a[0]<0.5 for a in predicted ]
	confusion = confusion_matrix(y, predicted)
	print(confusion)
	TP = confusion[1, 1]
	TN = confusion[0, 0]
	FP = confusion[0, 1]
	FN = confusion[1, 0]
	#Specificity
	SPE_cla = (TN/float(TN+FP))

	#False Positive Rate
	FPR = (FP/float(TN+FP))

	#False Negative Rate (Miss Rate)
	FNR = (FN/float(FN+TP))

	#Balanced Accuracy
	ACC_Bal = 0.5*((TP/float(TP+FN))+(TN/float(TN+FP)))

	#compute MCC
	MCC_cla = matthews_corrcoef(y, predicted)
	F1_cla = f1_score(y, predicted)
	PREC_cla = precision_score(y, predicted)
	REC_cla = recall_score(y, predicted)
	Accuracy_cla = accuracy_score(y, predicted)
	print('TP = ', TP)
	print('TN = ', TN)
	print('FP = ', FP)
	print('FN = ', FN)
	print('Recall/Sensitivity = %.5f' %REC_cla)
	print('Specificity = %.5f' %SPE_cla)
	print('Accuracy_Balanced = %.5f' %ACC_Bal)
	print('Overall_Accuracy = %.5f' %Accuracy_cla)
	print('FPR_bag = %.5f' %FPR)
	print('FNR_bag = %.5f' %FNR)
	print('Precision = %.5f' %PREC_cla)
	print('F1 = %.5f' % F1_cla)
	print('MCC = %.5f' % MCC_cla)
	print("\n")

if __name__ == "__main__":
	random.seed(100)
	#conf = SparkConf().setMaster("spark://137.30.125.208:7077").setAppName("JoblibSparkBackend")
	#conf.set("spark.executor.heartbeatInterval","120s")
	#conf.set("spark.network.timeout", "120s")
	#conf.set("spark.executor.memory", "64g")
	#sc = SparkContext( conf=conf)
	#from pyspark import broadcast
	import pickle


	def broadcast_dump(self, value, f):
		pickle.dump(value, f, 4)  # was 2, 4 is first protocol supporting >4GB
		f.close()
		return f.name


	#broadcast.Broadcast.dump = broadcast_dump
	#sc = SparkContext(master='spark://137.30.125.208:7077', appName='JoblibSparkBackend')
	#sc=None
	#sc.setLogLevel("ERROR")
	
	#register_spark()
	#from joblibspark.backend import *
	from dask.distributed import Client, SSHCluster
	#cluster = SSHCluster([
	#'localhost',#'localhost',
	#"nrl-03.cs.uno.edu", "nrl-05.cs.uno.edu",
	#"nrl-06.cs.uno.edu", "nrl-07.cs.uno.edu",
	# "gulfscei-linux.cs.uno.edu",
	#],
	#connect_options={"known_hosts": None},
	#scheduler_options={"port": 0, "dashboard_address": ":8799"},
	# #worker_module='dask_cuda.dask_cuda_worker'
	#worker_module = "distributed.cli.dask_worker"
	#)

	#worker_module='dask_cuda.dask_cuda_worker')
	#client = Client()#cluster,processes=False, threads_per_worker=1)# wait_for_workers_timeout=40)
	#from spark_sklearn.grid_search import GridSearchCV
	#client.restart()


	X = [ [1, 2, 3], [3, 4, 5,], [4, 5, 6,], [6, 7, 8], [2, 3, 4], [3, 4, 5], [2, 4, 5], [3, 4, 4]]
	X= np.array(X)
	y = [1, 0, 1, 1, 0, 0, 1, 0]

	classifiers = [ LogisticRegression(), LogisticRegression(C= 1000)]

	#step 1
	logistic_params = {'C': 1, 'class_weight': 'balanced', 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1, 'max_iter': 100, 'multi_class': 'ovr',  'penalty': 'l2', 'random_state': 100, 'solver': 'liblinear', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}
	random_forest = {'bootstrap': True, 'class_weight': 'balanced', 'criterion': 'gini', 'max_depth': 35, 'max_features': 'auto', 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 5000, 'n_jobs': -1, 'oob_score': False, 'random_state': 100, 'verbose': 0, 'warm_start': False}

	gb = {'criterion': 'friedman_mse', 'init': None, 'learning_rate': 0.05, 'loss': 'deviance', 'max_depth': 8, 'max_features': None, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 20, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 2000, 'presort': 'auto', 'random_state': 100, 'subsample': 1.0, 'verbose': 0, 'warm_start': False}
	sgd = {'alpha': 0.0001, 'average': False, 'epsilon': 0.1, 'eta0': 0.0, 'fit_intercept': True, 'l1_ratio': 0.15, 'learning_rate': 'optimal', 'loss': 'log', 'max_iter': None, 'n_jobs': -1, 'penalty': 'l2', 'power_t': 0.5, 'random_state': 100, 'shuffle': True, 'tol': None, 'verbose': 0, 'warm_start': False}
	svc_params = {"sample__gamma": 0.0078125, "sample__n_components": 2000, "sgd__C": 8192.0}
	
	svp = Pipeline([('scale', StandardScaler() ) , ('sample', Nystroem(random_state=100) ), ('sgd', LogisticRegression(class_weight='balanced', max_iter=1000, n_jobs=1) ) ])
	svp.set_params(**svc_params)	
	et_params =  {"lr__max_depth": 40, "lr__min_samples_split":10, "lr__n_estimators": 6000}
	et =  Pipeline([('anova', StandardScaler()), ('lr',ExtraTreesClassifier(random_state=100, class_weight='balanced' , n_jobs=1) ) ] )
	et.set_params(**et_params )	
	
	meta_params = {"lr__max_depth": 20, "lr__min_samples_split":50, "lr__n_estimators": 4000}
	meta = Pipeline([('anova', StandardScaler()), ('lr',ExtraTreesClassifier(random_state=100, class_weight='balanced' , n_jobs=1) ) ] )
	meta.set_params(**meta_params )
	#classifiers = [Pipeline([('anova', StandardScaler() ), ('lr', LogisticRegression(**logistic_params))]), Pipeline([('anova', StandardScaler() ), ('lr',RandomForestClassifier( **random_forest)) ] ), Pipeline([('anova', StandardScaler() ), ('lr',xgb.XGBClassifier(**gb) ) ]) ,]
	#qd = QDA()
	lr = Pipeline([('anova', StandardScaler() ), ('lr', LogisticRegression(**logistic_params))])
	classifiers = [lr, svp, et, ]#qd ]
	
	#classifiers = [Pipeline([('anova', StandardScaler() ), ('lr', LogisticRegression(**logistic_params))]), Pipeline([('anova', StandardScaler() ), ('lr',RandomForestClassifier( **random_forest)) ] ),]
	X_test, y_test = get_data(1, 4, test=True)
	X_train, y_train = get_data(1, 4, )
	
	#X_tr, X_te, y_tr, y_te = train_test_split( X_train, y_train, test_size=0.2, random_state=42, shuffle=True, stratify=y_train)
	#X_train = X_tr + X_te
	#y_train = y_tr + y_te	
	#X_train, X_test, y_train, y_test = train_test_split( X+X_test, y+y_test, test_size=0.2, random_state=42, shuffle=True)
	#print("got data")
	mystep_1 = Step(classifiers= classifiers, X = X_train, y = y_train, )
	
	#print("fitting")
	mystep_1.train(X_train, y_train)

	#X_test = [ [5, 6, 2], [3, 4, 5], [8, 9, 12], ]
	#y_test = [1, 0, 1]
	preds = np.array( mystep_1.get_predictions(X_test) )
	#print(preds)
	#print(X)
	report(y_test, [ [ a] for a in preds ] )
	#X = np.concatenate((X, preds), axis=1)


	with open(file_name + '1_2.pickle', 'wb') as handle:
		joblib.dump(mystep_1, handle,)# protocol=pickle.HIGHEST_PROTOCOL)
	#print(X)
	#step 2

	logistic_params = {'C': 0.1, 'class_weight': 'balanced', 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1, 'max_iter': 100, 'multi_class': 'ovr', 'penalty': 'l2', 'random_state': 100, 'solver': 'liblinear', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}
	random_forest = {'bootstrap': True, 'class_weight': 'balanced', 'criterion': 'gini', 'max_depth': 40, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 5000, 'n_jobs': -1, 'oob_score': False, 'random_state': 100, 'verbose': 0, 'warm_start': False}
	gb = {'criterion': 'friedman_mse', 'init': None, 'learning_rate': 0.05, 'loss': 'deviance', 'max_depth': 8, 'max_features': None, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 20, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 2000, 'presort': 'auto', 'random_state': 100, 'subsample': 1.0, 'verbose': 0, 'warm_start': False}
	sgd = {'alpha': 0.001, 'average': False, 'class_weight': 'balanced', 'epsilon': 0.1, 'eta0': 0.0, 'fit_intercept': True, 'l1_ratio': 0.15, 'learning_rate': 'optimal', 'loss': 'log', 'max_iter': None, 'n_jobs': -1, 'penalty': 'l2', 'power_t': 0.5, 'random_state': 100, 'shuffle': True, 'tol': None, 'verbose': 0, 'warm_start': False}
	
	#classifiers = [Pipeline([('anova', StandardScaler() ), ('lr', LogisticRegression(**logistic_params))]), Pipeline([('anova', StandardScaler() ), ('lr',RandomForestClassifier( **random_forest)) ] ), Pipeline([('anova', StandardScaler() ), ('lr',xgb.XGBClassifier(**gb) ) ]) ,]
	svc_params = {"sample__gamma": 0.0078125, "sample__n_components": 2000, "sgd__C": 32768.0}
	svp = Pipeline([('scale', StandardScaler() ) , ('sample', Nystroem(random_state=100) ), ('sgd', LogisticRegression(class_weight='balanced', max_iter=1000, n_jobs=1) ) ])
	svp.set_params(**svc_params)
	et_params =  {"lr__max_depth": 35, "lr__min_samples_split": 40, "lr__n_estimators": 6500}
	et =  Pipeline([('anova', StandardScaler()), ('lr',ExtraTreesClassifier(random_state=100, class_weight='balanced', n_jobs=1 ) ) ] )
	et.set_params(**et_params )
        #classifiers = [Pipeline([('anova', StandardScaler() ), ('lr', LogisticRegression(**logistic_params))]), Pipeline([('anova', StandardScaler() ), ('lr',RandomForestClassifier( **random_forest)) ] ), Pipeline([('anova', StandardScaler() ), ('lr',xgb.XGBClassifier(**gb) ) ]) ,]
	lr = Pipeline([('anova', StandardScaler() ), ('lr', LogisticRegression(**logistic_params))])
	classifiers = [lr, svp, et ]
	
	X_test, y_test = get_data(2, 4, test=True)
	X, y = get_data(2, 4, )
	
	train_length = len(X)
	X, y = X + X_test, y+y_test
	#X, X_test, y, y_test = train_test_split( X+X_test, y+y_test, test_size=0.3, random_state=42)
	#X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in np.array( mystep_1.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)
	
	#X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, shuffle=True)	
	#X_train, X_test, y_train, y_test = train_test_split( X_train+X_test, y_train+y_test, test_size=0.2, random_state=42, shuffle=True)
	#preds_ = np.array( mystep_1.get_predictions(X_test) )
	#X_test = np.concatenate((X_test, preds_), axis=1)
	X_train, y_train = X[:train_length], y[:train_length]
	X_test, y_test = X[train_length:], y[train_length:]	
	
	#X_tr, X_te, y_tr, y_te = train_test_split( X_train, y_train, test_size=0.2, random_state=42, shuffle=True, stratify=y_train)
	#X_train = X_tr + X_te
	#y_train = y_tr + y_te
	mystep_2 = Step(classifiers= classifiers, X = X_train, y = y_train, )
	mystep_2.train(X_train, y_train)
	preds = np.array( mystep_2.get_predictions(X_test) )
	report(y_test, [ [a] for a in preds ] )
	with open(file_name + '2_4.pickle', 'wb') as handle:
		joblib.dump(mystep_2, handle, protocol=pickle.HIGHEST_PROTOCOL)
	#step 3
	logistic_params = {'C': 100, 'class_weight': 'balanced', 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1, 'max_iter': 100, 'multi_class': 'ovr', 'penalty': 'l2', 'random_state': 100, 'solver': 'liblinear', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}
	random_forest = {'bootstrap': True, 'class_weight': 'balanced', 'criterion': 'gini', 'max_depth': 40, 'max_features': 'auto', 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 5000, 'n_jobs': -1, 'oob_score': False, 'random_state': 100, 'verbose': 0, 'warm_start': False}
	gb ={'criterion': 'friedman_mse', 'init': None, 'learning_rate': 0.05, 'loss': 'deviance', 'max_depth': 8, 'max_features': None, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 20, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 2000, 'presort': 'auto', 'random_state': 100, 'subsample': 1.0, 'verbose': 0, 'warm_start': False}
	sgd = {'alpha': 0.001, 'average': False, 'class_weight': 'balanced', 'epsilon': 0.1, 'eta0': 0.0, 'fit_intercept': True, 'l1_ratio': 0.15, 'learning_rate': 'optimal', 'loss': 'log', 'max_iter': None, 'n_jobs': -1, 'penalty': 'l2', 'power_t': 0.5, 'random_state': 100, 'shuffle': True, 'tol': None, 'verbose': 0, 'warm_start': False}
	#classifiers = [Pipeline([('anova', StandardScaler() ), ('lr', LogisticRegression(**logistic_params))]), Pipeline([('anova', StandardScaler() ), ('lr',RandomForestClassifier( **random_forest)) ] ), Pipeline([('anova', StandardScaler() ), ('lr',xgb.XGBClassifier(**gb) ) ]) ,]
	svc_params = {"sample__gamma": 0.0078125, "sample__n_components": 2000, "sgd__C": 32.0}
	svp = Pipeline([('scale', StandardScaler() ) , ('sample', Nystroem(random_state=100) ), ('sgd', LogisticRegression(class_weight='balanced', max_iter=1000, n_jobs=1) ) ])
	svp.set_params(**svc_params)
	et_params = {"lr__max_depth": 35, "lr__min_samples_split": 50, "lr__n_estimators": 6500}
	et =  Pipeline([('anova', StandardScaler()), ('lr',ExtraTreesClassifier(random_state=100, class_weight='balanced', n_jobs=1 ) ) ] )
	et.set_params(**et_params )
        #classifiers = [Pipeline([('anova', StandardScaler() ), ('lr', LogisticRegression(**logistic_params))]), Pipeline([('anova', StandardScaler() ), ('lr',RandomForestClassifier( **random_forest)) ] ), Pipeline([('anova', StandardScaler() ), ('lr',xgb.XGBClassifier(**gb) ) ]) ,
	lr = Pipeline([('anova', StandardScaler() ), ('lr', LogisticRegression(**logistic_params))])
	classifiers = [lr, svp, et ]

	X_test, y_test = get_data(3, 4, test=True)
	X, y = get_data(3, 4,)
	train_length = len(X)
	X, y = X + X_test, y+y_test
	#X, X_test, y, y_test = train_test_split( X+X_test, y+y_test, test_size=0.3, random_state=42)
	#X, y = X + X_test, y+y_test
	my_pred_1 = [ [a] for a in mystep_1.get_predictions(X) ]
	X = np.concatenate((X, my_pred_1), axis=1)
	#X, X_test, y, y_test = train_test_split( X, y, test_size=0.3, random_state=42)
	
	preds = [[a] for a in np.array( mystep_2.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)
	
	#X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, shuffle=True)
	X_train, y_train = X[:train_length], y[:train_length]
	X_test, y_test = X[train_length:], y[train_length:]
	
	#X_tr, X_te, y_tr, y_te = train_test_split( X_train, y_train, test_size=0.2, random_state=42, shuffle=True, stratify=y_train)
	#X_train = X_tr + X_te
	#y_train = y_tr + y_te
	#X_train, X_test, y_train, y_test = train_test_split( X+X_test, y+y_test, test_size=0.2, random_state=42, shuffle=True)
	mystep_3 = Step(classifiers= classifiers, X = X, y = y, )
	mystep_3.train(X_train, y_train)
	preds = np.array( mystep_3.get_predictions(X_test) )

	report(y_test, [ [a] for a in preds ] )
	with open(file_name + '3_4.pickle', 'wb') as handle:
		joblib.dump(mystep_3, handle,)# protocol=pickle.HIGHEST_PROTOCOL)
	#step 4
	logistic_params = {'C': 10, 'class_weight': 'balanced', 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1, 'max_iter': 100, 'multi_class': 'ovr',  'penalty': 'l2', 'random_state': 100, 'solver': 'liblinear', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}
	random_forest = {'bootstrap': True, 'class_weight': 'balanced', 'criterion': 'gini', 'max_depth': 40, 'max_features': 'auto', 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 6000, 'n_jobs': -1, 'oob_score': False, 'random_state': 100, 'verbose': 0, 'warm_start': False}
	gb ={'criterion': 'friedman_mse', 'init': None, 'learning_rate': 0.05, 'loss': 'deviance', 'max_depth': 8, 'max_features': None, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 20, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 2500, 'presort': 'auto', 'random_state': 100, 'subsample': 1.0, 'verbose': 0, 'warm_start': False}
	sgd = {'alpha': 0.001, 'average': False, 'class_weight': 'balanced', 'epsilon': 0.1, 'eta0': 0.0, 'fit_intercept': True, 'l1_ratio': 0.15, 'learning_rate': 'optimal', 'loss': 'log', 'max_iter': None, 'n_jobs': -1, 'penalty': 'l2', 'power_t': 0.5, 'random_state': 100, 'shuffle': True, 'tol': None, 'verbose': 0, 'warm_start': False}
	#classifiers = [Pipeline([('anova', StandardScaler() ), ('lr', LogisticRegression(**logistic_params))]), Pipeline([('anova', StandardScaler() ), ('lr',RandomForestClassifier( **random_forest)) ] ), Pipeline([('anova', StandardScaler() ), ('lr',xgb.XGBClassifier(**gb) ) ]) ,]
	svc_params = {"sample__gamma": 0.0078125, "sample__n_components": 2000, "sgd__C": 2048.0}
	svp = Pipeline([('scale', StandardScaler() ) , ('sample', Nystroem(random_state=100) ), ('sgd', LogisticRegression(class_weight='balanced', max_iter=1000, n_jobs=1) ) ])
	svp.set_params(**svc_params)
	et_params = {"lr__max_depth": 40, "lr__min_samples_split": 40, "lr__n_estimators": 5500}
	et =  Pipeline([('anova', StandardScaler()), ('lr',ExtraTreesClassifier(random_state=100, class_weight='balanced' , n_jobs=1 ) ) ] )
	et.set_params(**et_params )
	lr = Pipeline([('anova', StandardScaler() ), ('lr', LogisticRegression(**logistic_params))])
	classifiers = [lr, svp, et ]

	X_test, y_test = get_data(4, 4, test=True)
	#step 1
	X, y = get_data(4, 4,)
	#X, X_test, y, y_test = train_test_split( X+X_test, y+y_test, test_size=0.3, random_state=42)
	train_length = len(X)
	X, y = X + X_test, y+y_test
	
	my_pred_1 =[[a] for a in  mystep_1.get_predictions(X) ]
	X = np.concatenate((X, my_pred_1), axis=1)
	#step 2
	preds = [[a] for a in np.array( mystep_2.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)
	#step 3
	preds = [[a] for a in np.array( mystep_3.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)
	
	X_train, y_train = X[:train_length], y[:train_length]
	X_test, y_test = X[train_length:], y[train_length:]
	
	#X_tr, X_te, y_tr, y_te = train_test_split( X_train, y_train, test_size=0.2, random_state=42, shuffle=True, stratify=y_train)
	#X_train = X_tr + X_te
	#y_train = y_tr + y_te
	#X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, shuffle=True)
	#X_train, X_test, y_train, y_test = train_test_split( X+X_test, y+y_test, test_size=0.2, random_state=42, shuffle=True)
	mystep_4 = Step(classifiers= classifiers, X = X, y = y, )
	mystep_4.train(X_train, y_train)
	preds = np.array( mystep_4.get_predictions(X_test) )

	report(y_test, [ [ a] for a in preds ] )
	with open(file_name + '4_4.pickle', 'wb') as handle:
		joblib.dump(mystep_4, handle,)# protocol=pickle.HIGHEST_PROTOCOL)
	#step 5
	logistic_params ={'C': 1, 'class_weight': 'balanced', 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1, 'max_iter': 100, 'multi_class': 'ovr', 'penalty': 'l2', 'random_state': 100, 'solver': 'liblinear', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}
	random_forest = {'bootstrap': True, 'class_weight': 'balanced', 'criterion': 'gini', 'max_depth': 40, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 4000, 'n_jobs': -1, 'oob_score': False, 'random_state': 100, 'verbose': 0, 'warm_start': False}
	gb ={'criterion': 'friedman_mse', 'init': None, 'learning_rate': 0.05, 'loss': 'deviance', 'max_depth': 8, 'max_features': None, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 20, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 2500, 'presort': 'auto', 'random_state': 100, 'subsample': 1.0, 'verbose': 0, 'warm_start': False}
	sgd = {'alpha': 0.001, 'average': False, 'class_weight': 'balanced', 'epsilon': 0.1, 'eta0': 0.0, 'fit_intercept': True, 'l1_ratio': 0.15, 'learning_rate': 'optimal', 'loss': 'log', 'max_iter': None, 'n_jobs': -1, 'penalty': 'l2', 'power_t': 0.5, 'random_state': 100, 'shuffle': True, 'tol': None, 'verbose': 0, 'warm_start': False}
	#classifiers = [Pipeline([('anova', StandardScaler() ), ('lr', LogisticRegression(**logistic_params))]), Pipeline([('anova', StandardScaler() ), ('lr',RandomForestClassifier( **random_forest)) ] ), Pipeline([('anova', StandardScaler() ), ('lr',xgb.XGBClassifier(**gb) ) ]) ,]
	svc_params = {"sample__gamma": 0.0078125, "sample__n_components": 2000, "sgd__C": 32768.0}
	svp = Pipeline([('scale', StandardScaler() ) , ('sample', Nystroem(random_state=100) ), ('sgd', LogisticRegression(class_weight='balanced', max_iter=1000, n_jobs=1) ) ])
	svp.set_params(**svc_params)
	et_params = {"lr__max_depth": 30, "lr__min_samples_split": 40, "lr__n_estimators": 6500}
	et =  Pipeline([('anova', StandardScaler()), ('lr',ExtraTreesClassifier(random_state=100, class_weight='balanced', n_jobs=1  ) ) ] )
	et.set_params(**et_params )
	lr = Pipeline([('anova', StandardScaler() ), ('lr', LogisticRegression(**logistic_params))])
	classifiers = [lr, svp, et ]

	X_test, y_test = get_data(5, 4, test=True)
	X, y = get_data(5, 4, )
	train_length = len(X)
	X, y = X + X_test, y+y_test
	#X, X_test, y, y_test = train_test_split( X+X_test, y+y_test, test_size=0.3, random_state=42)

	my_pred_1 = [[a] for a in mystep_1.get_predictions(X) ]
	X = np.concatenate((X, my_pred_1), axis=1)
	
	preds = [ [a] for a in np.array( mystep_2.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)

	preds = [[a] for a in  np.array( mystep_3.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)
	
	preds = [[a] for a in np.array( mystep_4.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)
	#preds = np.array( mystep_4.get_predictions(X) )
	#X = np.concatenate((X, preds), axis=1)

	X_train, y_train = X[:train_length], y[:train_length]
	X_test, y_test = X[train_length:], y[train_length:]
	
	#X_tr, X_te, y_tr, y_te = train_test_split( X_train, y_train, test_size=0.2, random_state=42, shuffle=True, stratify=y_train)
	#X_train = X_tr + X_te
	#y_train = y_tr + y_te
	#X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, shuffle=True)
	#X_train, X_test, y_train, y_test = train_test_split( X+X_test, y+y_test, test_size=0.2, random_state=42, shuffle=True)
	mystep_5 = Step(classifiers= classifiers, X = X, y = y, )
	mystep_5.train(X_train, y_train)
	preds = np.array( mystep_5.get_predictions(X_test) )

	report(y_test, [ [a] for a in preds ] )
	with open(file_name+ '5_4.pickle', 'wb') as handle:
		joblib.dump(mystep_5, handle, )#protocol=pickle.HIGHEST_PROTOCOL)
	#step 6
	logistic_params = {'C': 1, 'class_weight': 'balanced', 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1, 'max_iter': 100, 'multi_class': 'ovr', 'penalty': 'l2', 'random_state': 100, 'solver': 'liblinear', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}
	random_forest = {'bootstrap': True, 'class_weight': 'balanced', 'criterion': 'gini', 'max_depth': None, 'max_features': 'auto', 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 6000, 'n_jobs': -1, 'oob_score': False, 'random_state': 100, 'verbose': 0, 'warm_start': False}
	gb ={'criterion': 'friedman_mse', 'init': None, 'learning_rate': 0.1, 'loss': 'deviance', 'max_depth': 8, 'max_features': None, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 20, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 2500, 'presort': 'auto', 'random_state': 100, 'subsample': 1.0, 'verbose': 0, 'warm_start': False}
	sgd = {'alpha': 0.001, 'average': False, 'class_weight': 'balanced', 'epsilon': 0.1, 'eta0': 0.0, 'fit_intercept': True, 'l1_ratio': 0.15, 'learning_rate': 'optimal', 'loss': 'log', 'max_iter': None, 'n_jobs': -1, 'penalty': 'l2', 'power_t': 0.5, 'random_state': 100, 'shuffle': True, 'tol': None, 'verbose': 0, 'warm_start': False}
	#classifiers = [Pipeline([('anova', StandardScaler() ), ('lr', LogisticRegression(**logistic_params))]), Pipeline([('anova', StandardScaler() ), ('lr',RandomForestClassifier( **random_forest)) ] ), Pipeline([('anova', StandardScaler() ), ('lr',xgb.XGBClassifier(**gb) ) ]) ,]
	svc_params = {"sample__gamma": 0.0078125, "sample__n_components": 2000, "sgd__C": 32768.0}
	svp = Pipeline([('scale', StandardScaler() ) , ('sample', Nystroem(random_state=100) ), ('sgd', LogisticRegression(class_weight='balanced', max_iter=1000, n_jobs=1) ) ])
	svp.set_params(**svc_params)
	et_params = {"lr__max_depth": 30, "lr__min_samples_split": 40, "lr__n_estimators": 6500}
	et =  Pipeline([('anova', StandardScaler()), ('lr',ExtraTreesClassifier(random_state=100, class_weight='balanced', n_jobs=1  ) ) ] )
	lr = Pipeline([('anova', StandardScaler() ), ('lr', LogisticRegression(**logistic_params))])
	classifiers = [lr, svp, et ]

	X_test, y_test = get_data(6, 4,test=True)
	X, y = get_data(6, 4, )
	train_length = len(X)
	#X, X_test, y, y_test = train_test_split( X+X_test, y+y_test, test_size=0.3, random_state=42)
	X, y = X + X_test, y+y_test

	my_pred_1 = [[a] for a in mystep_1.get_predictions(X) ]
	X = np.concatenate((X, my_pred_1), axis=1)
	
	preds = [[a] for a in np.array( mystep_2.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in np.array( mystep_3.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)
	
	preds = [[a] for a in np.array( mystep_4.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)
	
	preds = [[a] for a in np.array( mystep_5.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)

	X_train, y_train = X[:train_length], y[:train_length]
	X_test, y_test = X[train_length:], y[train_length:]
	#X_tr, X_te, y_tr, y_te = train_test_split( X_train, y_train, test_size=0.2, random_state=42, shuffle=True, stratify=y_train)
	#X_train = X_tr + X_te
	#y_train = y_tr + y_te
	#X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, shuffle=True)
	#X_train, X_test, y_train, y_test = train_test_split( X+X_test, y+y_test, test_size=0.2, random_state=42, shuffle=True)
	mystep_6 = Step(classifiers= classifiers, X = X, y = y, )
	mystep_6.train(X_train, y_train)
	preds = np.array( mystep_6.get_predictions(X_test) )

	report(y_test, [ [a] for a in preds ] )
	with open(file_name + '6_4.pickle', 'wb') as handle:
		joblib.dump(mystep_6, handle, )#protocol=pickle.HIGHEST_PROTOCOL)
	#step 7
	logistic_params = {'C': 10, 'class_weight': 'balanced', 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1, 'max_iter': 100, 'multi_class': 'ovr','penalty': 'l2', 'random_state': 100, 'solver': 'liblinear', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}
	random_forest = {'bootstrap': True, 'class_weight': 'balanced', 'criterion': 'gini', 'max_depth': None, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 3500, 'n_jobs': -1, 'oob_score': False, 'random_state': 100, 'verbose': 0, 'warm_start': False}
	gb ={'criterion': 'friedman_mse', 'init': None, 'learning_rate': 0.05, 'loss': 'deviance', 'max_depth': 8, 'max_features': None, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 20, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 2500, 'presort': 'auto', 'random_state': 100, 'subsample': 1.0, 'verbose': 0, 'warm_start': False}
	sgd = {'alpha': 0.001, 'average': False, 'class_weight': 'balanced', 'epsilon': 0.1, 'eta0': 0.0, 'fit_intercept': True, 'l1_ratio': 0.15, 'learning_rate': 'optimal', 'loss': 'log', 'max_iter': None, 'n_jobs': -1, 'penalty': 'l2', 'power_t': 0.5, 'random_state': 100, 'shuffle': True, 'tol': None, 'verbose': 0, 'warm_start': False}
	#classifiers = [Pipeline([('anova', StandardScaler() ), ('lr', LogisticRegression(**logistic_params))]), Pipeline([('anova', StandardScaler() ), ('lr',RandomForestClassifier( **random_forest)) ] ), Pipeline([('anova', StandardScaler() ), ('lr',xgb.XGBClassifier(**gb) ) ]) ,]
	svc_params = {"sample__gamma": 0.0078125, "sample__n_components": 2000, "sgd__C": 512.0}
	svp = Pipeline([('scale', StandardScaler() ) , ('sample', Nystroem(random_state=100) ), ('sgd', LogisticRegression(class_weight='balanced', max_iter=1000, n_jobs=1) ) ])
	svp.set_params(**svc_params)
	et_params = {"lr__max_depth": 40, "lr__min_samples_split": 40, "lr__n_estimators": 6500}
	et =  Pipeline([('anova', StandardScaler()), ('lr',ExtraTreesClassifier(random_state=100, class_weight='balanced', n_jobs=1  ) ) ] )
	et.set_params(**et_params )
	lr = Pipeline([('anova', StandardScaler() ), ('lr', LogisticRegression(**logistic_params))])
	classifiers = [lr, svp, et ]

	X_test, y_test = get_data(7, 4, test=True)
	X, y = get_data(7, 4, )
	train_length = len(X)
	#X, X_test, y, y_test = train_test_split( X+X_test, y+y_test, test_size=0.3, random_state=42)
	X, y = X + X_test, y+y_test
	my_pred_1 = [[a] for a in mystep_1.get_predictions(X) ]
	X = np.concatenate((X, my_pred_1), axis=1)
	preds = [[a] for a in np.array( mystep_2.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in np.array( mystep_3.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in np.array( mystep_4.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in np.array( mystep_5.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in np.array( mystep_6.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)

	X_train, y_train = X[:train_length], y[:train_length]
	X_test, y_test = X[train_length:], y[train_length:]
	
	#X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, shuffle=True)
	#X_train, X_test, y_train, y_test = train_test_split( X+X_test, y+y_test, test_size=0.2, random_state=42, shuffle=True)
	mystep_7 = Step(classifiers= classifiers, X = X, y = y, )
	mystep_7.train(X_train, y_train)
	preds = np.array( mystep_7.get_predictions(X_test) )

	report(y_test, [ [a] for a in preds ] )
	with open(file_name + '7_4.pickle', 'wb') as handle:
		joblib.dump(mystep_7, handle, )#protocol=pickle.HIGHEST_PROTOCOL)
	#step 8
	logistic_params = {'C': 0.1, 'class_weight': 'balanced', 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1, 'max_iter': 100, 'multi_class': 'ovr', 'penalty': 'l2', 'random_state': 100, 'solver': 'liblinear', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}
	random_forest = {'bootstrap': True, 'class_weight': 'balanced', 'criterion': 'gini', 'max_depth': 40, 'max_features': 'auto', 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 6000, 'n_jobs': -1, 'oob_score': False, 'random_state': 100, 'verbose': 0, 'warm_start': False}
	gb ={'criterion': 'friedman_mse', 'init': None, 'learning_rate': 0.05, 'loss': 'deviance', 'max_depth': 8, 'max_features': None, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 20, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 3000, 'presort': 'auto', 'random_state': 100, 'subsample': 1.0, 'verbose': 0, 'warm_start': False}
	sgd = {'alpha': 0.001, 'average': False, 'class_weight': 'balanced', 'epsilon': 0.1, 'eta0': 0.0, 'fit_intercept': True, 'l1_ratio': 0.15, 'learning_rate': 'optimal', 'loss': 'log', 'max_iter': None,  'n_jobs': -1, 'penalty': 'l2', 'power_t': 0.5, 'random_state': 100, 'shuffle': True, 'tol': None, 'verbose': 0, 'warm_start': False}

	#classifiers = [Pipeline([('anova', StandardScaler() ), ('lr', LogisticRegression(**logistic_params))]), Pipeline([('anova', StandardScaler() ), ('lr',RandomForestClassifier( **random_forest)) ] ), Pipeline([('anova', StandardScaler() ), ('lr',xgb.XGBClassifier(**gb) ) ]) ,]
	svc_params = {"sample__gamma": 0.0078125, "sample__n_components": 2000, "sgd__C": 32768.0}
	svp = Pipeline([('scale', StandardScaler() ) , ('sample', Nystroem(random_state=100) ), ('sgd', LogisticRegression(class_weight='balanced', max_iter=1000, n_jobs=1) ) ])
	svp.set_params(**svc_params)
	et_params =  {"lr__max_depth": 35, "lr__min_samples_split": 40, "lr__n_estimators": 6500}
	et =  Pipeline([('anova', StandardScaler()), ('lr',ExtraTreesClassifier(random_state=100, class_weight='balanced', n_jobs=1  ) ) ] )
	et.set_params(**et_params )
	lr = Pipeline([('anova', StandardScaler() ), ('lr', LogisticRegression(**logistic_params))])
	classifiers = [lr, svp, et ]

	X_test, y_test = get_data(8, 4,test=True)
	X, y = get_data(8, 4, )
	train_length = len(X)
	#X, X_test, y, y_test = train_test_split( X+X_test, y+y_test, test_size=0.3, random_state=42)
	X, y = X + X_test, y+y_test


	my_pred_1 = [[a] for a in mystep_1.get_predictions(X) ]
	X = np.concatenate((X, my_pred_1), axis=1)
	preds = [[a] for a in np.array( mystep_2.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1) 
	preds = [[a] for a in np.array( mystep_3.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in np.array( mystep_4.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in np.array( mystep_5.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in np.array( mystep_6.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in np.array( mystep_7.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)

	X_train, y_train = X[:train_length], y[:train_length]
	X_test, y_test = X[train_length:], y[train_length:]
	
	#X_tr, X_te, y_tr, y_te = train_test_split( X_train, y_train, test_size=0.2, random_state=42, shuffle=True, stratify=y_train)
	#X_train = X_tr + X_te
	#y_train = y_tr + y_te
	#X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, shuffle=True)
	#X_train, X_test, y_train, y_test = train_test_split( X+X_test, y+y_test, test_size=0.2, random_state=42, shuffle=True)
	mystep_8 = Step(classifiers= classifiers, X = X, y = y, )
	mystep_8.train(X_train, y_train)

	preds = np.array( mystep_8.get_predictions(X_test) )

	report(y_test, [ [a] for a in preds ] )
	with open(file_name + '8_4.pickle', 'wb') as handle:

		joblib.dump(mystep_8, handle,)# protocol=pickle.HIGHEST_PROTOCOL)
	#step 9
	logistic_params = {'C': 100, 'class_weight': 'balanced', 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1, 'max_iter': 100, 'multi_class': 'ovr', 'penalty': 'l2', 'random_state': 100, 'solver': 'liblinear', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}
	random_forest = {'bootstrap': True, 'class_weight': 'balanced', 'criterion': 'gini', 'max_depth': None, 'max_features': 'auto', 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 3500, 'n_jobs': -1, 'oob_score': False, 'random_state': 100, 'verbose': 0, 'warm_start': False}
	gb ={'criterion': 'friedman_mse', 'init': None, 'learning_rate': 0.05, 'loss': 'deviance', 'max_depth': 8, 'max_features': None, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 20, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 3000, 'presort': 'auto', 'random_state': 100, 'subsample': 1.0, 'verbose': 0, 'warm_start': False}
	sgd = {'alpha': 0.001, 'average': False, 'class_weight': 'balanced', 'epsilon': 0.1, 'eta0': 0.0, 'fit_intercept': True, 'l1_ratio': 0.15, 'learning_rate': 'optimal', 'loss': 'log', 'max_iter': None,  'n_jobs': -1, 'penalty': 'l2', 'power_t': 0.5, 'random_state': 100, 'shuffle': True, 'tol': None, 'verbose': 0, 'warm_start': False}
	#classifiers = [Pipeline([('anova', StandardScaler() ), ('lr', LogisticRegression(**logistic_params))]), Pipeline([('anova', StandardScaler() ), ('lr',RandomForestClassifier( **random_forest)) ] ), Pipeline([('anova', StandardScaler() ), ('lr',xgb.XGBClassifier(**gb) ) ]) ,]
	svc_params = {"sample__gamma": 0.0078125, "sample__n_components": 2000, "sgd__C": 32768.0}
	svp = Pipeline([('scale', StandardScaler() ) , ('sample', Nystroem(random_state=100) ), ('sgd', LogisticRegression(class_weight='balanced', max_iter=1000, n_jobs=1) ) ])
	svp.set_params(**svc_params)
	et_params = {"lr__max_depth": 35, "lr__min_samples_split": 40, "lr__n_estimators": 6000}
	et =  Pipeline([('anova', StandardScaler()), ('lr',ExtraTreesClassifier(random_state=100, class_weight='balanced', n_jobs=1  ) ) ] )
	et.set_params(**et_params )
	lr = Pipeline([('anova', StandardScaler() ), ('lr', LogisticRegression(**logistic_params))])
	classifiers = [lr, svp, et ]

	X_test, y_test = get_data(9, 4, test=True)
	X, y = get_data(9, 4, )
	train_length = len(X)
	#X, X_test, y, y_test = train_test_split( X+X_test, y+y_test, test_size=0.3, random_state=42)
	X, y = X + X_test, y+y_test

	preds = [[a] for a in mystep_1.get_predictions(X) ]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in np.array( mystep_2.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in np.array( mystep_3.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in np.array( mystep_4.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in np.array( mystep_5.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in np.array( mystep_6.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in np.array( mystep_7.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in np.array( mystep_8.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)

	X_train, y_train = X[:train_length], y[:train_length]
	X_test, y_test = X[train_length:], y[train_length:]

	#X_tr, X_te, y_tr, y_te = train_test_split( X_train, y_train, test_size=0.2, random_state=42, shuffle=True, stratify=y_train)
	#X_train = X_tr + X_te
	#y_train = y_tr + y_te	
	#X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, shuffle=True)
	#X_train, X_test, y_train, y_test = train_test_split( X+X_test, y+y_test, test_size=0.2, random_state=42, shuffle=True)
	mystep_9 = Step(classifiers= classifiers, X = X, y = y, )
	mystep_9.train(X_train, y_train)

	preds = np.array( mystep_9.get_predictions(X_test) )

	report(y_test, [ [a] for a in preds ] )
	with open(file_name + '9_4.pickle', 'wb') as handle:
		joblib.dump(mystep_9, handle, )#protocol=pickle.HIGHEST_PROTOCOL)
	#step 10
	logistic_params = {'C': 1, 'class_weight': 'balanced', 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1, 'max_iter': 100, 'multi_class': 'ovr',  'penalty': 'l2', 'random_state': 100, 'solver': 'liblinear', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}
	random_forest = {'bootstrap': True, 'class_weight': 'balanced', 'criterion': 'gini', 'max_depth': None, 'max_features': 'auto', 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 3500, 'n_jobs': -1, 'oob_score': False, 'random_state': 100, 'verbose': 0, 'warm_start': False}
	gb ={'criterion': 'friedman_mse', 'init': None, 'learning_rate': 0.05, 'loss': 'deviance', 'max_depth': 8, 'max_features': None, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 20, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 3000, 'presort': 'auto', 'random_state': 100, 'subsample': 1.0, 'verbose': 0, 'warm_start': False}
	sgd = {'alpha': 0.001, 'average': False, 'class_weight': 'balanced', 'epsilon': 0.1, 'eta0': 0.0, 'fit_intercept': True, 'l1_ratio': 0.15, 'learning_rate': 'optimal', 'loss': 'log', 'max_iter': None,  'penalty': 'l2', 'power_t': 0.5, 'random_state': 100, 'shuffle': True, 'tol': None, 'verbose': 0, 'warm_start': False}

	#classifiers = [Pipeline([('anova', StandardScaler() ), ('lr', LogisticRegression(**logistic_params))]), Pipeline([('anova', StandardScaler() ), ('lr',RandomForestClassifier( **random_forest)) ] ), Pipeline([('anova', StandardScaler() ), ('lr',xgb.XGBClassifier(**gb) ) ]) ,]
	svc_params = {"sample__gamma": 0.0078125, "sample__n_components": 2000, "sgd__C": 8192.0}
	svp = Pipeline([('scale', StandardScaler() ) , ('sample', Nystroem(random_state=100) ), ('sgd', LogisticRegression(class_weight='balanced', max_iter=1000, n_jobs=1) ) ])
	svp.set_params(**svc_params)
	et_params = {"lr__max_depth": 35, "lr__min_samples_split": 40, "lr__n_estimators": 6500}
	et =  Pipeline([('anova', StandardScaler()), ('lr',ExtraTreesClassifier(random_state=100, class_weight='balanced', n_jobs=1  ) ) ] )
	et.set_params(**et_params )
	lr = Pipeline([('anova', StandardScaler() ), ('lr', LogisticRegression(**logistic_params))])
	classifiers = [lr, svp, et ]

	X_test, y_test = get_data(10, 4, test=True)
	X, y = get_data(10, 4, )
	train_length = len(X)
	
	#X, X_test, y, y_test = train_test_split( X+X_test, y+y_test, test_size=0.3, random_state=42)
	X, y = X + X_test, y+y_test


	preds =[[a] for a in mystep_1.get_predictions(X) ]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in mystep_2.get_predictions(X) ]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in mystep_3.get_predictions(X) ]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in mystep_4.get_predictions(X) ]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in np.array( mystep_5.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in np.array( mystep_6.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in np.array( mystep_7.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in np.array( mystep_8.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in np.array( mystep_9.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)

	X_train, y_train = X[:train_length], y[:train_length]
	X_test, y_test = X[train_length:], y[train_length:]

	#X_tr, X_te, y_tr, y_te = train_test_split( X_train, y_train, test_size=0.2, random_state=42, shuffle=True, stratify=y_train)
	#X_train = X_tr + X_te
	#y_train = y_tr + y_te	
	#X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, shuffle=True)
	#X_train, X_test, y_train, y_test = train_test_split( X+X_test, y+y_test, test_size=0.2, random_state=42, shuffle=True)
	mystep_10 = Step(classifiers= classifiers, X = X, y = y, )
	mystep_10.train(X_train, y_train)
	preds = np.array( mystep_10.get_predictions(X_test) )

	report(y_test, [ [a] for a in preds ] )
	with open(file_name + '10_4.pickle', 'wb') as handle:
		joblib.dump(mystep_10, handle, )#protocol=pickle.HIGHEST_PROTOCOL)
	#step 11
	logistic_params = {'C': 1, 'class_weight': 'balanced', 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1, 'max_iter': 100, 'multi_class': 'ovr',  'penalty': 'l2', 'random_state': 100, 'solver': 'liblinear', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}
	random_forest = {'bootstrap': True, 'class_weight': 'balanced', 'criterion': 'gini', 'max_depth': None, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 5000, 'n_jobs': -1, 'oob_score': False, 'random_state': 100, 'verbose': 0, 'warm_start': False}
	gb ={'criterion': 'friedman_mse', 'init': None, 'learning_rate': 0.05, 'loss': 'deviance', 'max_depth': 8, 'max_features': None, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 20, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 2500, 'presort': 'auto', 'random_state': 100, 'subsample': 1.0, 'verbose': 0, 'warm_start': False}
	sgd = {'alpha': 0.001, 'average': False, 'class_weight': 'balanced', 'epsilon': 0.1, 'eta0': 0.0, 'fit_intercept': True, 'l1_ratio': 0.15, 'learning_rate': 'optimal', 'loss': 'log', 'max_iter': None, 'n_jobs': -1, 'penalty': 'l2', 'power_t': 0.5, 'random_state': 100, 'shuffle': True, 'tol': None, 'verbose': 0, 'warm_start': False}

	#classifiers = [Pipeline([('anova', StandardScaler() ), ('lr', LogisticRegression(**logistic_params))]), Pipeline([('anova', StandardScaler() ), ('lr',RandomForestClassifier( **random_forest)) ] ), Pipeline([('anova', StandardScaler() ), ('lr',xgb.XGBClassifier(**gb) ) ]) ,]
	svc_params = {"sample__gamma": 0.0078125, "sample__n_components": 2000, "sgd__C": 2048.0}
	svp = Pipeline([('scale', StandardScaler() ) , ('sample', Nystroem(random_state=100) ), ('sgd', LogisticRegression(class_weight='balanced', max_iter=1000, n_jobs=1) ) ])
	svp.set_params(**svc_params)
	et_params =  {"lr__max_depth": 35, "lr__min_samples_split": 40, "lr__n_estimators": 6500} 
	et =  Pipeline([('anova', StandardScaler()), ('lr',ExtraTreesClassifier(random_state=100, class_weight='balanced', n_jobs=1  ) ) ] )
	et.set_params(**et_params )
	lr = Pipeline([('anova', StandardScaler() ), ('lr', LogisticRegression(**logistic_params))])
	classifiers = [lr, svp, et ]

	X_test, y_test = get_data(11, 4, test=True)
	X, y = get_data(11, 4, )
	#X, X_test, y, y_test = train_test_split( X+X_test, y+y_test, test_size=0.3, random_state=42)
	train_length = len(X)
	
	X, y = X + X_test, y+y_test

	preds = [[a] for a in mystep_1.get_predictions(X) ]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in mystep_2.get_predictions(X) ]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in mystep_3.get_predictions(X) ]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in mystep_4.get_predictions(X) ]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in mystep_5.get_predictions(X) ]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in mystep_6.get_predictions(X) ]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in mystep_7.get_predictions(X)]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in mystep_8.get_predictions(X)]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in mystep_9.get_predictions(X) ]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in np.array( mystep_10.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)
	
	X_train, y_train = X[:train_length], y[:train_length]
	X_test, y_test = X[train_length:], y[train_length:]
	
	#X_tr, X_te, y_tr, y_te = train_test_split( X_train, y_train, test_size=0.2, random_state=42, shuffle=True, stratify=y_train)
	#X_train = X_tr + X_te
	#y_train = y_tr + y_te
	#X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, shuffle=True)
	#X_train, X_test, y_train, y_test = train_test_split( X+X_test, y+y_test, test_size=0.2, random_state=42, shuffle=True)
	mystep_11 = Step(classifiers= classifiers, X = X, y = y, )
	mystep_11.train(X_train, y_train)
	preds = np.array( mystep_11.get_predictions(X_test) )

	report(y_test, [ [a] for a in preds ] )
	with open(file_name + '11_4.pickle', 'wb') as handle:
		joblib.dump(mystep_11, handle, )#protocol=pickle.HIGHEST_PROTOCOL)
	#step 12
	logistic_params = {'C': 0.1, 'class_weight': 'balanced', 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1, 'max_iter': 100, 'multi_class': 'ovr',  'penalty': 'l2', 'random_state': 100, 'solver': 'liblinear', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}
	random_forest = {'bootstrap': True, 'class_weight': 'balanced', 'criterion': 'gini', 'max_depth': None, 'max_features': 'auto', 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 3500, 'n_jobs': -1, 'oob_score': False, 'random_state': 100, 'verbose': 0, 'warm_start': False}
	gb ={'criterion': 'friedman_mse', 'init': None, 'learning_rate': 0.05, 'loss': 'deviance', 'max_depth': 8, 'max_features': None, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 20, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 2500, 'presort': 'auto', 'random_state': 100, 'subsample': 1.0, 'verbose': 0, 'warm_start': False}
	sgd = {'alpha': 0.001, 'average': False, 'class_weight': 'balanced', 'epsilon': 0.1, 'eta0': 0.0, 'fit_intercept': True, 'l1_ratio': 0.15, 'learning_rate': 'optimal', 'loss': 'log', 'max_iter': None, 'n_jobs': -1, 'penalty': 'l2', 'power_t': 0.5, 'random_state': 100, 'shuffle': True, 'tol': None, 'verbose': 0, 'warm_start': False}

	svc_params = {"sample__gamma": 0.0078125, "sample__n_components": 2000, "sgd__C": 8192.0}
	svp = Pipeline([('scale', StandardScaler() ) , ('sample', Nystroem(random_state=100) ), ('sgd', LogisticRegression(class_weight='balanced', max_iter=1000, n_jobs=1) ) ])
	svp.set_params(**svc_params)
	et_params =  {"lr__max_depth": 40, "lr__min_samples_split": 40, "lr__n_estimators": 6000}
	et =  Pipeline([('anova', StandardScaler()), ('lr',ExtraTreesClassifier(random_state=100, class_weight='balanced', n_jobs=1  ) ) ] )
	et.set_params(**et_params )
	#classifiers = [Pipeline([('anova', StandardScaler() ), ('lr', LogisticRegression(**logistic_params))]), Pipeline([('anova', StandardScaler() ), ('lr',RandomForestClassifier( **random_forest)) ] ), Pipeline([('anova', StandardScaler() ), ('lr',xgb.XGBClassifier(**gb) ) ]) ,]
	lr = Pipeline([('anova', StandardScaler() ), ('lr', LogisticRegression(**logistic_params))])
	classifiers = [lr, svp, et ]

	X_test, y_test = get_data(12, 4, test=True)
	X, y = get_data(12, 4, )
	train_length = len(X)

	#X, X_test, y, y_test = train_test_split( X+X_test, y+y_test, test_size=0.3, random_state=42)
	X, y = X + X_test, y+y_test

	preds = [[a] for a in np.array( mystep_1.get_predictions(X) )]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in np.array( mystep_2.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in np.array( mystep_3.get_predictions(X) )]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in np.array( mystep_4.get_predictions(X) )]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in np.array( mystep_5.get_predictions(X) )]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in np.array( mystep_6.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in np.array( mystep_7.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in np.array( mystep_8.get_predictions(X) )]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in np.array( mystep_9.get_predictions(X) )]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in np.array( mystep_10.get_predictions(X) )]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in np.array( mystep_11.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)

	X_train, y_train = X[:train_length], y[:train_length]
	X_test, y_test = X[train_length:], y[train_length:]
	
	#X_tr, X_te, y_tr, y_te = train_test_split( X_train, y_train, test_size=0.2, random_state=42, shuffle=True, stratify=y_train)
	#X_train = X_tr + X_te
	#y_train = y_tr + y_te	
	#X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, shuffle=True)
	#X_train, X_test, y_train, y_test = train_test_split( X+X_test, y+y_test, test_size=0.2, random_state=42, shuffle=True)
	mystep_12 = Step(classifiers= classifiers, X = X, y = y,)
	mystep_12.train(X_train, y_train)
	preds = np.array( mystep_12.get_predictions(X_test) )

	report(y_test, [ [a] for a in preds ] )
	with open(file_name + '12_4.pickle', 'wb') as handle:
		joblib.dump(mystep_12, handle, )#protocol=pickle.HIGHEST_PROTOCOL)
	#step 13
	logistic_params = {'C': 1, 'class_weight': 'balanced', 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1, 'max_iter': 100, 'multi_class': 'ovr',  'penalty': 'l2', 'random_state': 100, 'solver': 'liblinear', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}
	random_forest = {'bootstrap': True, 'class_weight': 'balanced', 'criterion': 'gini', 'max_depth': None, 'max_features': 'auto', 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 6000, 'n_jobs': -1, 'oob_score': False, 'random_state': 100, 'verbose': 0, 'warm_start': False}
	gb = {'criterion': 'friedman_mse', 'init': None, 'learning_rate': 0.05, 'loss': 'deviance', 'max_depth': 8, 'max_features': None, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 20, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 2500, 'presort': 'auto', 'random_state': 100, 'subsample': 1.0, 'verbose': 0, 'warm_start': False}
	sgd = {'alpha': 0.0001, 'average': False, 'class_weight': 'balanced', 'epsilon': 0.1, 'eta0': 0.0, 'fit_intercept': True, 'l1_ratio': 0.15, 'learning_rate': 'optimal', 'loss': 'log', 'max_iter': None, 'n_jobs': -1, 'penalty': 'l2', 'power_t': 0.5, 'random_state': 100, 'shuffle': True, 'tol': None, 'verbose': 0, 'warm_start': False}
	
	#classifiers = [Pipeline([('anova', StandardScaler() ), ('lr', LogisticRegression(**logistic_params))]), Pipeline([('anova', StandardScaler() ), ('lr',RandomForestClassifier( **random_forest)) ] ), Pipeline([('anova', StandardScaler() ), ('lr',xgb.XGBClassifier(**gb) ) ]) ,]
	svc_params = {"sample__gamma": 0.0078125, "sample__n_components": 2000, "sgd__C": 512.0}
	svp = Pipeline([('scale', StandardScaler() ) , ('sample', Nystroem(random_state=100) ), ('sgd', LogisticRegression(class_weight='balanced', max_iter=1000, n_jobs=1) ) ])
	svp.set_params(**svc_params)
	et_params = {"lr__max_depth": 35, "lr__min_samples_split": 40, "lr__n_estimators": 6000}
	et =  Pipeline([('anova', StandardScaler()), ('lr',ExtraTreesClassifier(random_state=100, class_weight='balanced', n_jobs=1  ) ) ] )
	et.set_params(**et_params )
	lr = Pipeline([('anova', StandardScaler() ), ('lr', LogisticRegression(**logistic_params))])
	classifiers = [lr, svp, et ]

	X_test, y_test = get_data(13, 4, test=True)
	X, y = get_data(13, 4,)
	train_length = len(X)
	#X, X_test, y, y_test = train_test_split( X+X_test, y+y_test, test_size=0.3, random_state=42)
	X, y = X + X_test, y+y_test
	preds = [[a] for a in np.array( mystep_1.get_predictions(X) )]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in np.array( mystep_2.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in np.array( mystep_3.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in np.array( mystep_4.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in np.array( mystep_5.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in np.array( mystep_6.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in np.array( mystep_7.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in np.array( mystep_8.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in np.array( mystep_9.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in np.array( mystep_10.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in np.array( mystep_11.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in np.array( mystep_12.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)
	
	X_train, y_train = X[:train_length], y[:train_length]
	X_test, y_test = X[train_length:], y[train_length:]
	
	#X_tr, X_te, y_tr, y_te = train_test_split( X_train, y_train, test_size=0.2, random_state=42, shuffle=True, stratify=y_train)
	#X_train = X_tr + X_te
	#y_train = y_tr + y_te
	#X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, shuffle=True)
	#X_train, X_test, y_train, y_test = train_test_split( X+X_test, y+y_test, test_size=0.2, random_state=42, shuffle=True)
	mystep_13 = Step(classifiers= classifiers, X = X, y = y,)
	mystep_13.train(X_train, y_train)
	preds = np.array( mystep_13.get_predictions(X_test) )

	report(y_test, [ [a] for a in preds ] )
	with open(file_name + '13_4.pickle', 'wb') as handle:
		joblib.dump(mystep_13, handle, )#protocol=pickle.HIGHEST_PROTOCOL)
	#step 14
	logistic_params = {'C': 0.01, 'class_weight': 'balanced', 'dual': False, 'fit_intercept': True, 'intercept_scaling': 1, 'max_iter': 100, 'multi_class': 'ovr', 'penalty': 'l2', 'random_state': 100, 'solver': 'liblinear', 'tol': 0.0001, 'verbose': 0, 'warm_start': False}
	random_forest = {'bootstrap': True, 'class_weight': 'balanced', 'criterion': 'gini', 'max_depth': None, 'max_features': 'sqrt', 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 1, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 4000, 'n_jobs': -1, 'oob_score': False, 'random_state': 100, 'verbose': 0, 'warm_start': False}
	gb ={'criterion': 'friedman_mse', 'init': None, 'learning_rate': 0.05, 'loss': 'deviance', 'max_depth': 8, 'max_features': None, 'max_leaf_nodes': None, 'min_impurity_decrease': 0.0, 'min_impurity_split': None, 'min_samples_leaf': 20, 'min_samples_split': 2, 'min_weight_fraction_leaf': 0.0, 'n_estimators': 3000, 'presort': 'auto', 'random_state': 100, 'subsample': 1.0, 'verbose': 0, 'warm_start': False}
	sgd = {'alpha': 0.0001, 'average': False, 'class_weight': 'balanced', 'epsilon': 0.1, 'eta0': 0.0, 'fit_intercept': True, 'l1_ratio': 0.15, 'learning_rate': 'optimal', 'loss': 'log', 'max_iter': None,  'n_jobs': -1, 'penalty': 'l2', 'power_t': 0.5, 'random_state': 100, 'shuffle': True, 'tol': None, 'verbose': 0, 'warm_start': False}
	 
	#classifiers = [Pipeline([('anova', StandardScaler() ), ('lr', LogisticRegression(**logistic_params))]), Pipeline([('anova', StandardScaler() ), ('lr',RandomForestClassifier( **random_forest)) ] ), Pipeline([('anova', StandardScaler() ), ('lr',xgb.XGBClassifier(**gb) ) ]) ,]
	svc_params = {"sample__gamma": 0.0078125, "sample__n_components": 2000, "sgd__C": 32768.0}
	svp = Pipeline([('scale', StandardScaler() ) , ('sample', Nystroem(random_state=100) ), ('sgd', LogisticRegression(class_weight='balanced', max_iter=1000, n_jobs=1) ) ])
	svp.set_params(**svc_params)
	et_params = {"lr__max_depth": 35, "lr__min_samples_split": 40, "lr__n_estimators": 6000}
	et =  Pipeline([('anova', StandardScaler()), ('lr',ExtraTreesClassifier(random_state=100, class_weight='balanced', n_jobs=1  ) ) ] )
	et.set_params(**et_params )
	lr = Pipeline([('anova', StandardScaler() ), ('lr', LogisticRegression(**logistic_params))])
	classifiers = [lr, svp, et ]

	X_test, y_test = get_data(14, 4, test=True)
	X, y = get_data(14, 4, )
	train_length = len(X)
	#X, X_test, y, y_test = train_test_split( X+X_test, y+y_test, test_size=0.3, random_state=42)
	X, y = X + X_test, y+y_test
	preds = [[a] for a in np.array( mystep_1.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in np.array( mystep_2.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in np.array( mystep_3.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in np.array( mystep_4.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in np.array( mystep_5.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in np.array( mystep_6.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in np.array( mystep_7.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in np.array( mystep_8.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in np.array( mystep_9.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in np.array( mystep_10.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in np.array( mystep_11.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in np.array( mystep_12.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)
	preds = [[a] for a in np.array( mystep_13.get_predictions(X) ) ]
	X = np.concatenate((X, preds), axis=1)

	X_train, y_train = X[:train_length], y[:train_length]
	X_test, y_test = X[train_length:], y[train_length:]	
	
	#X_tr, X_te, y_tr, y_te = train_test_split( X_train, y_train, test_size=0.2, random_state=42, shuffle=True, stratify=y_train)
	#X_train = X_tr + X_te
	#y_train = y_tr + y_te
	#X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, shuffle=True)
	#X_train, X_test, y_train, y_test = train_test_split( X+X_test, y+y_test, test_size=0.2, random_state=42, shuffle=True)
	mystep_14 = Step(classifiers= classifiers, X = X, y = y, )
	mystep_14.train(X_train, y_train)
	preds = np.array( mystep_14.get_predictions(X_test) )
	#client.shutdown()
	report(y_test, [ [a] for a in preds ] )
	with open(file_name + '14_4.pickle', 'wb') as handle:
		joblib.dump(mystep_14, handle, )#protocol=pickle.HIGHEST_PROTOCOL)


