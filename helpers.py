from imblearn.under_sampling import *
from pprint import pprint
import numpy as np
import pandas as pd

from sklearn.externals import joblib
from sklearn.model_selection import * #ShuffleSplit
#from sklearn.cross_validation import *
import os
from sklearn.ensemble import *

from sklearn.datasets import load_digits
from pprint import pprint
from subprocess import call
from sklearn.metrics import roc_curve, roc_auc_score, accuracy_score, precision_score, confusion_matrix, recall_score, f1_score, auc, matthews_corrcoef
#from subprocess import call
from sklearn.model_selection import *

import warnings

import numpy as np
import scipy.sparse as sp
from scipy.linalg import svd

from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.utils import check_array, check_random_state, as_float_array
from sklearn.utils.extmath import safe_sparse_dot
from sklearn.utils.validation import check_is_fitted
from sklearn.metrics.pairwise import pairwise_kernels, KERNEL_PARAMS
from sklearn.utils.validation import check_non_negative, _deprecate_positional_args
import warnings
import numbers
import time
from traceback import format_exc
from contextlib import suppress

import scipy.sparse as sp
from joblib import Parallel, delayed

from sklearn.base import is_classifier, clone
from sklearn.utils import (indexable, check_random_state, _safe_indexing,
                     _message_with_time)
from sklearn.utils.validation import _check_fit_params
from sklearn.utils.validation import _num_samples
from sklearn.utils.validation import _deprecate_positional_args
from sklearn.utils.metaestimators import _safe_split
from sklearn.metrics import check_scoring
from sklearn.metrics._scorer import _check_multimetric_scoring, _MultimetricScorer
from sklearn.exceptions import FitFailedWarning
from sklearn.model_selection._split import check_cv
from sklearn.preprocessing import LabelEncoder

#step = sys.argv[1]
#window_ = sys.argv[2]
#paths = os.listdir("buoy_data/test_step" + step +"/window_"+ window_ +"_dataset")
#random.shuffle(paths)
#sc = SparkContext(master='spark://nrl-05.cs.uno.edu:7077', appName='spark_features')

#register_spark() # register spark backend
#register()
chromosomes = "101111110111001100000000110000010010010001100110011000000011001111101001100100010101001000001000010101001010000000011010001100000100000110101110101000010111110110011101110111011101110101001011111001001111000011100111100"
def fit_classifier(klass, param, metric, X_train, Y_train, X_validation, Y_validation):
    classifier = klass.fit(X_train, Y_train)
    Y_predict = classifier.predict(X_validation)
    score = metric(Y_validation, Y_predict)
    return (param, score)

def get_data(step, window_):
    import random
    global chromosomes
    random.seed(100)
    #step = sys.argv[1]
    #window_ = sys.argv[2]
    X_initial, y_initial = [], []
    step = str(step)
    window_= str(window_)
    paths = os.listdir("buoy_data/test_step" + step +"/window_"+ window_ +"_dataset")
    random.shuffle(paths)
    for file in paths[: int(0.1 * len(paths) )]:
        with open("buoy_data/test_step" + step +"/window_"+ window_ +"_dataset/" + file, "r") as f:

            for line in f.readlines():#[:100000]
                        #output = int(line.split(",")[0]
                features = [float(x) for x in line.strip().split(",") if x !='']
                inputs = features[1:]
                        #for x in inputs:
                        #       if np.isnan(x):
                        #               print(x)    
                if len(X_initial) == 0 or len(inputs) == len(X_initial[0] ):
                    X_initial.append(inputs  )
                    y_initial.append( int(features[0]) )
    indices = []
    for x in range(len(chromosomes)):
        if chromosomes[x] == '1':
            #print( chromosomes[x] )
            indices.append(x )
    X_ = []
    for x in X_initial:
        X_.append( np.array( [ x[b] for b in indices ] )  )
    X_initial = X_
    #X, X_test, y, y_test = train_test_split( X_initial, y_initial, test_size=0.2, random_state=1000, shuffle=True, stratify=y_initial)
    #return np.array(X_test), np.array(y_test )
    return np.array(X_initial), np.array(y_initial)


def print_scores(y, predicted):
    print( roc_auc_score(y, predicted) )
    print( 1 - roc_auc_score(y, predicted) )
    predicted = [ a[0]>0.5 for a in predicted ]
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



class Nystroem(TransformerMixin, BaseEstimator):
	def __init__(self, kernel="rbf", *, gamma=None, coef0=None, degree=None,
		kernel_params=None, n_components=100, random_state=None):
		self.kernel = kernel
		self.gamma = gamma
		self.coef0 = coef0
		self.degree = degree
		self.kernel_params = kernel_params
		self.n_components = n_components
		self.random_state = random_state
	def fit(self, X, y=None):
		#X = self._validate_data(X, accept_sparse='csr')
		rnd = check_random_state(self.random_state)
		n_samples = X.shape[0]
		if self.n_components > n_samples:
			n_components = n_samples
			warnings.warn("n_components > n_samples. This is not possible.\n"
				"n_components was set to n_samples, which results"
				" in inefficient evaluation of the full kernel.")

		else:
			n_components = self.n_components
		n_components = min(n_samples, n_components)
		n_folds = int( n_samples / n_components)
		rus = RandomUnderSampler(random_state=42)
		X_res, y_res = rus.fit_resample(X, y)
		skf = StratifiedKFold(n_splits = n_folds)
		_, indices = [a for a in skf.split(X_res, y_res) ] [0]
		basis = X_res[indices]
		basis_kernel = pairwise_kernels(basis, metric=self.kernel,
			filter_params=True,
			**self._get_kernel_params())
		U, S, V = svd(basis_kernel)
		S = np.maximum(S, 1e-12)
		self.normalization_ = np.dot(U / np.sqrt(S), V)
		self.components_ = basis
		self.component_indices_ = indices
		return self

	def transform(self, X):
		check_is_fitted(self)
        
		X = check_array(X, accept_sparse='csr')
		kernel_params = self._get_kernel_params()
		embedded = pairwise_kernels(X, self.components_,
			metric=self.kernel,
			filter_params=True,
			**kernel_params)
		return np.dot(embedded, self.normalization_.T)


	def _get_kernel_params(self):
		params = self.kernel_params
		if params is None:
			params = {}
		if not callable(self.kernel) and self.kernel != 'precomputed':
			for param in (KERNEL_PARAMS[self.kernel]):
				if getattr(self, param) is not None:
					params[param] = getattr(self, param)
		else:
			if (self.gamma is not None or
				self.coef0 is not None or
				self.degree is not None):
				raise ValueError("Don't pass gamma, coef0 or degree to "
					"Nystroem if using a callable "
					"or precomputed kernel")
		return params
	














