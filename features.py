from scipy.stats import kurtosis
from scipy.stats import kstat
from scipy.stats import skew


from scipy.spatial.distance import correlation

import numpy as np
import math
from scipy.signal import *
import pywt
from scipy.spatial import distance
from scipy.signal import butter, lfilter
from joblib import Parallel, delayed

#array is the spectral data
def discrete_fourier_transform(arr):
	return_val = []
	for val in arr:
		#print(val)
		cA, cD = pywt.dwt(val, 'db2', mode='smooth' )
		inputs = cA + cD
		inputs = np.ndarray.flatten( np.array(inputs) )		
		return_val.extend(inputs)
	return return_val
from joblib import Parallel
def covariance_and_correlation(arr):
	arr = np.reshape(arr,(-1,7))
	#arr = [a[1:] for a in arr]
	return_val = []
	for val1 in arr:
		vals = []
		#min_1 = Parallel(n_jobs=-1) (delayed (distance.minkowski) (val1, val2, 1) for val2 in arr)
		#min_2 = Parallel(n_jobs=-1) (delayed (distance.minkowski) (val1, val2, 2) for val2 in arr)
		for val2 in arr:
			#if val1 == val2:
			#	continue
			#to_append = np.ndarray.flatten(np.cov(val1, val2) ).tolist()
			#coors = np.correlate(val1, val2).tolist()
			to_append = []
			#minkowski distance 1 and 2
			min_1 = distance.minkowski(val1, val2, 1)
			min_2 = distance.minkowski(val1, val2, 2)
			#kstat = [kstatp
			#cor = correlation(val1, val2)
			#to_append += coors 
			#to_append.append(min_1)
			#to_append.append(min_2)
			min_d = [float(min_1), float(min_2) ]
			to_append += min_d
			vals.append(to_append)
		return_val.append(np.mean(vals ) )
		#return_val.append(np.mean(vals,))# axis=1 ) )
	return np.mean(return_val, axis=0)
from math import*
from decimal import Decimal
def nth_root(value, n_root):
	root_value = 1/float(n_root)
	return round (Decimal(value) ** Decimal(root_value),3)
def minkowski_distance_(x,y,p_value):
	return nth_root(sum(pow(abs(a-b),p_value) for a,b in zip(x, y)),p_value)


#let p val be 0, 1 or np.inf
def minkowski_distance(arr, p_value):
	return_val = []
	for val1 in arr:
		vals = []
		for val2 in arr:
			#to_append = np.ndarray.flatten(np.cov(val1, val2) )
			to_append = minkowski_distance_(val1, val2, p_value)
			vals.append( float(to_append) )
		return_val.append(vals )
	return np.mean(return_val, axis=1)

def cross_val(arr):
	#forward pass
	arr = np.reshape(arr,(-1,7))
	#arr = [a[1:] for a in arr]
	return_val = []
	vals = []
	for x in range(len(arr)-1 ):
		#vals = []
		#mean_vals = Parallel(n_jobs=-1) (delayed( np.mean) (arr[x:y]  for y in range(x+1, len(arr) ) ) ) 
		#median_vals = Parallel(n_jobs=-1) (delayed( np.median) (arr[x:y]  for y in range(x+1, len(arr) ) ) ) 
		# first one is arr[:x] 
		my_arr = []
		#for y in range(x+1, len(arr)):
		mean_vals = np.mean(arr[0:x+1] , axis = 0)
		std_vals = kurtosis(arr[0:x+1] )
		skews = skew(arr[0:x+1] )
		#	mean_vals = np.mean(arr[x:y] )
		med_vals = np.median(arr[0:x+1] , axis = 0)
			#my_arr = []
		#kstat_ = [ kstat(arr[0:x+1], 1), kstat(arr[0:x+1], 2), kstat(arr[0:x+1], 3), kstat(arr[0:x+1], 4) ]
		my_arr = []
			#med_vals = np.median(arr[x:y] )
		my_arr.append(  mean_vals)
		my_arr.append(med_vals)
		my_arr.append(std_vals)
		#my_arr.append(kstat(arr[0:x+1], 1) )
		#my_arr.append(kstat(arr[0:x+1], 2) )	
		#my_arr.append(kstat(arr[0:x+1], 3) )
		#my_arr.append(kstat(arr[0:x+1], 4) )
		#my_arr.extend(kstat_ )
		my_arr.append(skews)
		#	#my_arr.append(med_vals/mean_vals)
		vals.append( np.ndarray.flatten( np.array(my_arr) ) )
	#return_val.append(np.ndarray.flatten(np.array(vals ) ) )#
	#return_val.append(np.mean( np.array(vals ) , axis = 0) )
	return_val.extend( np.mean(np.array(vals), axis=0  ) )
	return_val.extend( np.mean(np.array(vals), axis=1  ) )
	#print (return_val ) 
	#back pass
	back_arr = arr[::-1]
	vals_ = []
	for x in range(len(arr)-1 ):
		#vals = []
		my_arr = []
		#mean_vals = Parallel(n_jobs=-1) (delayed( np.mean) (arr[x:y]  for y in range(x+1, len(arr) ) ) )
	#	median_vals = Parallel(n_jobs=-1) (delayed( np.median) (arr[x:y]  for y in range(x+1, len(arr) ) ) )
		#for y in range(x+1, len(arr)):
		mean_vals = np.mean(back_arr[0:x+1] , axis = 0)
		std_vals = kurtosis(back_arr[0:x+1] , axis = 0 )
		med_vals = np.median(arr[0: x+1] , axis = 0)
		#kstat_ = [ kstat(arr[0:x+1], 1), kstat(arr[0:x+1], 2), kstat(arr[0:x+1], 3), kstat(arr[0:x+1], 4) ]
		skews = skew(arr[0:x+1] )
		#	mean_vals = np.mean(arr[x:y] )
			#my_arr =[]
			#my_arr = []
	#	#	#my_arr.append(med_vals/ mean_vals)
			#my_arr.append(  mean_vals)
			#med_vals = np.median(arr[x:y] )
		my_arr.append(med_vals)
		my_arr.append(std_vals)
			#my_arr = []
		my_arr.append( mean_vals)
		#my_arr.extend(kstat_ )
		#my_arr.append(kstat(arr[0:x+1], 1) )
		#my_arr.append(kstat(arr[0:x+1], 2) )
		#my_arr.append(kstat(arr[0:x+1], 3) )
		#my_arr.append(kstat(arr[0:x+1], 4) )
		my_arr.append(skews)

			#my_arr.append( med_vals)
		#vals.append(my_arr)
		#my_arr.append(mean_vals, axis = 0)
		#my_arr.append(median_vals, axis = 0)
		vals_.append( np.ndarray.flatten( np.array(my_arr) ) )
	#return_val.append(np.mean(vals , axis = 0 ) )
	return_val.extend( np.mean( np.array(vals_ ) , axis=0 ) )
	return_val.extend( np.mean( np.array(vals_ ) , axis=0 ) ) 
	#return_val.append(np.mean( np.array(vals_ ) , axis = 0) )
	#return_val.append(np.ndarray.flatten(np.array(vals ) ) )
	#return_val.append(np.mean(vals , axis = 0) )
	#return_val = np.mean(return_val, axis = 1 ) 		
	#return np.mean(return_val, axis=1 )
	return np.ndarray.flatten(np.array(return_val ) )



def butter_bandpass(lowcut, highcut, fs, order=5):
	nyq = 0.5 * fs
	low = lowcut / nyq
	high = highcut / nyq
	b, a = butter(order, [low, high], btype='band')
	return b, a

#fs = sample rate
#low cut = lowest frequency, high_cut = highest frequency
#order [3-10] ?
#data = spectrum
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
	return_val = []
	for val in data:
		b, a = butter_bandpass(lowcut, highcut, fs, order=order)
		y = lfilter(b, a, val)
		#return y
		ratio_band_energy = np.sum( [a^2 for a in y] )/ np.sum([b^2 for b in val] )
		return_val.append(ratio_band_energy )
	return return_val








if __name__ == "__main__":
	my_val = [ 2, 3, 4, 1, 3, 4, 5, 6, 3, 7, 6, 5, 9, 10] 
	print( cross_val(my_val) )
