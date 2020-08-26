
#window_size = 1
#step_size = 2
import findspark

findspark.init()

import sys

window_size = int (sys.argv[2] )
step_size = int ( sys.argv[1] )
from pyspark import SparkContext, SparkConf
from pathlib import Path
import multiprocessing as mp

from joblib import Parallel, delayed
from features import *
import numpy as np

import random
import math
from scipy.optimize import fsolve
import sys
import os
import copy
#from bfi import *
import bfi
from itertools import islice
random.seed(200)

def get_filepaths(directory):
	file_paths = []  # List which will store all of the full filepaths.

	# Walk the tree.
	for root, directories, files in os.walk(directory):
		for filename in files:
			# Join the two strings in order to form the full filepath.
			filepath = os.path.join(root, filename)
			file_paths.append(filepath)  # Add it to the list.
	return file_paths  # Self-explanatory.


from collections import deque

def window(seq, n=2):
	it = iter(seq)
	win = deque((next(it, None) for _ in range(n)), maxlen=n)
	yield list(win)
	append = win.append
	for e in it:
		append(e)
		yield list(win)


def separate_into_energy_features(lines):
	#loop through file and generate first features
	return_val = []
	vals = [x.strip().split(" ") for x in lines.split("\n") if x.strip() !='' and x.strip()!=[] ]
	spec_data = [x for x in lines.split("\n")[10:] if x!='' and x!=[] ]
	try:
		for spec in spec_data:
			dat = [x for x in spec.strip().split(" ") if x.strip()!="" ]
			#print(dat)
			return_val.append([ float(x) for x in dat[1: 8] ] )
	except Exception as e:
		print(e)
		return_val  = []
	#print(return_val)
	return return_val

def separate_into_other_features(path):
	pass
def multiply(list_of_arrays):
	arr = [1 for x in np.ndarray.flatten(np.array(list_of_arrays[0] ) ) ]
	arr = [ np.array([1, 1, 1, 1, 1, 1,1] ) for x in np.array(list_of_arrays[0] ) ]
	#print(arr)
	temp_arr = []
	for x in range(len(list_of_arrays)) :# loop over array
		temp = arr
		try:
			val = np.multiply( arr, np.array(list_of_arrays[x] ) )
			arr = val
		except Exception as e:
			return [0] 	
	cross_v = cross_val(arr)
	means = np.mean ( np.array( arr) , axis=0 )
	cov_corr = covariance_and_correlation(arr ) 
	ret_val = []
	ret_val.extend( np.ndarray.flatten( np.array ( cov_corr) ) ) 
	ret_val.extend(np.ndarray.flatten( np.array (cross_v) ) )
	
	ret_val.extend(np.ndarray.flatten( np.array ( means) ) ) 
	means_2= np.mean( np.array([  m[2:] for m in arr ] ) , axis=1)
	ret_val.extend(np.ndarray.flatten( np.array ( means_2) ) )
	return np.ndarray.flatten(  np.array(ret_val  ) )
	#return first_val

def gen_data(arr):
	to_write = []
	#for filename in paths:
	temps = []
	current_data = ""
	#for line in f.readlines():
	#print(arr)
	for line in arr:
		if line[:5] == "File ":
			data = [x for x in line.split("\n")]
			#process the data
			if data != [''] and data != '' and data!=[]:
				temps.append(current_data)
				current_data = line #
		else:
			current_data += line	
	return temps

def calculate_other_features(lines):
	#calculate other features
	try:
		other_vars = [x for x in lines.split("\n")[6].split(" ") if x.strip()!=""]
		hs = float(other_vars[1])
		tp = float(other_vars[3])
		ta = float(other_vars[7])
		return [hs, tp, ta, float(other_vars[5]) ]
	except Exception as e:
		print(e)
	return [0, 0, 0]

files = []

def get_station(mylist):
	#print(mylist[0] )
	try:
		station_name = " ".join([ val for val in mylist.split("\n")[1].split(" ") if val.strip()!="" ][2:6] )
		#print(mylist.split("\n") )
	except Exception as e:
		return [0]
	return station_name
def calculate_bfi(spectra):
	tot_energy = 0
	outs = []
	outs = [] #[0, 0, 0, 0]
	for val in spectra:
		tot_energy += float(val[1] ) * float(val[0]) 
		#vals = float(val[0] * )
		outs.append([ float(x) * float(val[0] ) * float(val[1] ) for x in val[3: 7] ] ) # we discard the first frequency values coz we dont need it
	fourier_vals = np.sum(outs, axis=0)
	#print(fourier_vals)
	#print(fourier_vals)
	try:
		bfi_ = bfi.bfi(fourier_vals[0]/tot_energy, fourier_vals[1]/tot_energy, fourier_vals[2]/tot_energy, fourier_vals[3]/tot_energy )	
	except Exception as e:
		print( e) 
		return -1 
	return bfi_


def calculate_depth(my_val):
	#return my_val.split("\n").split(" ")
	#print(my_val)
	other_vars = [x for x in my_val.split("\n")[3].split(" ") if x.strip()!=""]
	return int(other_vars[2] )
def split_array(arr, stations_list):
	returns = []
	#indices = []
	#remove indices with [0] ie no use
	my_arr = []
	for val in arr:
		if isinstance(val, dict):#val == [0]:
			my_arr.append(val )

	arr = my_arr
	stations = {}
	for val in range(0, len(arr) - 1 ):
		if stations_list[val] not in stations:
			stations[ stations_list[val]  ] = [val]
		else:
			stations[ stations_list[val] ] = stations[ stations_list[val] ] + [val]
	for val in stations:
		if isinstance(val, str):
			returns.append([ arr[x] for x in stations[val] ] )
	return returns
if __name__ == "__main__":
	import os
	#if not os.path.exists("step" + str(step_size) + "/window_" + str(window_size) + "_dataset" ):
	#	Path( "step" + str(step_size) + "/window_" + str(window_size) + "_dataset").mkdir(parents=True, exist_ok=True)
	conf = SparkConf().setAppName("features")
	# Run the above function and store its results in a variable.   
	full_file_paths = get_filepaths("inputs/")
	#random.seed(100)
	paths = [ x for x in full_file_paths if x.split("/")[-1][:2] =="sp" ]#[ : int( 0.3 * len(full_file_paths) )]
	random.shuffle(paths)
	
	sc = SparkContext(master='spark://137.30.125.208:7077', appName='spark_features')
	#local files to import 
	sc.addPyFile('bfi.py')
	sc.addPyFile('features.py')
	for f in paths:
		fil = open(f).readlines()
		val = gen_data(fil)
		depth = calculate_depth(val[1]) 
		val = sc.parallelize(val)
		features = val.map( lambda x: separate_into_energy_features(x) )#.map(lambda a: window(a) )
		llist = features.collect()
		val = [ a for a in window(llist, n = window_size) if a != [] and a[0] != [] ] 
		my_val = sc.parallelize(val).map(lambda a : multiply(a) )
		vals = my_val.collect()
		bfis =  features.map(lambda a : calculate_bfi(a) ).collect()
		with open("feature_files/" + f.split("/")[-1], "w" ) as myfile:
			#since we loop over outputs rather than inputs, it is implicitly - window_size too
			inputs = [x for x in vals ][1:][:len(vals ) - step_size]# - window_size  ] # first element is [0] by some thing and window already starts like that
			outputs = [s for s in bfis ][1:] [window_size:] [ step_size : ] #first element if [0] by something
			#other_vals_wind = [s for s in other_vals_wind ][1:] [:len( other_vals_wind ) - step_size ] 
			try:
				for v in range(len(outputs)):
					#try:
					if len(inputs[v]) == len(inputs[0] ) and outputs[v] != -1:# and depth > 20:
						myfile.write(str(outputs[v]  ) + "," )
						myfile.write(",".join( str(x) for x in np.ndarray.flatten( np.array( inputs[v] ) ) )   )
						myfile.write("\n")
			except Exception as e:
				print(e)
		#for line in llist:
		#	print(line)
