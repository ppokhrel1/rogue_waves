#import subprocess
from collections import defaultdict
#from multiprocessing import set_start_method

#set_start_method("spawn")
from multiprocessing import get_context

import sys
import operator
from subprocess import Popen, PIPE
import numpy as np
import pandas as pd
#arr = sys.argv[2]
#text = sys.argv[1]
#print(arr)
#my_val = []
#for a in arr:#.split(","):
#	try:
#		m = subprocess.Popen(["find", "./buoy_data/nodc","-name", a], stdout=PIPE, stderr=PIPE)
#		if operator.contains(a, text):
#			print(a)
#			my_val.append(a)
#		stdout, stderr = m.communicate()
#		#print(stdout)
#	except Exception as e:
#		print(e)
#
#print( len(my_val ) )


import multiprocessing
import copy

def worker(procnum, send_end):
	'''worker function'''
	result = str(procnum)+' represent!'
	print(result)
	send_end.send([ procnum, result])

def main():
	jobs = []
	pipe_list = []
	for b in range(5):
		for j in range(5):
			i=b+j
			recv_end, send_end = multiprocessing.Pipe(False)
			p = multiprocessing.Process(target=worker, args=(i, send_end))
			jobs.append(p)
			pipe_list.append(recv_end)
			p.start()
	for proc in jobs:
		proc.join()
	result_list = [x.recv() for x in pipe_list]
	print(result_list)

def get_pred(X_train, y_train, clas_no, clas, fold_no, X_test, send_end):
	
	clas=copy.copy(clas)
	clas.fit(X_train, y_train)
	val =  [ a[0] for a in clas.predict_proba(X_test ) ]
	send_end.send([clas_no, fold_no, val] )
def get_pred(X_train, y_train, clas_no, clas, fold_no, X_test,):
        clas=copy.copy(clas)
        clas.fit(X_train, y_train)
        val =  [ a[0] for a in clas.predict_proba(X_test ) ]
        return [clas_no, fold_no, val] 
def chunks(l, n):
	n = max(1, n)
	return (l[i:i+n] for i in range(0, len(l), n))
def train_base(X_val, y_val, classifiers, skfold, batch_size = 50000):
	#batches = KFold(n_splits= int(len(X)/batch_size), random_state=100,)
	jobs = []
	pipe_list = []
	#for a in range(len(classifiers) ):
	i=0
	#recv_end = multiprocessing.Queue()
	#recv_end, send_end = multiprocessing.Pipe(False)
	pool = multiprocessing.Pool(multiprocessing.cpu_count())
	#for ind in range(int(len(X_val/batch_size))  ):
	for ind in chunks( [a for a in range(len(X_val))] , batch_size):
	
		i = 0
		X = X_val[ind]# * batch_size:(ind+1) * batch_size]
		y = y_val[ind]# * batch_size:(ind+1) * batch_size]
		for train, test in skfold.split(X,y):
			X_train, y_train = X[train], y[train]
			X_test, y_test = X[test], y[test]
			for a in range(len(classifiers) ):
				cls = classifiers[a]
				#with get_context("spawn").Pipe() as pipe:
				recv_end, send_end = multiprocessing.Pipe(False)
				#p = multiprocessing.Process(target=get_pred, args=(X_train, y_train, a, cls, i, X_test, send_end))
				p = pool.apply_async(get_pred, args=(X_train, y_train, a, cls, i, X_test, ),  )
				#p=pool.starmap(get_pred, (X_train, y_train, a, cls, i, X_test, recv_end ) )
				jobs.append(p)
			
				pipe_list.append(recv_end)
			#p.start()
			i+=1
	#for proc in jobs:
	#	#proc.terminate()
	#	proc.join()
	#	#proc.terminate()
	#pool.close()
	#pool.join()
	print("joined")
	#result_list = [recv_end.recv() for x in pipe_list]
	result_list = [a.get() for a in jobs ] 
	print(len(result_list) )
	pool.close()
	pool.join()
	#pool.close()
	#print(result_list)	
	pred = []
	#my_d = pd.DataFrame(result_list)
	#my_dict = my_d.groupby(0).groups #first column, classifier
	d = defaultdict(list)
	for a in range(len(result_list)):
		d[ result_list[a][0] ].append( a)		
	my_dict = d
	for v in my_dict:
		indices = my_dict[v]
		vals = [result_list[a] for a in indices ]
		vals = [a[1:] for a in vals]#get folds, the predictions arrays
		vals = np.array(vals)
		my_v = vals[vals[:, 0].argsort() ]
		my_v = np.array([ np.array(a[1:]) for a in my_v] )
		#print(my_v)
		#my_v = np.array([a[0] for a in my_v ] )
		#print(my_v)
		#[item for sublist in my_v for item in sublist]
		my_predictions = []
		for a in range( len(my_v[0] )):
			append_v = []
			for b in range(len(my_v)):
				append_v.append( my_v[b][a] )
			my_predictions.append(append_v)	
		#print(my_predictions)
		#print(len(my_predictions))
		#my_v = np.concatenate(my_v, axis=1)
		my_predictions = [item for sublist in my_predictions for item in sublist]
		my_predictions = [item for sublist in my_predictions for item in sublist]
		#print(len(my_predictions) )
		mp = []
		my_predictions =np.array(my_predictions)
		#print(len(my_predictions))
		#for a in range(len(my_predictions[0] ) ):
		#	mp.append(a)
		
		pred.append(my_predictions)
	return_val = np.array(pred)#.T
	#ret_val = []
	#for a in return_val:
	#	to_append = a#[item for sublist in a for item in sublist]
	#	ret_val.append(to_append)
	#return_val = np.concatenate(np.array(my_predictions).T, axis=1)
	ret_val = return_val.T
	#print(ret_val)
	print(len(ret_val))
	return ret_val
if __name__ == '__main__':
    main()

