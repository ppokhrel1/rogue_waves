
chromosomes = "1000100010000010010000000111001000000000000001010000011010100000001011000100011100011100011001001100000100001110110110001111100001010110101001000011111001000011111111"


import sys
import numpy as np

filename = sys.argv[1]

X_val = []

with open(filename, "r") as f:
	for line in f:
		X_val.append([ float(x) for x in line.split(",") ] )		 
X_ = []
for x in X_val:
	X_.append( np.array( [ x[b] for b in indices ] )  )
X_val = X_
X_val = np.array(X_val)

with open("classifiers/step1_window4.pickle") as f:
	clas1 = pickle.load(f)			
	preds = [[a] for a in np.array( clas1.get_predictions(X) ) ]
	X_val = np.concatenate((X_val, preds), axis=1)
	preds = np.array( clas1.get_predictions(X_val) )
	print("Window 1: " + str(preds) )

with open("classifiers/step2_window4.pickle") as f:
	clas2 = pickle.load(f) 
	preds = [[a] for a in np.array( clas2.get_predictions(X) ) ]
	X_val = np.concatenate((X_val, preds), axis=1)
	preds = np.array( clas2.get_predictions(X_val) )
	print("Window 2: " + str(preds) )

with open("classifiers/step3_window4.pickle") as f:
	clas3 = pickle.load(f)
	preds = [[a] for a in np.array( clas3.get_predictions(X) ) ]
	X_val = np.concatenate((X_val, preds), axis=1)
	preds = np.array( clas3.get_predictions(X_val) )
	print("Window 3: " + str(preds) )

with open("classifiers/step4_window4.pickle") as f:
	clas4 = pickle.load(f)
	preds = [[a] for a in np.array( clas4.get_predictions(X) ) ]
	X_val = np.concatenate((X_val, preds), axis=1)
	preds = np.array( clas4.get_predictions(X_val) )
	print("Window 4: " + str(preds) )

with open("classifiers/step5_window4.pickle") as f:
	clas5 = pickle.load(f)
	preds = [[a] for a in np.array( clas5.get_predictions(X) ) ]
	X_val = np.concatenate((X_val, preds), axis=1)
	preds = np.array( clas5.get_predictions(X_val) )
	print("Window 5: " + str(preds) )

with open("classifiers/step6_window4.pickle") as f:
	clas6 = pickle.load(f)
	preds = [[a] for a in np.array( clas6.get_predictions(X) ) ]
	X_val = np.concatenate((X_val, preds), axis=1)
	preds = np.array( clas6.get_predictions(X_val) )
	print("Window 6: " + str(preds) )

with open("classifiers/step7_window4.pickle") as f:
	clas7 = pickle.load(f)
	preds = [[a] for a in np.array( clas7.get_predictions(X) ) ]
	X_val = np.concatenate((X_val, preds), axis=1)
	preds = np.array( clas7.get_predictions(X_val) )
	print("Window 7: " + str(preds) )

with open("classifiers/step8_window4.pickle") as f:
	clas8 = pickle.load(f)
	preds = [[a] for a in np.array( clas8.get_predictions(X) ) ]
	X_val = np.concatenate((X_val, preds), axis=1)	
	preds = np.array( clas8.get_predictions(X_val) )
	print("Window 8: " + str(preds) )

with open("classifiers/step9_window4.pickle") as f:
	clas9 = pickle.load(f)
	preds = [[a] for a in np.array( clas9.get_predictions(X) ) ]
	X_val = np.concatenate((X_val, preds), axis=1)
	preds = np.array( clas9.get_predictions(X_val) )
	print("Window 9: " + str(preds) )

with open("classifiers/step10_window4.pickle") as f:
	clas10 = pickle.load(f)
	preds = [[a] for a in np.array( clas10.get_predictions(X) ) ]
	X_val = np.concatenate((X_val, preds), axis=1)
	preds = np.array( clas10.get_predictions(X_val) )
	print("Window 10: " + str(preds) )

with open("classifiers/step11_window4.pickle") as f:
	clas11 = pickle.load(f)
	preds = [[a] for a in np.array( clas11.get_predictions(X) ) ]
	X_val = np.concatenate((X_val, preds), axis=1)
	preds = np.array( clas11.get_predictions(X_val) )
	print("Window 11: " + str(preds) )

with open("classifiers/step12_window4.pickle") as f:
	clas12 = pickle.load(f)
	preds = [[a] for a in np.array( clas12.get_predictions(X) ) ]
	X_val = np.concatenate((X_val, preds), axis=1)
	preds = np.array( clas12.get_predictions(X_val) )
	print("Window 12: " + str(preds) )

with open("classifiers/step13_window4.pickle") as f:
	clas13 = pickle.load(f)
	preds = [[a] for a in np.array( clas13.get_predictions(X) ) ]
	X_val = np.concatenate((X_val, preds), axis=1)
	preds = np.array( clas13.get_predictions(X_val) )
	print("Window 13: " + str(preds) )

with open("classifiers/step14_window4.pickle") as f:
	clas14 = pickle.load(f)
	preds = [[a] for a in np.array( clas14.get_predictions(X) ) ]
	X_val = np.concatenate((X_val, preds), axis=1)
	preds = np.array( clas14.get_predictions(X_val) )
	print("Window 14: " + str(preds) )











