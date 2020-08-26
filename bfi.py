

from __future__ import division
import math
from scipy.optimize import fsolve



#k = (2 * math.pi * f )^^2 / g
#omega = bandwidth normalized by peak frequency
#Hs = measure of total spectral energy = 4 * math.sqrt(Energy) 
#bfi = sqrt(2) * kp * Hs / ( 4 * omega)
#def benjamin_feir_index( Hs, bandwidth, peak_freq):
#	return math.sqrt(2)* (( 2 * math.pi *peak_freq)**2 / 9.81 * Hs)/(4 * bandwidth / (2 * math.pi * peak_freq) )

#print( benjamin_feir_index(2, 0.0100, 0.1300) )


#returns bfi from the fourier coefficients using formula for kurtosis
#kurtosis = κ = 3 + pi/sqrt(3) * BFI**2.
#paper = observations of sea and swell using directional wave buoy
#another paper = rogue wave formation in adverse ocean gradients
def bfi(a1, b1, a2, b2):
	theta = math.atan(b1/a1)
	m1 = (a1**2+ b1**2 )**(1/2)
	m2 = a2*math.cos(2*theta)+b2*math.sin(2*theta)
	n2 = b2 * math.cos(2*theta) - a2*math.sin(2*theta)
	val = (1 - m2/ 2) ** (3/2)
	if val == 0:
		val = 0.000000000001
	skew = - n2 / val
	#kurtosis = (math.sqrt(3)/math.pi * ( (6 - 8*m1 + 2*m2)/( (2*(1-m1) )**2 )  - 3  ) )
	kurtosis = (6 - 8*m1 + 2*m2)/( (2*(1-m1) )**2 ) 
	#kurtosis = abs(kurtosis) #sometimes gives smaller negative values which is close to 0
	#bfi_val = math.sqrt(math.sqrt(3) * (kurtosis - 3) / math.pi )
	if (kurtosis < 2+abs(skew) and abs(skew) <=4) or (kurtosis < 6 and abs(skew)>4):
		return 1
	return 0	

#kp = peak wavenumber of the spectrum, omega(w) is bandwidth delta_w normalized by peak angular frequency omega_p. 
#dispersion relation (2*pi*f)**2 = g*k*tanh(k*d)

#def solve_kp(x, depth, frequency):
#	peak_wavenumber = 9.81 * x * math.tanh(x*depth) - (2 * math.pi * frequency) ** 2 
#	return peak_wavenumber
def benjamin_feir_index(depth, frequency, bandwidth, total_energy):
	def equation(x):
		return 9.81 * x *math.tanh(x * depth) - (2 * math.pi * frequency)**2	
	kp = fsolve(equation, 0 ) #peak wavenumber
	#print(kp)
	omega = bandwidth / ( 2 * math.pi * frequency )
	bfi = math.sqrt(2) * kp * 4 * math.sqrt(total_energy) / (4 * omega)
	return bfi 	
	
#array = [
#	[-0.1194,  -0.5663,  -0.6039,   0.3199 ],
#	[ -0.2440,  -0.9289,  -0.7878,   0.4401 ],
#	[-0.1681,  -0.9067,  -0.7986,   0.3066 ],
#	[-0.0999,  -0.9072,  -0.7640,   0.1723 ],
#	[-0.0279,  -0.7738,  -0.7741,   0.2086],
#	[-0.0316,  -0.8764,  -0.8136,  -0.0138 ],
#	[-0.4494,  -0.7726,  -0.3060,   0.6843],
#	[-0.2251,  -0.7126,  -0.7191,   0.2231],
#	[-0.2915,  -0.8496,  -0.6999,   0.4999 ],	
#	[-0.3425,  -0.8009,  -0.6251,   0.5103 ],
#	[-0.8602,   0.0550,   0.7613,  -0.2513 ],
#	[0.1465,  -0.6238,  -0.3180,  -0.4264],
#	[-0.4095,   0.0773,  -0.1163,  -0.2167],
#	[0.1696,  -0.8108,  -0.6031,  -0.4343],
#]
#print( benjamin_feir_index(20, 0.0050, 0.0800,  1.1 ) )
#for val in array:
#	print(bfi (val[0], val[1], val[2], val[3] ) )
