""" 
Homework #5 Code

Instructions: 
- Use the rejection sampling ("reject/accept", "hit-or-miss", etc. from Feb. 14 lecture) to sample from a distribution 
  and visualize the target distribution, proposal distribution, and random samples. I would recommend trying to do at 
  least one of the two following exercises (in order of relevance):
  
    1) Choose a "non-trivial" (not uniform, not piece-wise linear) function defined on the interval [0, 1]. Using a 
       uniform distribution (or something better matching your target function) sample your target function using 
       rejection sampling. Visualize the results (ideally each step of the procedure).
       
    2) Sample random points in a closed 2D or 3D space (NOT just a circle) using rejection sampling and visualize the 
       results. Try to make the density "not uniform" in some way over the domain of points (ex. a 3D sphere of ellipse)
       
Post all code and figures to your GitHub repository "PHSX815_Week5" - don't forget to include a README.md file explaining how to run your code.

Authors: @aelieber1 
    - code adapted from @crogan Week 5
    - function definition help from: https://www.tutorialspoint.com/how-to-plot-a-function-defined-with-def-in-python-matplotlib
"""

import numpy as np
import sys
import matplotlib.pyplot as plt

sys.path.append(".")
from Random import Random 

#global variables
bin_width = 0.
Xmin = -3.
Xmax = 3.
random = Random()

# Normal distribution with mean zero and sigma = 1
# Note: multiply by bin width to match histogram
def Gaus(x):
	return (1./np.sqrt(2.*np.arccos(-1)))*np.exp(-x*x/2.)

def PlotGaus(x,bin_width):
	return bin_width*Gaus(x)

# Uniform (flat) distribution scaled to Gaussian max
# Note: multiply by bin width to match histogram
def PropFunction(x):
	return 1./np.sqrt(2*np.arccos(-1.))
	
def PlotPropFunction(x,bin_width):
	return bin_width*PropFunction(x)

# Get a random X value according to a flat distribution
def SamplePropFunction():
	return Xmin + (Xmax-Xmin)*random.rand()

# Creating our proposal function
# Note: multiply by bin width to match histogram
def TargetFunction(x):
    coeff = 3
    amplitude = 0.02
    return amplitude + np.sin(2*x) * np.sin(coeff*x)  

def PlotTargetFunction(x, bin_width):
    return bin_width*TargetFunction(x);

# Get a random X value according to sine wave distribution
def SampleTargetFunction():
  R = random.rand()
  Nexp = np.sqrt(2./np.arccos(-1))/Xmax*(1 - np.arcsin(-3.*Xmax*Xmax/8.))
  Ntot = 2.*Nexp + Xmax/2./np.sqrt(2.*np.arccos(-1))
  if R <= Nexp/Ntot:
    F = random.rand()
    return 2./Xmax*np.log(F*np.arcsin(-Xmax*Xmax/2.)+(1.-F)*np.arcsin(-Xmax*Xmax/8))
  if R >= (1.-Nexp/Ntot):
    F = random.rand()
    return -2./Xmax*np.log(F*np.arcsin(-Xmax*Xmax/2.)+(1.-F)*np.arcsin(-Xmax*Xmax/8))
  else:
	  return Xmin/4. + (Xmax-Xmin)/4.*random.rand();



""" Main Function """
if __name__ == "__main__":

	# default number of samples
	Nsample = 100

	doLog = False
	doExpo = False

	# read the user-provided seed from the command line (if there)
	#figure out if you have to have -- flags before - flags or not
	if '-Nsample' in sys.argv:
		p = sys.argv.index('-Nsample')
		Nsample = int(sys.argv[p+1])
	if '-range' in sys.argv:
		p = sys.argv.index('-range')
		Xmax = float(sys.argv[p+1])
		Xmin = -float(sys.argv[p+1])
	if '--log' in sys.argv:
		p = sys.argv.index('--log')
		doLog = bool(sys.argv[p])
	if '--expo' in sys.argv:
		p = sys.argv.index('--expo')
		doExpo = bool(sys.argv[p])
	if '-h' in sys.argv or '--help' in sys.argv:
		print ("Usage: %s [-Nsample] number [-range] Xmax [--log] [--expo] " % sys.argv[0])
		print
		sys.exit(1)  

        
	data = []
	Ntrial = 0.
	i = 0.
	while i < Nsample:
		Ntrial += 1
		if doExpo:
			X = SampleTargetFunction()
			R = PropFunction(X)/TargetFunction(X)
		else:
			X = SamplePropFunction()
			R = Gaus(X)/PropFunction(X)
		rand = random.rand()
		if(rand > R): #reject if outside
			continue
		else: #accept if inside
			data.append(X)
			i += 1 #increase i and continue
            
                
	if Ntrial > 0:
		print("Efficiency was",float(Nsample)/float(Ntrial))  # # of accepts / # 
        
	#normalize data for probability distribution
	weights = np.ones_like(data) / len(data)
	n = plt.hist(data,weights=weights,alpha=0.3,label="samples from f(x)",bins=100)
	plt.ylabel("Probability / bin")
	plt.xlabel("x")
	bin_width = n[1][1] - n[1][0]
	hist_max = max(n[0])

	if not doLog:
		plt.ylim(min(bin_width*Gaus(Xmax),1./float(Nsample+1)),
		1.5*max(hist_max,bin_width*Gaus(0)))
	else:
		plt.ylim(min(bin_width*Gaus(Xmax),1./float(Nsample+1)),
		80*max(hist_max,bin_width*Gaus(0)))
		plt.yscale("log")


	x = np.arange(Xmin,Xmax,0.001)
	y_norm = list(map(PlotGaus,x,np.ones_like(x)*bin_width))
	# y_norm = list(map(PlotGaus(x,bin_width))
	plt.plot(x,y_norm,color='green',label='target f(x)')

	if not doExpo:
		y_flat = list(map(PlotPropFunction,x,np.ones_like(x)*bin_width))
	else:
		y_flat = list(map(PlotTargetFunction,x,np.ones_like(x)*bin_width))

	
	plt.plot(x,y_flat,color='red',label='proposal g(x)')
	plt.title("Density estimation with Monte Carlo")

	
	plt.legend()
	plt.show()
	#plt.savefig("RandomGaussPy.pdf")