import numpy as np
import sys
import matplotlib.pyplot as plt

sys.path.append(".")
from Random import Random

if __name__ == "__main__":

	#set default number of samples
	Nsample = 1000

	# read the user-provided seed from the command line (if there)
	if '-Nsample' in sys.argv:
		p = sys.argv.index('-Nsample')
		Nsample = int(sys.argv[p+1])
	if '-h' in sys.argv or '--help' in sys.argv:
		print ("Usage: %s -Nsample [number]" % sys.argv[0])
		print
		sys.exit(1) 

	nAccept = 0
	nTotal = 0
	
	# accepted values
	Xaccept = []
	Yaccept = []

	# reject values
	Xreject = []
	Yreject = []

	# sample number
	isample = []
	# calculated values of Pi (per sample)
	calcPi = []

	random = Random()

	idraw = max(1,int(Nsample)/100000)
	for i in range(0,Nsample):
		X = random.rand()
		Y = random.rand()

		nTotal += 1
		if(X*X + Y*Y <= 1): #accept if inside
			nAccept += 1
			if(i % idraw == 0):
				Xaccept.append(X)
				Yaccept.append(Y)
		else: # reject if outside
			if(i % idraw == 0):
				Xreject.append(X)
				Yreject.append(Y)
		if(i % idraw == 0):
			isample.append(nTotal)
			calcPi.append(4*nAccept/nTotal)



	#plot calculated pi vs sample number
	fig1 = plt.figure()
	plt.plot(isample,calcPi)
	plt.ylabel(r'Approximate $\pi$')
	plt.xlabel("Sample number")
	plt.xlim(0,isample[len(isample)-1])
	ax = plt.gca()
	ax.axhline(y=np.arccos(-1),color='green',label=r'true $\pi$')
	plt.title(r'Approximation of $\pi$ as a function of number of samples')
	plt.legend()

	fig1.savefig("calculatedPiPy.pdf")


	#plot accept/reject points
	fig2 = plt.figure()
	plt.plot(Xaccept,Yaccept,marker='o',linestyle='',color='green',label='accept')
	plt.plot(Xreject,Yreject,marker='o',linestyle='',color='red',label='reject')
	plt.ylabel("Y")
	plt.xlabel("X")
	plt.legend()


	x_circle = np.arange(min(min(Xaccept),min(Xreject)),max(max(Xaccept),max(Xreject)),0.001)
	y_circle = [np.sqrt(1-i*i) for i in x_circle]
	plt.plot(x_circle,y_circle,color='blue',label=r'$x^2 + y^2 = 1$')
	plt.legend()
	plt.title('Sampled points')
	fig2.savefig("circleQuadPy.pdf")











