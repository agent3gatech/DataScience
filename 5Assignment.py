#Assignment 4. Using PACE Conda env. 

import numpy as np                                              #Import the usual suspects
import matplotlib.pyplot as plt                                 #Import Matplotlib
import scipy

from scipy.optimize import minimize
from scipy.stats import norm                                    #Import Gaussian curve from scipy.stats
from scipy.stats import poisson
from scipy.stats import binom
from scipy.stats import uniform
from scipy.stats import chisquare				#Import Chi2 'value calculator function'
from scipy.stats import chi2					#Import Ch2 distribution
#1.1
photon_data = np.genfromtxt('Project5_Set7.txt')         #Read in the data text file.
photon_data = photon_data[photon_data[:] < 3000] 	 #There is one photon at 10 times the energy of the others, I cut it as it screws with the plots and isn't significant as it's just one photon.

#1.2
Nt = photon_data.size #Count the number of photons in the file.
print('\nThe total number of photons, signal plus background (Ns + Nb), in the data set is;', Nt)

#1.3
plot1 = plt.figure(1)
n_bins = 100 
photon_energy_hist = plt.hist(photon_data, n_bins, label='Photon Energy Distribution')

plt.suptitle('Histogram of Photon Energies')
plt.xlabel('Photon Energy (a.u.)')
plt.ylabel('Photon Counts')
plt.legend(loc="upper right")
plt.grid()

E0_true = 205
gamma_true = 28

guess_Nb = 324 	#Dissicult to tell the ratio of Nb to the possible signal. Looks like there might be a small signal. 
guess_Ns = 46

print('\nWe can use this plot to eyeball some initial values for the parameters Ns and Nb for our fit.')
print('It seems like ~46 photons are signal and about 324 are background, this is a very rough guess...')

#1.4
def loglikelihood(params): #Define our -2*log(likelihood), expression taken from extended likelihood example in the book 
    Ns    = params[0]

    return -2*np.sum(np.log( ((Nt-Ns)/100)*np.exp(-photon_data/100) + (Ns/(np.pi*gamma_true))*(1/(1+(((photon_data-E0_true)/gamma_true)*((photon_data-E0_true)/gamma_true))))))

min_outcome = minimize(loglikelihood, [guess_Ns]) #minimize the log likelihood with respect to our parameter space 
print("\nThe minimization was successful: " + str(min_outcome.success))
print("I found a minimum of: {:.2f}".format(min_outcome.fun))
min1 = min_outcome.fun

fit_Ns = min_outcome.x[0]	#Get the best fit parameter values from the MLM
myfit_Ns = fit_Ns

print("Best fit values are Ns = {:.3f}".format(fit_Ns))
print("And so Nb is then 370 - {:.3f} = {:.3f}".format(fit_Ns, 370-fit_Ns))

#1.5	Now we want to replot the photon distribution histogram with our best fit curves for Signal and Bkg on top.
Ex = np.linspace(0,1000,1000)	#Make some arrays for our values for the best fit curves
exp_fit = Ex.copy()
cauchy_fit = Ex.copy()


for i in range(len(Ex)): #Populate the arrays with the signal/bkg values from the fitted distributions
	exp_fit[i] = ((Nt-fit_Ns)/Nt)*(1/100)*np.exp(-Ex[i]/100)
	cauchy_fit[i] =	(fit_Ns/Nt)*(1/(np.pi*gamma_true))*(1/(1+(((Ex[i]-E0_true)/gamma_true)*((Ex[i]-E0_true)/gamma_true))))

plot2 = plt.figure(2)

photon_energy_hist = plt.hist(photon_data, n_bins, density=True, label='Photon Energy Distribution')
plt.plot(Ex, exp_fit, 'cyan', label='Background; exponential distribution', linestyle='dashed')
plt.plot(Ex, cauchy_fit, 'orange', label='Signal; Cauchy distribution', linestyle='dashed')
plt.plot(Ex, exp_fit + cauchy_fit, 'r', label='Summed Signal + Background', linestyle='dashed')

plt.suptitle('Photon Energy Distribution with MLM fit \n to signal & background distributions')
plt.xlabel('Photon Energy (a.u.)')
plt.ylabel('Photon Counts')
plt.legend(loc="upper right")
plt.grid()

#1.6 We've already calculated the numerator in the expression for the likelihood ratio, we just need the value of the log likelihood with only background, for the ratio.

def loglikelihoodratio(params): 
    Ns    = params[0]

    return -2*(np.sum(np.log( ((Nt-Ns)/100)*np.exp(-photon_data/100) )))

min_outcome = minimize(loglikelihoodratio, [guess_Ns], bounds=((0,400),)) #minimize the log likelihood with respect to our parameter space 
print("The minimum is: {:.2f}".format(min_outcome.fun)) #Get the value for the MLR
min2 = min_outcome.fun

myfit_TS =-min1+min2
print("And so the TS from the likelihood ratio is; {:.3f} - {:.3f} = {:.3f}".format(-min1, -min2, -min1+min2))

#1.7/8/9
#Main loop
experiments = 10000 			#Number of simulation sets
Ns_array = np.empty(experiments)	#Arrays for the number of signal photons in each set and the Test Statistics for later
TS_array = np.empty(experiments)

for i in range(experiments):
	sim_data = np.random.exponential(100,Nt)	#Get random photon energies from the background distribution. Same number in each experiment as in our data files.
	def loglikelihood(params):  	#The usual song and dance...
		Ns    = params[0]

		return -2*np.sum(np.log( ((Nt-Ns)/100)*np.exp(-sim_data/100) + (Ns/(np.pi*gamma_true))*(1/(1+(((sim_data-E0_true)/gamma_true)*((sim_data-E0_true)/gamma_true))))))

	min_outcome = minimize(loglikelihood, [guess_Ns], bounds=((0,400),))  #Bounded range for Ns now as it wants to go negative...
	fit_Ns = min_outcome.x[0]
	Ns_array[i] = fit_Ns
	min1 = min_outcome.fun

	def loglikelihoodratio(params):		#Denominator for LR
		Ns    = params[0]

		return -2*(np.sum(np.log( ((Nt-Ns)/100)*np.exp(-photon_data/100) )))

	min_outcome = minimize(loglikelihoodratio, [guess_Ns], bounds=((0,400),)) 
	min2 = min_outcome.fun
	TS_array[i] = -min1+min2
#End main loop

#Make plots of the Ns and TS.
plot3 = plt.figure(3)
Ns_hist = plt.hist(Ns_array, 50, label='Ns values')

plt.suptitle('Distribution of best fit number of \n signal from purely background distribution')
plt.xlabel('Ns')
plt.ylabel('Counts')
plt.grid()
plt.axvline(x=myfit_Ns, color='green', linestyle='dashed', label='Ns from MLM fit to data file')
plt.legend(loc="upper right")

plot4 = plt.figure(4)
TS_hist = plt.hist(TS_array, 50, label='TS values')

q1 = np.quantile(TS_array, 0.01)	#Get the 1% quantile.
print("From our simulations, if we pick our criteria such that we require the TS to be less than {:.3f} to reject the null hypothesis, we get a type 1 error 1 percent of the time.".format(q1))

plt.suptitle('Distribution of TS values \n from purely background distribution')
plt.xlabel('TS')
plt.ylabel('Counts')
plt.grid()
plt.axvline(x=q1, color='orange', linestyle='dashed', label='1% Quantile for TS threshold')
plt.axvline(x=myfit_TS, color='green', linestyle='dashed', label='TS from MLR on data file')
plt.legend(loc="upper right")

plt.show()

#10
print("The null hypothesis (only background in the data file) is clearly rejected for this data set. The TS from the likelihood ratio test on our data is far below (almost an order of magnitude below) the 1% quantile TS threshold defined previously.")
#11
print("The more experiments we simulate the more accurate, i.e. the smaller the errors, are on the threshold we define for the TS accepting/rejecting the null hypothesis. In our usual way; Limit of large number of experiments -> invoke the central limit theorem -> Gaussian errors we see that with 10,000 simulation sets we have a error of sqrt(10000) = 100, or about 1%. This is on the level of precision required byour threshold (itself being on the order of a percent). If we only had 100 experiments we would only be sensitive on the order of 10%. We could use 1 million experiments and be even more confident, but, if you aren't using a powerful computer, you'd have to be more patient to get your results.") 
print("Using the above method to approximate; for a 5-sigma confidence interval, which is equivelant to 1/1744278 being outside this range we would want about, the square of the inverse of this number, 3x10^12 experiments.")
print("In the set that I got I think you can fairly easily eyeball what looks like a signal componant. If E0 and Gamma were not known then it definitely would not be easy to judge by eye if it is real or not.")
print("Not sure how long this one took, spent a reasonable amount of time looking into the subtleties of how you define the ratio. Maybe 4-5 hours?")





