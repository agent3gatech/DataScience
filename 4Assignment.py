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
photon_data = np.genfromtxt('Project4_Set9.txt')         #Read in the data text file.
photon_data = photon_data[photon_data[:] < 10000] 	 #There is one photon at 100 times the energy of the others, this is possible due to the exponential background, but I cut it as it screws with the plots and isn't significant as it's just one photon.

#1.2
Nt = photon_data.size #Count the number of photons in the file.
print('\nThe total number of photons, signal plus background (Ns + Nb), in the data set is;', Nt)

#1.3
plot1 = plt.figure(1)
n_bins = 50 #Tested a few different binnings, this shows morphology of distribution without 'comb effect'.
photon_energy_hist = plt.hist(photon_data, n_bins, label='Photon Energy Distribution')

plt.suptitle('Histogram of Photon Energies')
plt.xlabel('Photon Energy (a.u.)')
plt.ylabel('Photon Counts')
plt.legend(loc="upper right")
plt.grid()

guess_gamma = 30
guess_E0 = 205
guess_Ns = 200

print('\nWe can use this plot to eyeball some initial values for the parameters gamma and E0 in our minimization. E0 corresponds to the peak of the Cauchy distribution and gamma to the half-width at half-maximum.')
print('E0 looks it is around', guess_E0)
print('Gamma looks it is around', guess_gamma)
print('It seems like ~200 photons are signal and about 300 are background, this is a very rough guess...')

#1.4
def loglikelihood(params): #Define our -2*log(likelihood), expression taken from extended likelihood example in the book 
    Ns    = params[0]
    E0    = params[1]
    gamma = params[2]

    return -2*np.sum(np.log( ((Nt-Ns)/100)*np.exp(-photon_data/100) + (Ns/(np.pi*gamma))*(1/(1+(((photon_data-E0)/gamma)*((photon_data-E0)/gamma))))))

min_outcome = minimize(loglikelihood, [guess_Ns, guess_E0, guess_gamma]) #minimize the log likelihood with respect to our parameter space 
print("\nThe minimization was successful: " + str(min_outcome.success))
print("I found a minimum of: {:.2f}".format(min_outcome.fun))

fit_Ns = min_outcome.x[0]	#Get the best fit parameter values from the MLM
fit_E0 = min_outcome.x[1]
fit_gamma = min_outcome.x[2]

print("Best fit values are Ns = {:.3f} , E0 = {:.3f} , gamma = {:.3f}".format(fit_Ns,fit_E0,fit_gamma))
print("And so Nb is then 500 - {:.3f} = {:.3f}".format(fit_Ns, 500-fit_Ns))

#1.5	Now we want to replot the photon distribution histogram with our best fit curves for Signal and Bkg on top.
Ex = np.linspace(0,1000,1000)	#Make some arrays for our values for the best fit curves
exp_fit = Ex.copy()
cauchy_fit = Ex.copy()


for i in range(len(Ex)): #Populate the arrays with the signal/bkg values from the fitted distributions
	exp_fit[i] = ((Nt-fit_Ns)/Nt)*(1/100)*np.exp(-Ex[i]/100)
	cauchy_fit[i] =	(fit_Ns/Nt)*(1/(np.pi*fit_gamma))*(1/(1+(((Ex[i]-fit_E0)/fit_gamma)*((Ex[i]-fit_E0)/fit_gamma))))

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

#1.6	Make the plots of likelihood
def vecloglikelihood(E0_space,gamma_space):	#Similar approach to Ignacio's example code, define a vectorized log likelhood function with Ns/Nb taking the values of the MLM fit so it's a conditional likelihood, strictly speaking.
    return -2*np.sum(np.log( ((Nt-fit_Ns)/100)*np.exp(-photon_data/100) + (fit_Ns/(np.pi*gamma_space))*(1/(1+(((photon_data-E0_space)/gamma_space)*((photon_data-E0_space)/gamma_space))))))


dim1 = 1000	#Definte the dimension of our parameter phase spaces and the ranges for the params
dim2 = 250
E0_min = 150
E0_max = 250
gamma_min = 15
gamma_max = 40

E0_space = np.linspace(E0_min,E0_max,dim1+1)	#Make the 1D and 2D parameter spaces
gamma_space = np.linspace(gamma_min,gamma_max,dim2+1)
E0_grid,gamma_grid=np.meshgrid(E0_space,gamma_space)

vllh = np.vectorize(vecloglikelihood)	#Vectorize the likelihood 
Lmap = vllh(E0_grid, gamma_grid)	#Map it

min_fromgrid = np.where(Lmap == np.amin(Lmap)) #Get the minimum from the minimization for plotting

minimum2LogL = float(Lmap[min_fromgrid[0],min_fromgrid[1]])	#Get the minimum value for the likelihood and the found values for E0 and gamma
print('\nMinimum -2logL : ', minimum2LogL)
E0_scan= E0_space[min_fromgrid[1]]
gamma_scan = gamma_space[min_fromgrid[0]]
print('MLM value for E0 from vectorized scan;', E0_scan)	#Print the values and 1-sigma uncertainties eyeballed from the contour on the plot.
print('Eyeballed 1 sigma undertainty for E0, upper = +3.2')
print('Eyeballed 1 sigma undertainty for E0, lower = -2.9')
print('MLM value for gamma from vectorized scan;', gamma_scan)
print('Eyeballed 1 sigma undertainty for gamma, upper = +2.8')
print('Eyeballed 1 sigma undertainty for gamma, lower = -2.5')

plot3 = plt.figure(3)

axes = plt.axes()
Levs = [minimum2LogL+1]		#definte the 1 sigma uncertaintity
contour = axes.contour(E0_grid, gamma_grid, Lmap,cmap='Reds',levels=Levs) #Add the uncertaintity contour
plt.plot(E0_space[min_fromgrid[1]], gamma_space[min_fromgrid[0]], 'r+',label='Fit value \n (1 Sigma uncertainty shown in white)')
plt.legend()
plt.xlabel("E0")
plt.ylabel("Gamma")
plt.suptitle('Likelihood value scan for values of E0 and gamma \n with MLM value and 1 sigma uncertainty contour.')
im = axes.imshow(Lmap,interpolation='none',extent=[E0_min,E0_max,gamma_min,gamma_max],origin='lower') #Format the axes

# 1.7
print("\nTo be explicit; I'm talking about the Cauchy distribution here. The (relativistic) Breit-Wigner distribution is not what we are working with, and if memory serves, is parameterized slightly differently, even in the non-relativistic limit, than the Cauchy distribution and so the 'gamma' in the B-W distribution is actually rhe full-width at half-maximum (I think). The 'gamma', in the Cauchy distribution we use here, is the 'scale' parameter, which specifies the half-width at half-maximum. It is also equal to half the 'interquartile range' (the difference between the 75th and 25th percentiles). In either distribution it is inversely related to the 'lifetime' of the transition that is responsible for the photon emission, people often relate the energy uncetainty/transition lifetime with the Heisenberg uncertainty principle.")

#2.1
plot4 = plt.figure(4)
bin_array = np.linspace(0,400,26) #Make our array of 26 bin edges, for our 25 bins
photon_energy_hist2 = plt.hist(photon_data, bin_array, label='Photon Energy Distribution')

plt.suptitle('Histogram of Photon Energies in 25 \n equally spaced bins between 0 and 400 a.u.')
plt.xlabel('Photon Energy (a.u.)')
plt.ylabel('Photon Counts')
plt.legend(loc="upper right")
plt.grid()

plt.show()

#2.2
def MyChi2(arguments):   #Define our Chi2 function in a way that will allow easy minimization later.
	Ns    = arguments[0]
	E0    = arguments[1]
	gamma = arguments[2]

	obs = photon_energy_hist2[0]	#Get the measured data from the binned histogram
	bin_centers = photon_energy_hist2[1]+8		#Get bin centers for use in the 'expected' value analytic function
	bin_centers = bin_centers[bin_centers[:] < 400]
	exp = obs.copy()
    
	for i in range(len(exp)): #Make the array of expected values from the distributions, given the parameters, to compare with the binned data
		exp[i] = ((Nt-Ns)/100)*np.exp(-bin_centers[i]/100) + (Ns/(np.pi*gamma))*(1/(1+(((bin_centers[i]-E0)/gamma)*((bin_centers[i]-E0)/gamma))))

	return chisquare(obs,exp)[0]

min_outcome = minimize(MyChi2, [guess_Ns, guess_E0, guess_gamma])	#Minimize the Chi2 to find the best fit params from this
chi2min = min_outcome.fun

ns_fit = min_outcome.x[0]	#Get the best fit paramater values from the Chi2 minimization
E0_fit = min_outcome.x[1]
ga_fit = min_outcome.x[2]

nbins_chi2 = np.size(bin_array)-1	#How many bins for the Chi2? 25
ndof = nbins_chi2-3 			#Since we have a binned data set the ndof is related to the number of bins, not number of photons.
pvalue = 1-chi2.cdf(chi2min,ndof)	#Calculate the p-value

#2.2/2.3
#Print the things
print("\nThe minimized chi-squared is {:.3f} for {:d} degrees of freedom".format(chi2min,ndof))
print("The p-value for this fit is {:.3f}".format(pvalue))
print("The best fit values of the parameters from the Chi Square fit are: Ns = {:.3f}, Nb = {:.3f}, E0 =  {:.3f} and Gamma = {:.3f}".format(ns_fit,Nt-ns_fit,E0_fit,ga_fit))
print("The best fit values of the parameters from the MLM are: Ns = {:.3f}, Nb = {:.3f}, E0 =  {:.3f} and Gamma = {:.3f}".format(fit_Ns,Nt-fit_Ns,fit_E0,fit_gamma))

#2.4
#P value is zero!?! Not a good test, chi2 needs normally distributed error/uncertainties to be valid, this is only true for large statistics in each bin which is not the case here.
print("\nThe chi-square for binned data is not a good method to use in this problem; the method requires normally distributed uncertainties and this is only true for large counts in each bin, which is not the case here. There is even a bin with zero entries! The calculated p-value is zero, but we should not trust this as per the reasoning above. ")




