#Assignment 2. Using PACE Conda env. 

import numpy as np                                              #Import Numpy
import matplotlib.pyplot as plt                                 #Import Matplotlib
import scipy
from scipy.stats import norm                                    #Import Gaussian curve from scipy.stats
from scipy.stats import poisson
from scipy.stats import binom
from scipy.stats import uniform

#Question 1
print("\n Question 1: As the x and y coordinates for a randomly picked point are drawn from uniform distribution, each point inside the square is equally likely to be chose. So the probability that a point falls within the triangle is simply the ration of the area of the triangle to the whole square. In this case it is 1/8.")

#Question 2
print("\n Question 2:")

A_MS = 0.8		#Define variables indicating the Market Share of each company (out of 1)
B_MS = 0.15
C_MS = 0.05

A_DR = 0.04		#Define variables indicating the Defect Rate of each company (again, out of 1)
B_DR = 0.06
C_DR = 0.09

A_DEF = A_MS*A_DR		#Calculate the probabilities of a module being defective or not from each company if we pick a random module from the whole pool.
A_OK  = A_MS*(1-A_DR)
B_DEF = B_MS*B_DR
B_OK  = B_MS*(1-B_DR)
C_DEF = C_MS*C_DR
C_OK  = C_MS*(1-C_DR)

TOTAL_DEF = A_DEF + B_DEF + C_DEF		#Sum the defective/okay probabilites from each company.
TOTAL_OK  = A_OK + B_OK + C_OK
TOTAL_PROB = TOTAL_DEF + TOTAL_OK

print("The total probability that a random module is defective is; ", TOTAL_DEF)
print("The total probability that a random module is okay is; ", TOTAL_OK)
print("As a sanity check, these probabilities total to; ", TOTAL_PROB)

print("\n If a module is found to be defective then the probability that it is from each company is;")
print("AM corporation  ", A_DEF/TOTAL_DEF)
print("Bryant company  ", B_DEF/TOTAL_DEF)
print("Chartair company", C_DEF/TOTAL_DEF)

#Question 3
print("\n Question 3: This scenario seems to satisfy the conditions for Poissonian probability; even a nanogram of Plutonium contains over 10^14 atoms (i.e. n -> 'infinity', but the average decay rate remains finite as the individual decay chance is small.)")

mu = 2.3*2
#Code is self explanatory here; pull the relevent poisson probabilities from the in-built function.
print("The probability that in a two-second period we observe exactly 3 decays is; ", poisson.pmf(3,mu))
print("Rather than summing all the probabilites for n > 2, we use the fact that the total probabilty for all decay sum to 1")

poisson_prob_012 = poisson.pmf(0,mu) + poisson.pmf(1,mu) + poisson.pmf(2,mu)

print("The probability that in a two-second period we observe 3 or more decays is; ", 1 - poisson_prob_012)

#Question 4
#Utilize the in-built binomial functionality. Use the probabilty mass function and the cumulative distribution function for the p-value.
print("\n Question 4: The binomial distribution is exactly the correct statistical probability to use here; we want the probability that k of n independent trails turn out a certain way, each with a set probability (n is small so definitely not in the Poisson limit). We have 16 total 'trials' each with a 2/7 chance that the neutrino event falls on the weekend.")
print("The probability that exactly 11 of the 16 events fall on the weekend is; ", binom.pmf(11,16,0.2857142857142857))
print("The probability that 11 or more of the 16 events fall on the weekend is; p-value = ", 1 - binom.cdf(10,16,0.2857142857142857))

#Question 5
x1 = np.linspace(0,10,11)			#We've seen this before; setup arrays of values and probabilities for the plots.
x2 = np.linspace(0,10,1001)
bi_pmf1 = scipy.stats.binom.pmf(x1,10,0.4)
bi_pmf2 = scipy.stats.binom.pmf(x2,10,0.4)
bi_pmf3 = scipy.stats.binom.pmf(x1,1000,0.004)
po_pmf1 = scipy.stats.poisson.pmf(x1,4)
po_pmf2 = scipy.stats.poisson.pmf(x2,4)


#First figure with binomial p*n = mu from the poisson distrivution. Not in the poisson limit of large n but constant p*n.
plot1 = plt.figure(1)
plt.plot(x1, bi_pmf1, label='Binomial distribution p=0.4')
plt.plot(x1, po_pmf1, label='Poissonian distribution mu=4')

plt.suptitle('Binom. vs Poisson. \n p*n = 0.4*10 = 4 = mu \n Not in the Poisson limit, distributions do not match.')
plt.xlabel('Positive Trials')
plt.ylabel('Probability')
plt.legend(loc="upper right")
plt.grid()

#Second figure with binomial p*n = mu from the poisson distrivution. Now in the poisson limit of large n but constant p*n. Binomial and Poisson distributions now match!
plot1 = plt.figure(2)
plt.plot(x1, po_pmf1, label='Poissonian distribution mu=4', linewidth=5.0)
plt.plot(x1, bi_pmf3, label='Binomial distribution p=0.004')

plt.suptitle('Binom. vs Poisson. \n p*n = 0.004*1000 = 4 = mu \n Now in the Poisson limit, distributions do match.') 
plt.xlabel('Positive Trials')
plt.ylabel('Probability')
plt.legend(loc="upper right")
plt.grid()

#Question 6
#Basically repeat Q5 but with Poisson and Gaussian dists.

x3 = np.linspace(300,500,201)
po_pmf3 = scipy.stats.poisson.pmf(x3,400)	#Another poisson dist. with mu far away from zero. 
ga_pdf = norm.pdf(x3, loc=400,scale=20)		#A gaussian dist. with the mean the same as the mu parameter from the poisson, add scale parameter for width/normalization of the built in normal pdf.

plot3 = plt.figure(3)
plt.plot(x3, po_pmf3, label='Poissonian mu=400', linewidth=5.0)
plt.plot(x3, ga_pdf, label='Gaussian mean=400')

plt.suptitle('Poisson. vs Gauss. \n  mu = 400 = mean for Gaussian \n Now in the limit of "N" large (i.e. Mu much larger than zero), the Poissonian distribution can be approximated by a Gaussian distribution.')
plt.xlabel('Positive Trials')
plt.ylabel('Probability')
plt.legend(loc="upper right")
plt.grid()

#Question 7
x4 = np.linspace(-1,1,101)  	#Setup some arrays of x values
x5 = np.linspace(-1,0,101)
x6 = np.linspace(0,1,101)

uni1 = scipy.stats.uniform.pdf(x4,loc=-1,scale=2)     #Make a uniform dist for fun, the cdf and survival function can form the left/right parts of our desired triangle distribution. 
cdf1 = scipy.stats.uniform.cdf(x5,loc=-1,scale=1)
sf1 = scipy.stats.uniform.sf(x6,loc=0,scale=1)
#total = cdf1 + sf1

plot4 = plt.figure(4)
#plt.plot(x4,uni1)
plt.plot(x5,cdf1, label='cdf1')
plt.plot(x6,sf1, label='sf1')
#plt.plot(x4,total,label='total')
plt.legend(loc="upper right")

#I've wasted hours on this part; I have no idea what the question is actaully getting at here. I can't find how to add the left/right parts of the distributions I created here together and draw random numbers from them so I am just going to pull numbers from a triangular distribution for the next part so I can continue. The scipy documentation is also as clear as mud when it comes to potentially feeding in a uniformly distributed list of numbers into the ppf, like the question might be telling us to do??

rands = np.random.triangular(-1, 0, 1, 100000)		#The triangle distribution that any sane person would use.
h = plt.hist(rands, bins=100, density='true')		#Plot the random values from the triangle dist. and my made up triangle distribution to show that it would have worked.
plt.show()

plot5 = plt.figure(5)

M = 10000				#Set up our number of averages to take
mu = 0					#Expected mean of the distribution
sigma_x = np.sqrt(1/6)			#Variance of the triangle distribution is 1/6, std dev is sqrt of this

for N in [1,2,5,10]:			#Loop over our sample sizes, calculating means and doing the plots with the averages and the expected gaussians from the clt.
	sigma = sigma_x/np.sqrt(N)
	s = [np.mean( np.random.triangular(-1, 0, 1, N) ) for _ in range(M) ]
	
	count, bins, ignored = plt.hist(s,25,density=True)
	plt.plot(bins, 1./(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=2, color='r')
	plt.xlabel('x')
	plt.ylabel('pdf')

	print("=================================================")    
	print("Take the average of {:d} numbers from our triangle distribution".format(N))    
	print("Expected central limit average {:.3f}".format(mu))    
	print("Sample average {:.3f}".format(np.mean(s)))    
	print("Expected central limit std. dev. {:.3f}".format(sigma))    
	print("Sample std. dev. {:.3f}".format(np.std(s)))    
	plt.show()

print("The mean of the sample is;", np.mean(rands))
print("The unbiased variance of the sample is;", np.var(rands,ddof=1))
print("The std. dev. from the unbiased variance of the sample is;", np.sqrt(np.var(rands,ddof=1)))
print("The skewness of the sample is;", scipy.stats.skew(rands))
print("The kurtosis  of the sample is;", scipy.stats.kurtosis(rands,fisher='true'))

print("\n \n Time taken; ~8.5 hours... >.<")




























