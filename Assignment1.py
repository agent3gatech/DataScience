#Assignment 1. Using PACE Conda env. As per the assignment instruction things are only printed if asked for; I assume the actual notes and code in this script is what is primarily being assessed?

import numpy as np						#Import Numpy
import matplotlib.pyplot as plt					#Import Matplotlib
from scipy.stats import norm					#Import Gaussian curve from scipy.stats

B = np.random.randint(-2, 6, size=(10,10)) 			#Generate our array of random ints. Note that the upper bound (6) is exclusive.

Bt = B.transpose() 						#Generate the transpose of B

# We can, trivially write B = 0.5*(B + Bt) + 0.5*(B - Bt)
# You can quickly convince yourself that A = 0.5*(B + Bt) is the symmetric matrix we need and C = 0.5*(B - Bt) is the skewsymmetic matrix we need that, by desing, sum to B.

A = (B + Bt)/2
C = (B - Bt)/2

At = A.transpose()
Ct = C.transpose()

np.allclose(A, At, rtol=1e-05, atol=1e-08, equal_nan=False) 	#Returns true if A = Transpose of A. The definition of symmetric.
np.allclose(C, -Ct, rtol=1e-05, atol=1e-08, equal_nan=False)	#Returns true if C = The negative of the transpose of C. The definition of skewsymmetric.

Adiag = np.diagonal(A)						#Get the diagonal elements of A.

D = A.copy()							#Create a new matrix, D, by copying A as fill_diagonal will fill in place.
np.fill_diagonal(D,-Adiag) 					#Fill the diagonal of D with the negative of the diagonal elements of A.

detA = np.linalg.det(A)						#Calcualte the determinant of A. 
Ainv = np.linalg.inv(A)						#Calculate the inverse of A. 

One = A@Ainv							#Matrix multplication of A and Ainv

I10 = np.identity(10)						#Create 10x10 identity
np.allclose(One, I10, rtol=1e-10, atol=1e-10, equal_nan=False)  #Check that One is consistent with identity to greater than 1e-7 absolute precision. Relative precision is orders of magnitude better than this too.

print("Definition of tolerance(s) from the documentation; Returns true if absolute(a - b) <= (atol + rtol * absolute(b))")		#Can't find any definition of relative 'precision' in numpy documentation. Here is how the absolute and relative tolerances are used to check consistency in allclose.

V = np.random.randint(-10, 11, size=(10,1))			#Create our column vector V, again upper bound is exclusive.

X = Ainv@V							#To solve AX = V we simply pre-multiply both sides with Ainv.

Evalues, Evectors = np.linalg.eig(A)				#Make arrays that contain the eigenvalues and eigenvectors of A.

np.allclose(A@Evectors[:,1], Evalues[1]*Evectors[:,1], rtol=1e-05, atol=1e-08, equal_nan=False) 	#We can check that this satisfies the eigenvalue equation, as here.

np.allclose(detA, np.prod(Evalues), rtol=1e-14, atol=1e-14, equal_nan=False) 	#The product of all eigenvalues is consistent with the determinant of A to relative/absolute tolerance of 1e-14

np.sum(Evalues)							#We sum the eigenvalues and see that they are consistent with the trace of A.
np.trace(A)

np.sort(X, axis=0)[::-1]               				#Sort elements in X from largest to smallest.

#Part 2 Plotting Ball Dynamics

g = -9.8							#Define gravitional acceleration
H0 = 10								#Initial height ball 1 is dropped from
u0 = 10								#Initial upward velovity that ball 2 is thrown with

t1 = np.linspace(0, 3, 5000)					#Define arrays for time, displacement, height and velocity for 2 balls
t2 = np.linspace(0.25, 3, 5000)
s1 = t1.copy()
s2 = t1.copy()
y1 = t1.copy()
y2 = t1.copy()
v1 = t1.copy()
v2 = t1.copy()

for i in range(len(t1)):					#Fill the arrays with values calculated from elementary dynamics
        s1[i] = (g*t1[i]*t1[i])/2
        y1[i] = H0 + s1[i]
        v1[i] = g*t1[i]
        if y1[i] < 0:
                y1[i] = 0
                v1[i] = 0

for i in range(len(t1)):
        s2[i] = u0*t1[i] + (g*t1[i]*t1[i])/2
        y2[i] = s2[i]
        v2[i] = u0 + g*t1[i]
        if y2[i] < 0:
                y2[i] = 0
                v2[i] = 0

plot1 = plt.figure(1)						#Generate, label and make pretty the plots
plt.plot(t1, y1, 'b', label='Ball 1 Height (m)')
plt.plot(t2, y2, 'r', label='Ball 2 Height (m)', linestyle='dashed', linewidth=5.0)

plt.suptitle('Height vs Time of Two Balls')
plt.xlabel('Time (s)')
plt.ylabel('Height (m)')
plt.legend(loc="upper right")
plt.grid()


plot2 = plt.figure(2)
plt.plot(t1, v1, 'b', label='Ball 1 Velocity (m/s)')
plt.plot(t2, v2, 'r', label='Ball 2 Velocity (m/s)', linestyle='dashed', linewidth=5.0)

plt.suptitle('Velocity vs Time of Two Balls')
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.legend(loc="upper right")
plt.grid()

#plt.show()

#Part 3 Plotting Power Dissapation with errors

Resistances = np.array([100, 200, 300, 400])					#Setup our arrays with resistances, errors, power and names
Powers = np.array([[1.05, 0.08], [0.49, 0.02], [0.31, 0.03], [0.23, 0.02]])
Resistors = np.array(['R1', 'R2', 'R3', 'R4'])

plot3 = plt.figure(3)				#Plot the values and make pretty

plt.errorbar(Resistances, Powers[:,0], yerr = Powers[:,1], ls='none', marker ='^', color='blue', elinewidth=2, label='Measured Power Dissapation')

x = np.linspace(90, 410, 320)
p = x.copy()

for i in range(len(x)):		#Array of values fitting Ohm/s law
        p[i] = 100/x[i]

plt.plot(x, p, label='Ohm\'s Law Prediction', linestyle='dashed')		#Make and pretty up the plot

plt.suptitle('Measured Power Dissapation and Ohm\'s law prediction')
plt.xlabel(r'Resistance ($\Omega$)')
plt.ylabel('Power (W) ')
plt.legend(loc="upper right")

for i, txt in enumerate(Resistors):
    plt.annotate(txt, (Resistances[i]+0.2, Powers[i,0]+0.05))

#plt.show()

# Part 4 Gaussian Distribution Histogram

mu = 0
sigma = 1

s = np.random.normal(mu, sigma, 500)		#Generate our array of 500 values drawn from our gaussian dist.

plot4 = plt.figure(4)
plt.hist(s, 40, density=True)			#Plot the values in 40 bins and normalize such that the area of the histogram is equal to 1. N.B. the comment on the assignment has really confused me. Dividing, or not, by the binwidth either normalizes the area, or the counts, to 1. For me the definition of density=true in the documentation is exaclty what I would mean by normalizing a histogram. I.E. the area of the hsitogram is important, not the sum of the counts, so the binwidth should be accounted for as density=true does. This might be a mismatch of terminology, or I am missing some subtlety??? If you really wanted the counts to be normalized to one then you could loop over all the bin contents and set them to be divided by the sum of the magnitudes of the array.

xmin, xmax = plt.xlim()				#Get the range of the x values for the normal curve
x = np.linspace(xmin, xmax, 100)		#Generate the x values for the normal curve
p = norm.pdf(x, mu, sigma)			#Generate the normal curve; note here how is is normalized such that the intrgral of it is equal to 1.

plt.plot(x, p, 'k', linewidth=2)

title = "500 random numbers drawn from \n a Gaussian distribution of mean = 0 and sigma = 1 \n Normalized such that the area of the histogram equals 1"
plt.title(title)

#plt.show()

print("Time spent on this assignment ~8 hours. (Having coded pretty much exclusively in C++ before, I spent a lot of time reading about exactly how python/numpy etd actually works as, obviosuly, the structure and behavior is very far from C.)") 

plt.show()

