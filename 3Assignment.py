#Assignment 3. Using PACE Conda env. 
#Unzipped file with gzip prior to running this code.

import numpy as np                                              #Import Numpy
import matplotlib.pyplot as plt                                 #Import Matplotlib
import scipy
from scipy.stats import norm                                    #Import Gaussian curve from scipy.stats
from scipy.stats import poisson
from scipy.stats import binom
from scipy.stats import uniform

all_data = np.genfromtxt('Births_US_2018.txt', dtype='unicode') 	#Read in the data text file.

# (A) RNG chose Male for rest of this study.
just_male_data = all_data[all_data[:,3] == 'M']   		#Only keep rows where they are males.
male_data = np.delete(just_male_data,3,1).astype(np.int) 	#Delete the column with sex as we know the remaining are all male, cast all elements as integers for easy comparisons and cleaning later.

# (B) To begin cleaning the data let's make some plots and try and pick out entries that are incorrect to the dataset.
weights = male_data[:,0]   	#Seperate the data for quick plotting, not necessary but whatever. Note the change of index of column of array for consultations and mother weight as I've deleted the sex column to cast all the elements as integers.
cigarettes = male_data[:,1]
mother_age = male_data[:,2]
consultations = male_data[:,3]
mother_weightgain = male_data[:,4]

#Quick code for the histograms, I checked these visually in a separate ipython session.
#plot1 = plt.figure(1)
#h = plt.hist(mother_weightgain, bins=100)
#plt.show()
#We can also check for certain number specifically e.g. 
# (cigarettes == 98).sum() or (cigarettes < 0).sum()


weights = weights[weights[:] < 9999]		#Remove any weights that are 9999. Obviously these aren't real weights in the dataset.
cigarettes = cigarettes[cigarettes[:] < 98]  	#Remove entries where 99 (or 98; there seems to be an abnormal number of entries of 98. This is probably due to mistyping '99') is entered for 'no data'. I'm not a demographer but I think if you smoked 98 a day, you'd be dead.
#Mother Ages look fine; nothing less than zero, or larger than 50. The 50 bin, where over 50 is also reported doesn't look too large as to be skewing the data significantly. We could remove it but I'll leave it in. (Although there are some ages 13 (!?!).) 
consultations = consultations[consultations[:] < 97] #Same argument as for cigarettes, likely to be mistypes for 98/97 entries. One entry of 93, this could just be a very extreme case; we aren't demographers.
mother_weightgain = mother_weightgain[mother_weightgain[:] < 98] #Same as for cigarettes case. Looking at the plotted distribution it seems that entries of 98 are more likely to be mistypes than real entries.

#Make the above cuts on the whole male dataset. Total male entries 1943273.
male_data = male_data[male_data[:,0] < 9999]	#Entries after cut 1941672.
male_data = male_data[male_data[:,1] < 98]	#Entries after cut 1932057.
male_data = male_data[male_data[:,3] < 97]	#Entries after cut 1884734.
male_data = male_data[male_data[:,4] < 98]	#Entries after cut 1834588.
print('The total number of remaining entries in the male dataset after cleaning is;', len(male_data))

# (C)
n_bins = 60 		#Number of bins for plots; tried 200/100/75/50, 60 looks best. Not too coarse or fine; features clearly visible and not washed out, but no 'comb effect'.
clean_weights = male_data[:,0]

q10 = np.quantile(clean_weights, 0.10)		#calculate the x values for the quantiles.
q25 = np.quantile(clean_weights, 0.25)
q50 = np.quantile(clean_weights, 0.50)
q75 = np.quantile(clean_weights, 0.75)
q90 = np.quantile(clean_weights, 0.9)

plot1 = plt.figure(1)
weight_pdf_plot = plt.hist(clean_weights, n_bins, density=True, histtype='step', label=' Weight Marginal PDF', linewidth=2.0)

plt.axvline(x=q10, color='green', linestyle='dashed', label='10% Quantile')		#Plot the PDF and quantiles for fun.
plt.axvline(x=q25, color='yellow', linestyle='dashed', label='25% Quantile')
plt.axvline(x=q50, color='orange', linestyle='dashed', label='50% Quantile')
plt.axvline(x=q75, color='red', linestyle='dashed', label='75% Quantile')
plt.axvline(x=q90, color='purple', linestyle='dashed', label='90% Quantile')

plt.suptitle('Male Birth Weight Marginal PDF from Cleaned CDC Data.')
plt.xlabel('Birth Weight (g)')
plt.ylabel('PDF')
plt.legend(loc="upper right")
plt.grid()

# (D)
plot2 = plt.figure(2)
weight_cdf_plot = plt.hist(clean_weights, n_bins, density=True, cumulative=True, histtype='step', label=' Weight CDF', linewidth=2.0)

plt.axvline(x=q10, color='green', linestyle='dashed', label='10% Quantile')		#Plot the CDF and quantiles.
plt.axvline(x=q25, color='yellow', linestyle='dashed', label='25% Quantile')
plt.axvline(x=q50, color='orange', linestyle='dashed', label='50% Quantile')
plt.axvline(x=q75, color='red', linestyle='dashed', label='75% Quantile')
plt.axvline(x=q90, color='purple', linestyle='dashed', label='90% Quantile')

plt.suptitle('Male Birth Weight Marginal CDF from Cleaned CDC Data.')
plt.xlabel('Birth Weight (g)')
plt.ylabel('CDF')
plt.legend(loc="upper left")
plt.grid()


# (E)
print('The mean male weight at birth, in grams, is;', np.mean(clean_weights))
print('The standard deviation of male weight at birthi, in grams, is;', np.std(clean_weights))

# (F) (RNG picked weight gain by mother as other quantity 'A')

clean_mother_weight_gain = male_data[:,4]

plot3 = plt.figure(3)		#Same as above; get data from cleaned set and plot it's pdf and quantiles for fun.
mother_weight_pdf_plot = plt.hist(clean_mother_weight_gain, 20, density=True, histtype='step', label=' Mothers Weight Gain \n Marginal PDF', linewidth=2.0)

plt.axvline(x=np.quantile(clean_mother_weight_gain, 0.10), color='green', linestyle='dashed', label='10% Quantile')
plt.axvline(x=np.quantile(clean_mother_weight_gain, 0.25), color='yellow', linestyle='dashed', label='25% Quantile')
plt.axvline(x=np.quantile(clean_mother_weight_gain, 0.50), color='orange', linestyle='dashed', label='50% Quantile')
plt.axvline(x=np.quantile(clean_mother_weight_gain, 0.75), color='red', linestyle='dashed', label='75% Quantile')
plt.axvline(x=np.quantile(clean_mother_weight_gain, 0.90), color='purple', linestyle='dashed', label='90% Quantile')

plt.suptitle('Mother Weight Gain Marginal PDF from Cleaned CDC Data.')
plt.xlabel('Mother Weight Gain (pounds)')
plt.ylabel('PDF')
plt.legend(loc="upper right")
plt.grid()

# (G)

plot4 = plt.figure(4)
#Plot the 2d histogram of the joint pdf, could use scatter plot or contour plot but I like this one with the colors. Couldn't get a scale to show up? Doesn't really matter anyway. The brighter the color the larger the join pdf is.
joint_pdf_plot = plt.hist2d(clean_weights, clean_mother_weight_gain, bins=[n_bins,20], density=True, label='Joint PDF')

plt.suptitle('Male Birth Weight and Mother Weight Gain Joint PDF.')
plt.xlabel('Birth Weight (g)')
plt.ylabel('Mother Weight Gain (pounds)')

# (H)
cov_mat = np.cov(clean_weights, clean_mother_weight_gain)	#Calculate the covariance matrix and the correlation coefficients which are just normalized to std. devs. so it's easier to look at than the covariance matrix. There's a small positive correlation between the quantities.
corr_mat = np.corrcoef(clean_weights, clean_mother_weight_gain)

print('\nThe matrix of correlation coefficients, between birth weight and mother weight gain, is; \n \n', corr_mat)
print('\nWe see that there is a small positive correltion between the quantities')

# (I)
joint_hist_data = joint_pdf_plot[0] 	#Get the info from the joint pdf histogram
weight_hist_data = np.divide(weight_pdf_plot[0],np.amax(weight_pdf_plot[0])/np.amax(joint_hist_data[:,4]))	#Get the info from the weight pdf histogram. Quick and dirty normalization here for the plot, I'd have to take more time to see if this is actually correct, but I've already spent too long on this project and these matplotlib plots below don't have a quick integral normalization option.
slice_bin_5 = joint_hist_data[:,4] #Choose a couple of bins for the cdf. Corresponds to mother weight gain of 22 pounds and 31 pounds respectively. 
slice_bin_7 = joint_hist_data[:,6] #As the quantities are only slightly correlated it doesn't show a drastic difference in the plot. Could pick some different bins in the joint pdf if you like. But we see that the green curve is transposed slightly right of the orange curve and the blue pdf is kind of in the middle.

bin_centers = np.delete(weight_pdf_plot[1],0) #Grab the bin edges and turn them into centers.
width = bin_centers[1] - bin_centers[0]
bin_centers = np.subtract(bin_centers, width/2)

plot5 = plt.figure(5)

weight_pdf_plot_2 = plt.plot(bin_centers, weight_hist_data, label=' Weight Marginal PDF', linewidth=2.0) #Finally plot these things.
weight_cpdf_plot_5 = plt.plot(bin_centers, slice_bin_5, label=' Weight Conditional PDF \n for Mother weight \n  gain of 22 pounds', linewidth=2.0)
weight_cpdf_plot_7 = plt.plot(bin_centers, slice_bin_7, label=' Weight Conditional PDF \n for Mother weight \n gain of 31 pounds', linewidth=2.0)

plt.suptitle('Weight Marginal PDF and Conditional PDF \n for Mother Weight Gain of 22 and 31 pounds.')
plt.xlabel('Birth Weight (g)')
plt.ylabel('PDF')
plt.legend(loc="upper right")

plt.show()

#(2)
print('Time spent on this project... around 8.5-9 hours or so I think.')
































