import numpy as np                                              
import matplotlib.pyplot as plt                                 
#from scipy.stats import norm   
from scipy.stats import skewnorm

t1 = skewnorm.rvs(a=4, loc=1.5)
t2 = np.random.triangular(left=-0.2, mode=0.2, right=0.2)
t3 = np.sin(np.pi*2*np.random.rand())
wait = t1 + t2 + t3
