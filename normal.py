import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

mu = 0
sigma = 1

s = np.random.normal(mu, sigma, 500)
#plt.hist(s, 40, density=True)

weights = np.ones_like(s)/sum(s)
plt.hist(s, 40, weights=weights)

xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, sigma)

plt.plot(x, p, 'k', linewidth=2)

title = "500, normalized, random numbers drawn from \n a Gaussian distribution of mean = 0 and sigma = 1"
plt.title(title)

plt.show()
