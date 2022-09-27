import numpy as np
import matplotlib.pyplot as plt

Resistances = np.array([100, 200, 300, 400])
Powers = np.array([[1.05, 0.08], [0.49, 0.02], [0.31, 0.03], [0.23, 0.02]])
Resistors = np.array(['R1', 'R2', 'R3', 'R4'])

plt.figure()

plt.errorbar(Resistances, Powers[:,0], yerr = Powers[:,1], ls='none', marker ='^', color='blue', elinewidth=2, label='Measured Power Dissapation')

x = np.linspace(90, 410, 320)
p = x.copy() 

for i in range(len(x)):
	p[i] = 100/x[i]

plt.plot(x, p, label='Ohm\'s Law Prediction', linestyle='dashed')

plt.suptitle('Measured Power Dissapation and Ohm\'s law prediction')
plt.xlabel(r'Resistance ([$\Omega$])')
plt.ylabel('Power (W) ')
plt.legend(loc="upper right")

for i, txt in enumerate(Resistors):
    plt.annotate(txt, (Resistances[i]+0.2, Powers[i,0]+0.05))

plt.show()
