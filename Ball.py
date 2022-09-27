
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

g = -9.8
H0 = 10
u0 = 10

t1 = np.linspace(0, 3, 5000)
t2 = np.linspace(0.25, 3, 5000)
s1 = t1.copy()
s2 = t1.copy()
y1 = t1.copy()
y2 = t1.copy()
v1 = t1.copy()
v2 = t1.copy()

for i in range(len(t1)):
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

plot1 = plt.figure(1)
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

plt.show()
