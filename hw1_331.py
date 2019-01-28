import matplotlib.pyplot as plt
import numpy as np

#Problem 2
eps = 0.58 #eV
a = 6.58 #Angstrom -1
r_e = 2.553 #Angstromg

def V(r):
	return eps*(np.exp(2*a*(r_e - r)) - 2*np.exp(a*(r_e - r)))

r = np.linspace(0, 10, 1000)
fig,ax = plt.subplots()
ax.set_ylim(-1,1)
ax.plot(r, V(r))
ax.set_xlabel("r")
ax.set_ylabel("Potential Energy")
ax.set_title("Morse Potential")
fig.savefig("morse_pot.pdf")
plt.show()


#Problem 4
#Part 1
N = 5
R_0 = 1
q_tilde = 1

def q_func(j):
	return (-1)**j * q_tilde


def pot_calc():
	pot = 0
	for i in range(N):
		for j in range(i+1,N):
			num = q_func(i)*q_func(j)
			denom = (j-i)*R_0
			pot += num/denom
	return pot

#Part 2
#The numerator within the sum is negative for nearest neighbors (largest
#contribution), so the lowest energy (largest negative value) occurs 
#when we minimize the denominator. This tells us that the interatomic 
#separation giving the lowest energy is that which approaches zero. 

#Part 3
print(-q_tilde**2/R_0 * np.log(2))
print(pot_calc()/N)
#For 2 sig figs, N is approximately 160.
#For 3 sig figs, N is approximatley 4000, more than a factor of 10 higher.

#Part 4
#Above, we found that even with only 100 atoms or so, we are seeing 
#a result consistent with the Bulk limit (to 2 sig figs). This implies that
#we don't need many atoms to exchibit this characteristic, so the maximum
#particle chain size isn't very large. Quantitatively, we are still
#within 1 significant figure within the limit when we drop to just 6 atoms. 
