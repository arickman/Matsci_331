import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d
from math import ceil
import random
from tqdm import tqdm

# Part 1
L = 3
N = 3
M = 3
# eps = 1.67  # * 10**(-14)ergs
# sigma = 3.4 # * 10**(-8) cm
eps = 1
sigma = 1
r_c = 1.3 * sigma
nbasis = 4
threshold = 10
# alpha = 5 *  10**(-11)
# force_tol = 10**(6)
alpha = 10**(-3)
force_tol = 10**(-2)

#If r_c is set so that we consider only nearest neighbors, we can 
#solve for the equilibrium radius by differentiating the potential and 
#setting it to zero. This value will be the lattice constant a/sqrt(2): 
#a = sigma * 2**(1/6) * np.sqrt(2) therefore 
#a = 0.95*sigma * 2**(2/3)
a = sigma * 2**(2/3)

def setup_cell(L, M, N):
	#define primitive cell
	basis = np.zeros((4,3))
	basis[0,:]=[0, 0, 0]
	basis[1,:]=[0.5, 0.5, 0]
	basis[2,:]=[0, 0.5, 0.5]
	basis[3,:]=[0.5, 0, 0.5]

	#make periodic copies of the primitive cell
	natoms = 0
	#atoms = np.zeros((L*N*M*nbasis + 1, 3))
	atoms = np.zeros((L*N*M*nbasis, 3))
	for l in range(0,L):
		for m in range(0, M):
			for n in range(N):
				for k in range(nbasis):
					atoms[natoms + k, :] = basis[k,:] + [l,m,n]
				natoms += nbasis

	#atoms[natoms] = [0.5,0.5,0.5]

	#make scaled atom coordinates, i.e. all atom positions range between 0 and 1
	# atoms[:,0]=atoms[:,0]/L
	# atoms[:,1]=atoms[:,1]/M
	# atoms[:,2]=atoms[:,2]/N

	#return a*atoms,natoms + 1
	return a*atoms, natoms

def E_pot(r):
	if r < r_c:
		return 4*eps * ((sigma/r)**12 - (sigma/r)**6)
	else: return 0

def derive_pot(r):
	if r < r_c:
		return 4*eps*((6/r)*(sigma/r)**6 - (12/r)*(sigma/r)**12)
		#return 10**(8) * 4*eps*((6/r)*(sigma/r)**6 - (12/r)*(sigma/r)**12) #for part 6 only
	else: return 0


atoms, natoms = setup_cell(L, M, N)
#print(np.shape(atoms))

# Parts 2 and 3 (see page attached for 3.1 work)
def E_tot_and_force(atoms, natoms, L, M, N):
	mmax = ceil(1.5*r_c/(a*M))
	nmax = ceil(1.5*r_c/(a*N))
	lmax = ceil(1.5*r_c/(a*L))
	#calculate total energy with periodic BC’s
	#loop over periodic images of computational cell
	E_tot = 0
	forces = np.zeros((natoms,3))
	for m in range(-mmax,mmax+1):  # can take mmin = mmax
		for n in range(-nmax,nmax+1): # can take nmin = nmax
			for l in range(-lmax,lmax+1):
			#loop over atoms in computational cell 
				for i in range(0, natoms): 
					for j in range(0, natoms):
						if i == j: continue
						r_i = atoms[i,:]
						r_j = atoms[j,:]
						shift = np.array([l*L*a, m*M*a, n*N*a])
						radial_vector = r_i - r_j + shift
						E_tot += E_pot(np.linalg.norm(radial_vector, 2))
						forces[i] += derive_pot(np.linalg.norm(radial_vector,2))* (radial_vector)/np.linalg.norm(radial_vector,2)
	return 0.5*E_tot,-0.5*forces

E_tot,forces = E_tot_and_force(atoms, natoms, L, M, N)
# print(E_tot)
# print(E_tot/natoms)
print(forces[8])

#2.1 For r_c = 1.3, M,N,L = 3, E_tot/natoms = -6.0. 
#This number should stay roughly the same in magnitude as we change N,L,M,
#since we are calculating the energy per atom. 
#For N,M,L = 4, E_tot/natoms =  -6.0., as expected. 
#The total energy does increase in magnitude with N,L,M, however. 

#2.2 For r_c = 3, M,N,L = 3, E_tot/natoms = -8.129509137272224 
#This is slightly more negative because as r_c increases, we are adding more
#negative contributions from non-nearest neighbor atoms. 

#3.3 Printing out the force for a random atom, (i=8) in this case, we 
#yield [ 1.57772181e-30  1.57772181e-30 -0.00000000e+00], a value very close to zero (numerical precision errors). This is also the case for other i.  
# This is what we expect since we are in a stable atomic configuration (no vacancies yet).  

#3.4 When we increase the lattice constant, the force on the i=8 atom changes
#to [ 6.9388939e-18  6.9388939e-18 -0.0000000e+00] (a ->2a).
#The force is still roughly zero, which makes sense since we are still in a stable atomic configuration. 
#We also notice the energy/atom decreasing in magnitude as we increase a, which makes sense,
#since the atoms aren't as close together anymore and experience less attraction/repulsion from one another. 
#We also eventually go behond the cutoff and contribute zero energy from Lennard Jones. 


#Part 4
to_remove = random.randint(0,natoms - 1)
vac_atoms = np.delete(atoms, to_remove, axis=0)
E_tot_vac, forces_vac =  E_tot_and_force(vac_atoms, natoms -1, L, M, N)
# print(E_tot/natoms)
# print(E_tot_vac/natoms)

#4.1 The energy/atom drops in magnitude from -6.0 to -5.88

#4.2 See page attached for work

#4.3 
#E_vac = E_tot_vac - ((natoms - 1)/natoms)*E_tot
# print(E_vac)
# E_vac = 6.0

#4.4
#The vacancy energy above makes sense being the opposite of the 
#total energy/atom since the vacancy energy is essentially
#the energy needed to remove the atom from the lattice (--6.0 = 6.0).
#In class, we saw the ratio of cohesive energy to vacancy energy for this
#pair potential being 1. E_coh = - (E_tot/natoms) = 6.0, therefore this 
#ratio is also 1 as expected. 

#4.5
#I calculate the force on a nearest neighbor atom of the vacancy:
#print(forces_vac[to_remove + 4])
#For r_c = 1.3 we see zeros forces still as expected from the results seen in class with the cutoff set before nearest
#neighbor. #With r_c = 1.3, nothing will happen when we minimize the energy and allow relaxation. Relaxation will occur
#for a larger cutoff however: see part 4.7 and Question 5. 

#See this plot to visualize the forces
#Ben Fearon's plotting code
#PLOT CELL
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.scatter3D(atoms[:,0], atoms[:,1], atoms[:,2]);
ax.scatter3D(vac_atoms[:,0], vac_atoms[:,1], vac_atoms[:,2]);
ax.quiver(vac_atoms[:,0],vac_atoms[:,1],vac_atoms[:,2], forces_vac[:,0], forces_vac[:,1], forces_vac[:,2])
fig.savefig("forces.pdf")


#4.6
def converged_E_vac(threshold):
	E_vac_old = 100000000000000000
	for l in range(2,10):
		atoms,natoms = setup_cell(l,l,l)
		to_remove = random.randint(0,natoms - 1)
		vac_atoms = np.delete(atoms, to_remove, axis=0)
		E_vac = E_tot_and_force(vac_atoms, natoms -1, l, l, l)[0] - ((natoms - 1)/natoms)*E_tot_and_force(atoms, natoms, l, l, l)[0]
		#print(E_vac)
		if np.abs(E_vac - E_vac_old) < threshold: 
			break
		E_vac_old = E_vac
	return E_vac, l

# E_vac_converged, LMN_opt = converged_E_vac(threshold)
# print(E_vac_converged)
# print(LMN_opt)

#We see a converged vacancy energy of 8.12950913727218 with L = M = N = 3 with a threshold of 0.1. 

#4.7 The forces on the atoms around the vacancy are now appreciable when we move to the r_c = 3 case:
#[-3.13931148e-03  2.56202393e-16 -6.27862296e-03] for an atom 4 slots from the vacancy for example. 
#This shows a force of essentially 0 for small r_c and a non-trivial force for larger r_c as desired since
#r_c = 3 allows for beyond nearest-neigbor interactions.   

#Part 5
#5.1 
def minimize_E(vac_atoms, natoms, L,M,N, alpha):
	for i in tqdm(range(100)):
		E_tot, forces_vac = E_tot_and_force(vac_atoms, natoms - 1, L, M, N)
		print(np.amax(np.abs(forces_vac)))
		if np.amax(np.abs(forces_vac)) < force_tol: 
			#print("Converged in {} steps.".format(i))
			break
		vac_atoms += alpha*forces_vac
		if i == 99 : print("Didn't converge on optimal atom positions")
	return vac_atoms, E_tot

#To visualize the movement of the atoms:
#vac_atoms_opt = minimize_E(vac_atoms, natoms, L, M, N, alpha)[0]
# movement = (vac_atoms - vac_atoms_opt)*200
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter3D(atoms[to_remove,0], atoms[to_remove,1], atoms[to_remove,2]);
# ax.scatter3D(vac_atoms[:,0], vac_atoms[:,1], vac_atoms[:,2]);
# ax.quiver(vac_atoms[:,0],vac_atoms[:,1],vac_atoms[:,2], movement[:,0], movement[:,1], movement[:,2])
# fig.savefig("forces_opt.pdf")

#5.2 
def converged_rel_E_vac(threshold):
	E_vac_old = 1000000000000000
	for l in range(2,10):
		atoms,natoms = setup_cell(l,l,l)
		to_remove = random.randint(0,natoms - 1)
		vac_atoms = np.delete(atoms, to_remove, axis=0)
		vac_atoms_opt = minimize_E(vac_atoms, natoms, l, l, l, alpha)[0]
		E_vac = E_tot_and_force(vac_atoms_opt, natoms - 1, l, l, l)[0] - ((natoms - 1)/natoms)*E_tot_and_force(atoms, natoms, l, l, l)[0]
		if np.abs(E_vac - E_vac_old) < threshold: 
			break
		print("converged loop")
		print(E_vac)
		print(E_vac_old)
		E_vac_old = E_vac
		if l ==10 : print("Did not converge")
	return E_vac, l

# E_vac_rel_converged, LMN_rel_opt = converged_rel_E_vac(threshold)
# print(E_vac_rel_converged)
# print(LMN_rel_opt)

#For convergence of the vacancy, we again find an optimal value of L = M = N = 3 when we now allow relaxation (threshold = 0.1).
#The converged value here is E_vac = 8.11552520433304. The energy clearly decreased from the unrelaxed case as desired,
#but the cell size is unchanged. 

#5.3
#Changing the lattice constant to 0.95a still requires a computational cell size 
# of L = N = M = 3 for convergence in the case of no relaxation (threshold = 0.1), and the energy 
#decreased to 8.08542532979493. 

#5.4
#In the case of relaxation, we again find the computational cell size for 0.95a to be
#L = N = M = 3 (threshold = 0.1), and the energy decreased even further to 7.711402741787879. 

#We find the same computational cell size required for convergence in the strained vs unstrained case,
#but the energy is now lower in the strained case. This makes sense since it is easier to remove an atom from a crystal under pressure. 

#5.5
def ratio(vac_atoms, natoms, L, M, N):
	num = minimize_E(vac_atoms, natoms, L, M, N, alpha)[1]
	E_coh = num/(natoms - 1)
	E_vac = converged_rel_E_vac(threshold)[0]
	ratio = E_coh/E_vac
	return E_coh, E_vac, ratio

# test = minimize_E(vac_atoms, natoms, 3, 3, 3, alpha)[1]
# print(test/(natoms - 1))
#E_coh = -8.013356666729525
#ratio = ratio(vac_atoms,natoms, L, M, N)
#print(ratio)

#Compute the ratio of the converged vacancy energy to the cohesive energy and compare to the values of around 0.3 that are observed for metals.
# E_coh = -8.013356666729525 from above, and from the previous calculation with relaxation we have the converged vacancy energy as 7.711402741787879. 
#Thus, the ratio is roughly 0.96. 

#5.6
#As seen above, the allowed relaxation does bring the ratio closer to 0.3, but only slightly (it is still very close to 1). Therefore,
#it seems we need more than just relaxation to accound for this discrepancy. 

#Part 6
#r_c = 10, a = sigma * 2**(2/3), threshold for convergence changed to 10, alpha =5 * 10**-11, force_tol =  10**(6) (for appropriate scaling)
#6.1
def density(T, relaxation):
	denom = 1.38 * T
	if relaxation: num = converged_rel_E_vac(threshold)[0]
	else: num = converged_E_vac(threshold)[1]
	return np.exp(- 100 * num/denom)

# d = density(20, True)
# print(d)
# We see an equilibrium concentration with relaxation of 4.4162325e-21, with E_vac = 12.935904455460388 * 10^-14 ergs. 


#6.2
#vacancies/site * nsites_min = 1, solve for n_sites_min
#print(1/d) 
#We find 2.2643735e+20 sites minimum. Since we have 108 atoms per computational
#cell here, we need roughly 2.0966422e+18 cells for this crystal. 


#6.3
#print(density(20, False))
# Now we have an equilibrium concentration without relaxation of 5.6682025e-22, with E_vac = 13.50266030075818 * 10^-14 ergs,
#smaller than above by an order of magnitude, so the atomic relaxation does impact the density appreciably.  

#7.1
#alpha is the hyper-parameter associated with the gradient descent minimization above. If we make
#alpha too big, we jump over the minimum repeatedly, and we don't see monotonic decrease in the energy. 
#If alpha is too small, it simply takes too long to converge. In part 6 for example, the alpha appropriate for 
#gradient descent for the earlier parts was now too big and missed the minimum, so I decreased it,
#but not too low so as to converge very slowly. 

#7.2
#Now we put in an interstital in the case of the density calculations above, and we immediately find that
# alpha is now too small, and takes many more steps to converge than without the interstitial. Before it took 
#roughly 27 steps to converge (talking about the energy minimization of course), and now we see orders of magnitude more steps.
# We need to increase alpha to get this algorithm to converge to the same tolerance as before. So clearly the initial atomic configuration does affect
#the optimal alpha value, and thus when optimizing this hyperparameter, we can look to the initial atomic configuration when possible.  




