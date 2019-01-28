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
eps = 1
sigma = 1
r_c = 3
nbasis = 4
threshold = 0.5
alpha = 10**(-3)
force_tol = 10**(-1)

#If r_c is set so that we consider only nearest neighbors, we can 
#solve for the equilibrium radius by differentiating the potential and 
#setting it to zero. This value will be the lattice constant a/sqrt(2): 
#a = sigma * 2**(1/6) * np.sqrt(2) therefore 
a = 0.95*sigma * 2**(2/3)

def setup_cell(L, M, N):
	#define primitive cell
	basis = np.zeros((4,3))
	basis[0,:]=[0, 0, 0]
	basis[1,:]=[0.5, 0.5, 0]
	basis[2,:]=[0, 0.5, 0.5]
	basis[3,:]=[0.5, 0, 0.5]

	#make periodic copies of the primitive cell
	natoms = 0
	atoms = np.zeros((L*N*M*nbasis, 3))
	for l in range(0,L):
		for m in range(0, M):
			for n in range(N):
				for k in range(nbasis):
					atoms[natoms + k, :] = basis[k,:] + [l,m,n]
				natoms += nbasis

	#make scaled atom coordinates, i.e. all atom positions range between 0 and 1
	# atoms[:,0]=atoms[:,0]/L
	# atoms[:,1]=atoms[:,1]/M
	# atoms[:,2]=atoms[:,2]/N

	return a*atoms,natoms

def E_pot(r):
	if r < r_c:
		return 4*eps * ((sigma/r)**12 - (sigma/r)**6)
	else: return 0

def derive_pot(r):
	if r < r_c:
		return 4*eps*((6/r)*(sigma/r)**6 - (12/r)*(sigma/r)**12)
	else: return 0


atoms, natoms = setup_cell(L, M, N)

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
# print(E_tot/natoms)
# print(forces[8])

#2.1 For r_c = 1.3, M,N,L = 3, E_tot/natoms = -6.0. 
#This number should stay roughly the same in magnitude as we change N,L,M,
#since we are calculating the energy per atom. 
#For N,M,L = 4, E_tot/natoms =  -6.0., as expected. 
#The total energy does increase in magnitude with N,L,M, however. 

#2.2 For r_c = 3, M,N,L = 3, E_tot/natoms = -8.129509137272224 
#This is slightly more negative because as r_c increases, we are adding more
#negative contributions from non-nearest neighbor atoms. 

#3.3 Printing out the force for a random atom, (i=8) in this case, we 
#yield [1.38777878e-17 4.16333634e-17 2.49800181e-16], very close to zero (numerical precision errors). 
# This is what we expect since we are in a stable atomic configuration (no vacancies yet).  

#3.4 When we double the lattice constant, the force on the i=8 atom changes
#to [ 6.9388939e-18  6.9388939e-18 -0.0000000e+00].
#The force is still roughly zero, which makes sense since we are still in a stable atomic configuration. 
#We also notice the energy/atom decreasing in magnitude to -0.18603515625000003, which makes sense,
#since the atoms aren't as close together anymore and experience less attraction/repulsion from one another. 


#Part 4
to_remove = random.randint(0,natoms - 1)
vac_atoms = np.delete(atoms, to_remove, axis=0)
E_tot_vac, forces_vac =  E_tot_and_force(vac_atoms, natoms -1, L, M, N)
# print(E_tot/natoms)
# print(E_tot_vac/natoms)

#4.1 The energy/atom drops in magnitude from -6.0 to -5.88

#4.2 See page attached for work

#4.3 
E_vac = E_tot_vac - ((natoms - 1)/natoms)*E_tot
#print(E_vac)
#E_vac = 6.0


#4.4
#The vacancy energy aboce makes sense being the opposite of the 
#total energy/atom since the vacancy energy is essentially
#the energy needed to remove the atom from the lattice (--6.0 = 6.0).
#In class, we saw the ratio of cohesive energy to vacancy energy for this
#pair potential being 1. E_coh = - (E_tot/natoms) = 6.0, therefore this 
#ratio is also 1 as expected. 

#4.5
#I calculate the force on a nearest neighbor atom of the vacancy:
#print(forces_vac[to_remove + 3])
#For r_c = 1.3
# [ 3.03090886e-14  2.52063676e-01 -2.52063676e-01], with non-trivial forces in two of the component directions
#The force increased around the vacancy as expected. If you minimize energy around the vacancy, we will see 
# relaxation/movement of particles towards the vacancy. 


#Ben Fearon's plotting script
# #PLOT CELL
# fig = plt.figure()
# ax = plt.axes(projection='3d')
# ax.scatter3D(atoms[:,0], atoms[:,1], atoms[:,2]);
# ax.scatter3D(vac_atoms[:,0], vac_atoms[:,1], vac_atoms[:,2]);
# ax.quiver(vac_atoms[:,0],vac_atoms[:,1],vac_atoms[:,2], forces_vac[:,0], forces_vac[:,1], forces_vac[:,2])
# plt.show()


#4.6
def converged_E_vac(threshold):
	E_vac_old = 100000000000000000
	for l in range(3,10):
		atoms,natoms = setup_cell(l,l,l)
		to_remove = random.randint(0,natoms - 1)
		vac_atoms = np.delete(atoms, to_remove, axis=0)
		E_vac = E_tot_and_force(vac_atoms, natoms -1, l, l, l)[0] - ((natoms - 1)/natoms)*E_tot_and_force(atoms, natoms, l, l, l)[0]
		if np.abs(E_vac - E_vac_old) < threshold: 
			break
		E_vac_old = E_vac
	return E_vac, l

E_vac_converged, LMN_opt = converged_E_vac(threshold)
print(E_vac_converged)
print(LMN_opt)
#We see a converged vacancy energy of 8.234344445653733 with L = M = N = 4 with a threshold of 0.5. 

#4.7 The forces on the nearest neighbor atoms are essentially unchanged in magnitude when we 
# move to the r_c = 3 case, however, we now see appreciable forces exerted on non-nearest neighbor
# atoms as expected. For 2 steps beyond nearest neighbor from the vacancy for example, 
# we moved from a force of [-5.55111512e-17  1.11022302e-16 -5.55111512e-17] (r_c = 1.3)
# to [-2.91433544e-16  3.11558880e-02  3.11558880e-02] (r_c = 3). This shows a force of essentially 0 
# for small r_c and a non-trivial force for larger r_c as desired since r_c = 3 allows for beyond nearest-neigbor interactions.   

#Part 5
#5.1 
def minimize_E(vac_atoms, natoms, L,M,N, alpha):
	forces_vac_old =  np.zeros((natoms - 1,3))
	for i in tqdm(range(100000)):
		E_tot, forces_vac = E_tot_and_force(vac_atoms, natoms - 1, L, M, N)
		if np.linalg.norm(forces_vac - forces_vac_old) < force_tol: 
			print("Converged in {} steps.".format(i))
			break
		vac_atoms += alpha*forces_vac
		forces_vac_old = forces_vac
		if i == 99999 : print("Didn't converge on optimal atom positions")
		#print(E_tot)
	return vac_atoms

#5.2 
def converged_rel_E_vac(threshold):
	E_vac_old = 100000000000000000
	for l in range(3,10):
		atoms,natoms = setup_cell(l,l,l)
		to_remove = random.randint(0,natoms - 1)
		vac_atoms = np.delete(atoms, to_remove, axis=0)
		vac_atoms_opt = minimize_E(vac_atoms, natoms, l, l, l, alpha)
		E_vac = E_tot_and_force(vac_atoms_opt, natoms - 1, l, l, l)[0] - ((natoms - 1)/natoms)*E_tot_and_force(atoms, natoms, l, l, l)[0]
		if np.abs(E_vac - E_vac_old) < threshold: 
			break
		E_vac_old = E_vac
		print(l)
		if l ==10 : print("Did not converge")

	return E_vac, l

E_vac_rel_converged, LMN_rel_opt = converged_rel_E_vac(threshold)
print(E_vac_rel_converged)
print(LMN_rel_opt)

#For convergence of the vacancy, we find an optimal value of L = M = N = 4 when we now allow relaxation (threshold = 0.5).
#The converged value here is E_vac = 8.210800820089844. The energy clearly decreased from the unrelaxed case, but the cell size is unchanged. 

#5.3
#Changing the lattice constant to 0.95a also requires a computational cell size 
# of L = N = M = 4 in the case of no relaxation (threshold = 0.5), but the energy 
#decreased to 8.08542532974343. 

#5.4
#In the case of relaxation, we find the new computational cell size for 0.95a to be
#L = N = M = (threshold = 0.5). The energy decreased even further to . 

#How do these results differ from the unstrained computational cell case?


#5.5
#Compute the ratio of the converged vacancy energy to the cohesive energy and com- pare to the values of around 0.3 that are observed for metals.
# E_coh = 6.0, and from the previous calculation with relaxation we have the converged vacancy energy as . Thus, the ratio is .

#5.6
#Do you think that relaxation of the atoms can account for the discrepancy in this ratio between pair potentials and metals discussed in class?

#Part 6
#6.1

#6.2






