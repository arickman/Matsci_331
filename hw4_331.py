import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d
from math import ceil
from math import floor
from tqdm import tqdm
from scipy import stats

L = 3
M = L
N = L
eps = 1
sigma = 1
rcut = 1.3 * sigma
nbasis = 4
a = sigma * 2**(2/3)
latvec = np.array([[a*L, 0, 0],[0, a*M, 0],[0,0, a*N]])
latvec *= 1.1 #Problem 4
m = 1
kb_T = 0.1 #defaults
dt = 0.01 #LJ time units/step
nsteps = 15*10**(3)
delta = 0.1 #defaults

def setup_cell():
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

	return a*atoms, natoms

def E_tot_and_force(atoms, natoms, force_flag=True):
	#if (2*rcut >= L): print("Requirement not met.")
	dummy = np.shape(atoms)[1]

	rcut_sqr = rcut*rcut
	rcut_6 = rcut_sqr*rcut_sqr*rcut_sqr
	rcut_12 = rcut_6*rcut_6
	ecut = (-1/rcut_6+1/rcut_12)
	E_tot = 0
	forces = np.zeros((natoms,3))
	for i in range(natoms - 1):
		for j in range(i+1, natoms):
			disp=atoms[i,:]- atoms[j,:]
			disp=disp- np.matmul([int(round(disp[0]/(a*L))), int(round(disp[1]/(a*M))), int(round(disp[2]/(a*N)))], latvec)
			#square of distance between atoms
			d_sqr = np.dot(disp,disp)
			#only calculate energy for atoms within cutoff distance
			#don't calculate interactions between the same atom
			if (d_sqr<rcut_sqr and d_sqr > 0):
				inv_r_6  = 1/(d_sqr*d_sqr*d_sqr)
				inv_r_12 = inv_r_6*inv_r_6
				E_tot = E_tot - inv_r_6 + inv_r_12 - ecut
				if (force_flag):
					fac = 24*(2*inv_r_12 - inv_r_6)/d_sqr
					forces[i,:] += fac*disp
					forces[j,:] -= fac*disp

	return 4*E_tot, forces

def init_vel(atoms):
	#Establish random velocities at temperature T
	temp = np.shape(atoms)[1]
	vels = np.random.uniform(-1,1, [natoms, 3])
	#make center of mass velocity zero
	vel_cm = np.sum(vels, axis = 0)/natoms
	for k in range(natoms):
		vels[k,:] -= vel_cm
	KE = 0.5*m* np.sum(np.sum(np.power(vels, 2), axis = 1))
	target_KE = 1.5*kb_T*natoms
	scale_factor = np.sqrt(target_KE/KE)
	vels *=scale_factor

	#Set previous atom positions for use with original Verlet 
	atoms_old = atoms - vels*dt
	return atoms_old, vels

def r_prop(atoms, vels, forces, dt):
	#Assume all particles of equal mass
	return atoms + dt * vels + 0.5*dt**2 * (1/m) * forces

def f_prop(propagated_atoms):
	return E_tot_and_force(propagated_atoms, natoms)[1]

def v_prop(vels, forces, propagated_forces, dt):
	return vels + dt/(2*m) * (forces + propagated_forces)

def KE_tot(vels):
	KE = 0.5*m* np.sum(np.sum(np.power(vels, 2), axis = 1))
	return KE, 2/(3*natoms) * KE #kb_T

def integrate(atoms_old, vels, dt): #atoms_old from init_vel(atoms)
	vels_matrix = np.zeros(shape=(natoms, 3, nsteps))
	vels_matrix[:,:,0] = vels 
	vels_old = vels
	forces_old = E_tot_and_force(atoms_old, natoms)[1]
	energy = KE_tot(vels_old)[0] + E_tot_and_force(atoms_old, natoms)[0]
	energy_0 = energy #this won't change
	kb_t_vec = np.zeros(nsteps)
	deviation = np.zeros(nsteps)
	kb_t_vec[0] = KE_tot(vels_old)[1]
	r_disp = np.zeros((natoms,3))
	msd = np.zeros(nsteps)
	r_0 = atoms_old
	e_vec_md = np.zeros(nsteps)
	e_vec_md[0] = energy_0
	for i in tqdm(range(1,nsteps)):
		#propagate the atoms with the old atoms
		prop_atoms = r_prop(atoms_old, vels_old, forces_old, dt)
		prop_forces = f_prop(prop_atoms)
		prop_vels = v_prop(vels_old, forces_old, prop_forces, dt)
		vels_matrix[:,:,i] = prop_vels
		#add to the energy and kb_T with the new time step
		e_step = KE_tot(prop_vels)[0] + E_tot_and_force(prop_atoms, natoms)[0]
		e_vec_md[i] = e_step
		energy += e_step
		deviation[i] = e_step - energy_0
		kb_t_vec[i] = KE_tot(prop_vels)[1]
		r_disp[:,:] = prop_atoms - r_0
		norm = np.linalg.norm(r_disp, axis = 1)
		entry = np.sum(np.square(norm), axis = 0)
		msd[i] = entry 
		#set the propagated atoms to be the old atoms to find the new propagation with
		atoms_old = prop_atoms
		vels_old = prop_vels
		forces_old = prop_forces
	return energy, kb_t_vec, 1/(3*natoms)* deviation, vels_matrix, 1/natoms * msd, e_vec_md

#Returns energy_list per accepted step, fail rate
def mc(atoms_old, nsteps):
	natoms = np.shape(atoms_old)[0]
	atoms = np.copy(atoms_old)
	energy_old = E_tot_and_force(atoms, natoms, force_flag=True)[0]
	fail = 0
	energy_list = []
	for i in range(nsteps):
		#move 1 atom
		disp = delta*(np.random.uniform(0,1, 3) - [0.5, 0.5, 0.5])
		pos = np.random.randint(natoms)
		atoms[pos] += disp
		#calculate the energy
		energy = E_tot_and_force(atoms, natoms, force_flag=True)[0]
		if energy < energy_old :
			energy_list.append(energy)
			energy_old = energy
		elif np.linalg.norm((disp/delta + 0.5))/np.sqrt(3) < (np.exp(-(energy - energy_old)/kb_T)): 
			energy_list.append(energy)
			energy_old = energy
		#reject
		else: 
			atoms[pos] -= disp
			fail += 1
	e_vec = np.array(energy_list)
	return e_vec, fail/nsteps

def modified_mc(atoms_old, nsteps, kb_T_init):
	natoms = np.shape(atoms_old)[0]
	atoms = np.copy(atoms_old)
	energy_old = E_tot_and_force(atoms, natoms, force_flag=True)[0]
	fail = 0
	energy_list = []
	kb_T_curr = kb_T_init
	r_disp = np.zeros((natoms,3))
	msd = np.zeros(nsteps)
	r_0 = np.copy(atoms_old)
	for i in tqdm(range(nsteps)):
		#move 1 atom
		disp = delta*(np.random.uniform(0,1) - 0.5)
		pos = np.random.randint(natoms * 3)
		row = floor(pos/3)
		col = pos%3 
		atoms[row, col] += disp
		#calculate the energy
		energy = E_tot_and_force(atoms, natoms, force_flag=True)[0]
		if energy < energy_old :
			energy_list.append(energy)
			energy_old = energy
			r_disp[:,:] = atoms - r_0
			norm = np.linalg.norm(r_disp, axis = 1)
			entry = np.sum(np.square(norm), axis = 0)
			msd[i] = entry 
		elif (disp/delta + 0.5) < (np.exp(-(energy - energy_old)/kb_T_curr)): 
			energy_list.append(energy)
			energy_old = energy
			r_disp[:,:] = atoms - r_0
			norm = np.linalg.norm(r_disp, axis = 1)
			entry = np.sum(np.square(norm), axis = 0)
			msd[i] = entry 
		#reject
		else: 
			atoms[row,col] -= disp
			fail += 1
		#ramp down kb_T regardless of pass or fail
		kb_T_curr -= kb_T_init/nsteps
	e_vec = np.array(energy_list)
	return e_vec, fail/nsteps, atoms, 1/natoms * msd

def e_fast(atoms, natoms):
	rcut_sqr = rcut*rcut
	rcut_6 = rcut_sqr*rcut_sqr*rcut_sqr
	rcut_12 = rcut_6*rcut_6
	ecut = (-1/rcut_6+1/rcut_12)
	E_tot = 0
	for i in range(natoms - 1):
		for j in range(i+1, natoms):
			disp = atoms[i,:]- atoms[j,:]
			disp = disp - np.matmul([int(round(disp[0]/(a*L))), int(round(disp[1]/(a*M))), int(round(disp[2]/(a*N)))], latvec)
			#square of distance between atoms
			d_sqr = np.dot(disp,disp)
			#only calculate energy for atoms within cutoff distance
			#don't calculate interactions between the same atom
			if (d_sqr<rcut_sqr and d_sqr > 0):
				inv_r_6  = 1/(d_sqr*d_sqr*d_sqr)
				inv_r_12 = inv_r_6*inv_r_6
				E_tot = E_tot - inv_r_6 + inv_r_12 - ecut
	return 4*E_tot



if __name__ == "__main__":

	#Problem 1

	atoms_init, natoms = setup_cell()
	#atoms_old, vels = init_vel(atoms_init)
	#E_tot, forces = E_tot_and_force(atoms_old, natoms)
	# T_series_eq = (integrate(atoms_old, vels, dt)[1])[500:]
	# LHS = np.var(T_series_eq)/(np.mean(T_series_eq))**2
	# c_over_k = (3/2)/(1 - (3*natoms/2)*LHS)

	#Part 1
	#Perfect harmonic crystal has c_v/k = 3.
	#print(c_over_k)
	#With 200 LJ time units (20000 steps with dt = 0.01),
	#c_v/k = 3.23 (trial 1),c_v/k = 3.29 (trial 2), compared to the true value of 3 stated above. 
	#This is already within 10% of the true value, meaning within 0.3 (c_v/k calculation lower than 3.3). 
	#Trial and error shows that we do need roughly 20000 steps (200 LJ time units) to reach this threshold.

	#Part 2
	#We are not assuming a perfect harmonic crystal (quadratic nuclear potential terms only) in that
	#we are using a LJ potential, but we are assuming quantum nuclear effects to be negligible, even
	#though we found them to be appreciate in this regime in the last problem set. The difference 
	#between our problem and the heat capacity of the perfect crystal given is that the given system
	#is assumed to be a perfect harmonic crystal, explaining the different heat capacities found. 
	#In both cases, quantum nuclear effects are ignored. 


	#Part 3
	#Now we use T = 4 (liquid) and a 3x3x3 cell (20000 time steps):
	#We now find c_v/k = 3.21 (trial 1) and 2.89 (trial 2), comparable to the solid crystal case above.


	#Problem 2

	#Part 1
	#e_vec_mc, reject_rate = mc(atoms_init, nsteps)

	#print(reject_rate)
	# x = np.arange(0,np.shape(e_vec_mc)[0])
	# fig,ax = plt.subplots()
	# ax.plot(x, e_vec_mc[x])
	# ax.set_xlabel("MC Step")
	# ax.set_ylabel("Potential Energy")
	# ax.set_title("Monte Carlo, 2x2x2, k_bT = 0.1, 10000 steps")
	# fig.savefig("hw4_2_1.pdf")

	#With this method, we see a fail rate of 0.3984 (trial 1) and 0.4112 (trial 2), and from our plot
	#note an equilibration period of roughly 250 accepted steps.

	#include equilibration for monte carlo energies:
	#e_vec_mc = e_vec_mc[250:]

	#Part 2
	# kb_T = 0.2 #just for md calculation
	# e_vec_md = (integrate(atoms_old, vels, dt)[5])[500 : np.shape(e_vec_mc)[0] + 500] 
	# x = np.arange(0,np.shape(e_vec_md)[0])
	# fig,ax = plt.subplots()
	# ax.plot(x, e_vec_md[x])
	# ax.set_xlabel("MD Step (0.01 LJ Time Units)")
	# ax.set_ylabel("Potential Energy")
	# ax.set_title("Molecular Dynamics, 2x2x2, k_bT = 0.1 equilibrated")
	# fig.savefig("hw4_2_2.pdf", bbox_inches = "tight")

	# avg_mc = np.mean(e_vec_mc)
	# avg_md = np.mean(e_vec_md)
	# var_mc = np.var(e_vec_mc)
	# var_md = np.var(e_vec_md)

	# print("Average Values: MC: {}, MD: {}.".format(avg_mc, avg_md))
	# print("Fluctuation Magnitude (Variance) : MC: {}, MD: {}.".format(var_mc, var_md))

	#From the plots, attached below, we see a clear similarity in average value, though a much larger
	#set of fluctuations (between ~ -59 - -61) for MC as opposed to MD (within ~ -60.90 - -60.92). This is 
	#confirmed by the printouts above: 
	#Average Values: MC: -60.413024334772004, MD: -60.921732557294476.
	#Fluctuation Magnitude (Variance) : MC: 0.6100566501607445, MD: 3.1666209773850308e-06.

	#Part 3
	# reject_rate = 1
	# while (np.round(reject_rate, 1) != 0.5):
	# 	delta += 0.01
	# 	reject_rate = mc(atoms_init, nsteps)[1]
	# print(' "Optimal" delta for ~0.5 acceptance rate = {}'.format(delta)) #String.format()
	# "Optimal" delta for ~0.5 acceptance rate = 0.12

	#Problem 3
	#Note: we will now use the optimal delta found above (0.12).

	#Part 1
	# k_bT = 0.1
	# delta = 0.12
	# c_v_over_k = (3/2) + np.var(e_vec_mc)/((kb_T)**2 * natoms) #use kb_T = 0.1 = const
	# print(c_v_over_k)
	#2.524 for 10000 iterations
	#To get a converged result within 10% of c/k = 3 for the perfect crystal as we did for 
	#MD, we find that we need roughly 15000 steps. The converged value was c/k =  2.75 (trial 1) and c/k = 3.13 (trial 2).

	#Part 2
	#This agrees with the value found for MD simulation above, as desired. The two are both
	#very close to 3. However, we note one difference is that MC takes only 15000 steps to 
	#converge to within 10% while MD took closer to 20000. 


	#Problem 4

	#Part 1
	#Increase delta for higher acceptance rate (right around 50%)
	# delta = 0.45
	# #print(E_tot_and_force(atoms_init, natoms, force_flag=True)[0])
	# e_vec_mc, reject_rate, atoms_final, msd = modified_mc(atoms_init, 2000, 2)
	# #print(reject_rate)
	# #print(e_vec_mc[-1])
	# print(atoms_init)
	# print(atoms_final)

	#The original atomic configuration gave energy -154.34 energy units, while the final 
	#state energy is closer to -92.33 energy units. This means the energy actually went up in this simulation,
	#which indicates that we are in an "unfavorable atomic configuration," which in this
	#case is a simulation of compression as desired. A simple printout of the initial and final
	#atomic positions reveals that while some of the positions were initallly fairly spread out,
	#(mostly greater than 1 distance unit away from zero), the final configuration including many
	# atoms within a single unit of 0, indicating compression and the beginnings of the formation
	#of a nanoparticle as discussed in the assignment handout. 

	#Part 2
	# delta = 0.45
	# e_vec_mc_2, reject_rate_2, atoms_final_2, msd2 = modified_mc(atoms_init, 10000, 4)
	# print(e_vec_mc_2[-1])
	# print(reject_rate_2)
	#I will now start from a hot (liquid state) at kb_T = 4 and 
	#schedule the annealing to be slower, decreasing to 0 over 100000 steps
	#as opposed to 2000. I will try a few runs to determine a delta for 50% 
	#acceptance as usual: delta = 0.45 again for ~ 50% acceptance. 
	#Performing this regimine yields a final energy of roughly: -78 energy units, not lower than
	# ~ -92 energy units that we found for the first annealing schedule we tried. 

	#Let's try something else: kb_T = 4, in 1000 steps
	# delta = 0.45
	# e_vec_mc_3, reject_rate_3, atoms_final_3, msd3 = modified_mc(atoms_init, 1000, 4)
	# print(e_vec_mc_3[-1])
	# print(reject_rate_3)
	#Even worse, energy ~-30

	#Another shot: kb_T = 1 to 0 in 10000 steps
	# delta = 0.45
	# e_vec_mc_4, reject_rate_4, atoms_final_4, msd4 = modified_mc(atoms_init, 10000, 1)
	# print(e_vec_mc_4[-1])
	# print(reject_rate_4)
	#We finally find a lower energy of -122.49 energy units with this schedule. 



	#Part 4
	msd = modified_mc(atoms_init, 10000, 1)[3]
	print(msd)
	t = np.arange(0,np.shape(msd)[0])
	fig,ax = plt.subplots()
	ax.plot(t, msd, label = 'Mean-Squared Displacement')
	ax.set_xlabel("MC Step")
	ax.set_ylabel("Mean-Squared Displacement")
	ax.set_title("Mean-Squared Displacement, 3x3x3, Annealing Kb_T from 1 to 0")
	fig.savefig("hw4_4_4.pdf")


	#Part 5




#faster e_calc
#not just displace in one spatial dimension










