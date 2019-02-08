import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d
from math import ceil
import random
from tqdm import tqdm

L = 2
M = L
N = L
eps = 1
sigma = 1
rcut = 1.3 * sigma
nbasis = 4
a = sigma * 2**(2/3)
latvec = np.array([[a*L, 0, 0],[0, a*M, 0],[0,0, a*N]])
m = 1
kb_T = 0.2
dt = 0.01
nsteps = 2*10**(3)


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
	#if (2*a >= L): print("Requirement not met.")
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

atoms, natoms = setup_cell()
atoms_old, vels = init_vel(atoms)
E_tot,forces = E_tot_and_force(atoms_old, natoms)

#Problem 2

#Part 1
def r_prop(atoms, vels, forces, dt):
	#Assume all particles of equal mass
	return atoms + dt * vels + 0.5*dt**2 * (1/m) * forces

def f_prop(propagated_atoms):
	return E_tot_and_force(propagated_atoms, natoms)[1]

def v_prop(vels, forces, propagated_forces, dt):
	return vels + dt/(2*m) * (forces + propagated_forces)

#Part 2
def KE_tot(vels):
	KE = 0.5*m* np.sum(np.sum(np.power(vels, 2), axis = 1))
	return KE, 2/(3*natoms) * KE #kb_T

#Part 3
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
	#r_disp = np.zeros((natoms,3,nsteps))
	#r_disp[0] = 0
	for i in range(1,nsteps):
		#propagate the atoms with the old atoms
		prop_atoms = r_prop(atoms_old, vels_old, forces_old, dt)
		prop_forces = f_prop(prop_atoms)
		prop_vels = v_prop(vels_old, forces_old, prop_forces, dt)
		vels_matrix[:,:,i] = prop_vels
		#add to the energy and kb_T with the new time step
		e_step = KE_tot(prop_vels)[0] + E_tot_and_force(prop_atoms, natoms)[0]
		energy += e_step
		deviation[i] = e_step - energy_0
		kb_t_vec[i] = KE_tot(prop_vels)[1]
		#r_disp[:,:,i] = np.sum(abs(prop_atoms - atoms_old)**2, )
		#set the propagated atoms to be the old atoms to find the new propagation with
		atoms_old = prop_atoms
		vels_old = prop_vels
		forces_old = prop_forces
	#r_disp_return = 1/natoms * np.sum(r_disp)
	return energy, kb_t_vec, 1/(3*natoms)* deviation, vels_matrix#, r_disp_return

#print(integrate(atoms_old, vels, dt)[0])
#We calculate a total energy of -55985.97874047868 for dt = 0.01

#Part 4 and 5
#Standalone plot for Kb_T(t):
#kb_t_vec = integrate(atoms_old, vels, dt)[1]
# t = np.arange(0,nsteps)
# fig,ax = plt.subplots()
# ax.set_ylim(0,0.25)
# ax.plot(t, kb_t_vec[t])
# ax.set_xlabel("Time Steps")
# ax.set_ylabel("kb_T")
# ax.set_title("Molecular Dynamics, 2x2x2 (dt = 0.01)")
# fig.savefig("hw3_3_1.pdf")

#One plot showing all cases of dt = 0.01, 0.02, 0.04 and 0.06. 
#Here we found that the time step of 0.06 diverges for this calculation. 
# t = np.arange(0, nsteps)
# energy_1, kb_t_vec_1, dev_1, vels_mat1, disp1  = integrate(atoms_old, vels, 0.01)
# energy_2, kb_t_vec_2, dev_2, vels_mat2, disp2  = integrate(atoms_old, vels, 0.02)
# energy_3, kb_t_vec_3, dev_3, vels_mat3,disp3  = integrate(atoms_old, vels, 0.04)
# #energy_4, kb_t_vec_4, dev_4, vels_mat4,disp4  = integrate(atoms_old, vels, 0.06) #Diverges
# fig, ax = plt.subplots()
# ax.set_ylim(-0.01,0.25)
# ax.plot(t, kb_t_vec_1[t], label = 'kb_T for dt = 0.01')
# ax.plot(t, kb_t_vec_2[t], label = 'kb_T for dt = 0.02')
# ax.plot(t, kb_t_vec_3[t], label = 'kb_T for dt = 0.04')
# #ax.plot(t, kb_t_vec_4[t], label = 'kb_T for dt = 0.06')
# ax.plot(t, dev_1[t], label = 'dev for dt = 0.01')
# ax.plot(t, dev_2[t], label = 'dev for dt = 0.02')
# ax.plot(t, dev_3[t], label = 'dev for dt = 0.04')
# # ax.plot(t, dev_4[t], label = 'dev for dt = 0.06')
# ax.legend()
# ax.set_xlabel("Time Steps")
# ax.set_ylabel("kb_T and Energy Deviations")
# ax.set_title("Molecular Dynamics Calculations")
# fig.savefig("hw3_1_5_1.pdf")

#Plot of the deviations only:
# t = np.arange(0, nsteps)
# dev_1 = integrate(atoms_old, vels, 0.01)[2]
# dev_2 = integrate(atoms_old, vels, 0.02)[2]
# dev_3 = integrate(atoms_old, vels, 0.04)[2]
# fig, ax = plt.subplots()
# ax.set_ylim(-0.01,0.01)
# ax.plot(t, dev_1[t], label = 'dev for dt = 0.01')
# ax.plot(t, dev_2[t], label = 'dev for dt = 0.02')
# ax.plot(t, dev_3[t], label = 'dev for dt = 0.04')
# ax.legend()
# ax.set_xlabel("Time Steps")
# ax.set_ylabel("Energy Deviations")
# ax.set_title("Molecular Dynamics Calculations")
# fig.savefig("hw3_1_5_2.pdf")

#Problem 3
#The only unacceptable time step is 0.06, 
#and we saw an increase and then divergence at this value, 
#so we can use the default step of 0.01 to be safe. 

#Part 1
#We have already plotted the time evolution of kb_T in a previous part. 
#Answering this roughly/subjectively, we can look at the plot from before and see that 
#around 500 time steps (5 seconds here) are required for the temperature to reach a 
#state where the average value and average variance do not significantly change with time. 
#This is performed simply by eye here. 
#From analyzing the fastest features of the temperature, we can again approximate the 
#time scale for one atomic vibration by eye as being 25 time steps (0.25 seconds). Therefore, 
# in 5 seconds there are about 20 atomic vibrations. 

#Part 2
# temp_eq = 0
# for i in range(499,nsteps):
# 	temp_eq += kb_t_vec[i]
# temp_eq *= (1/1500)
# print(temp_eq)
#kb_T_eq = 0.10346
#This is lower than the specified initial kb_t of 0.2. This makes sense since
# in our setup we constructed a lattice assuming 0 temperature and thus 
#zero specific heat (Kittel page 123). This means that the system is very 
#sensitive to thermal perturbations and thus the lattice loses thermal energy
#to the environment, dropping in magnitude by almost a factor of 2. 

#Part 3
#For a time step of 0.01, we now plot for L = M = N = 3 and compare to 
# the plot before for L = M = N = 2. 
# t = np.arange(0,nsteps)
# fig,ax = plt.subplots()
# ax.set_ylim(0,0.25)
# ax.plot(t, kb_t_vec[t])
# ax.set_xlabel("Time Steps")
# ax.set_ylabel("kb_T")
# ax.set_title("Molecular Dynamics, 3x3x3 (dt = 0.01)")
# fig.savefig("hw3_3_2.pdf")

#We see the magnitude of fluctuations decrease as we increase
#computational cell size. This makes sense according to the 
#central limit theorem--as the number of particles increases,
#the variance of this temperature random variable decreases accordingly. 


#Problem 4
#Now we want to only include post-equilibration terms. 
#In our vels_matrix below, we need to only include the terms beyond the
#500th time step. See below.


#Part 1
#See attached for work.


#Part 2
#Added code to calculate a velocity matrix at diff time steps
vels_matrix_unsliced = integrate(atoms_old, vels, dt)[3]
vels_matrix = vels_matrix_unsliced[:,:,500:]
def auto_corr(vels_matrix):
	fourier = np.fft.fft(vels_matrix, axis = 2)
	norm = abs(fourier)**2
	return 1/(3*natoms) * np.sum((np.sum(norm, axis = 1)), axis = 0)

auto = auto_corr(vels_matrix)
freq = np.fft.fftfreq(nsteps, d=dt)
fig,ax = plt.subplots()
ax.plot(freq, auto)
ax.set_xlim(0, 10)
ax.set_xlabel(r"$\omega$")
ax.set_ylabel(r"P($\omega$)")
ax.set_title("Autocorrelation Function, 2x2x2 (dt = 0.01)")
fig.savefig("hw3_4_1.pdf")

#Part 3
#Here we simply repeat the above for 1x1x1 and 3x3x3 case
# ax.set_title("Autocorrelation Function, 3x3x3 (dt = 0.01)")
# fig.savefig("hw3_4_3.pdf")

#Part 4


#Problem 5
# Unless kb_T >> h_bar*omega_max then we have quantum effects. 
#Now we compare kb_T = 0.2 to h_bar*omega_max to see if a classical approach is justified. 
#omega_max is given by

#Problem 6 

#Part 1
#Added code to calculate a displacement vector for different time steps. 
#msd = integrate(atoms_old, vels, dt)[4]













