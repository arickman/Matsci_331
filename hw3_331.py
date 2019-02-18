import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d
from math import ceil
import random
from tqdm import tqdm
from scipy import stats

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
kb_T = 4
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
	r_disp = np.zeros((natoms,3))
	msd = np.zeros(nsteps)
	r_0 = atoms_old
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
		r_disp[:,:] = prop_atoms - r_0
		norm = np.linalg.norm(r_disp, axis = 1)
		entry = np.sum(np.square(norm), axis = 0)
		msd[i] = entry 
		#set the propagated atoms to be the old atoms to find the new propagation with
		atoms_old = prop_atoms
		vels_old = prop_vels
		forces_old = prop_forces
	return energy, kb_t_vec, 1/(3*natoms)* deviation, vels_matrix, 1/natoms * msd

#print(integrate(atoms_old, vels, dt)[0])
#We calculate a total energy of -55985.97874047868 for dt = 0.01

#Part 4 and 5
#Standalone plot for Kb_T(t):
# kb_t_vec = integrate(atoms_old, vels, dt)[1]
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

# auto = auto_corr(vels_matrix)
# freq = np.fft.fftfreq(1500, d=dt)
# fig,ax = plt.subplots()
# ax.plot(freq, auto)
# ax.set_xlim(0, 10)
# ax.set_xlabel(r"$\omega$")
# ax.set_ylabel(r"P($\omega$)")
# ax.set_title("Autocorrelation Function, 2x2x2 (dt = 0.01)")
# fig.savefig("hw3_4_1.pdf")

#In the plot attached, we do see consistency with the sketch in that
#we expect a resonant frequency (a spike) to correspond to 
#available k-modes which depend on the periodic boundary condition (
#since the appreciable k-modes are those which lie within the first
#Brillouin zone). 

#Part 3
#Here we simply repeat the above for 1x1x1 and 3x3x3 case. 
#Starting with the 3x3x3 case, we see roughly the same number of 
#time steps (500) needed to reach equilibrium from the plot of
#kb_T for the 3x3x3 case performed above. Thus we can repeat
#the above (with L manually changed at the top to 3). 
# ax.set_title("Autocorrelation Function, 3x3x3 (dt = 0.01)")
# fig.savefig("hw3_4_2.pdf")

#Now for the 1x1x1 case:
# ax.set_title("Autocorrelation Function, 1x1x1 (dt = 0.01)")
# fig.savefig("hw3_4_3.pdf")

#From the plots generated for all 3 cases, it is clear that generally,
#the larger the computational cell size, the more resonant peaks
#we see. This makes sense as a higher cell size implies more allowed
#k-modes (limited by the periodic boundary conditions of the lattice) 
#and thus more resonant frequency values stand out in the plot. However, 
#being that the k-modes are limited by the periodic bcs, even though we 
#can say that 4x4x4 will yield more peaks than 3x3x3 which yields more than 
#2x2x2, we can not make the same argument going from 1x1x1 to 2x2x2. 
#It turns out that 1x1x1 yields the most peaks, which makes sense as we
#do not have any periodic boundary conditions in the single lattice point case, 
#and thus the momentum can take on many more values, resulting in many 
#more resonant frequencies shown. 

#Part 4
#The amplitude of the peaks should be proportional to the density of states,
#since many available states at the same frequency would imply large constructive
#interference. There will not necessarily be resonance peaks out of equilibrium
#because the system may be too noisy for any dominant behavior to 
#show through. In other words the system is changing rapidly and thus the 
#periodicity and thus the constructive interference building the peaks
#is not yet achieved before we hit equilibrium. 

#Problem 5

# Unless kb_T >> h_bar*omega_max then we have quantum effects. 
#Now we compare kb_T = 0.2 to h_bar*omega_max to see if a classical approach is justified. 
#omega_max is given by the largest omega corresponding to an appreciable (not noise)
#peak on the autocorrelation curve. This is estimated by eye, and for 
#a computational cell size of 2x2x2, we see that roughly omega_max = 4, 
#estimating on the high side which is okay since we are only asking
#whether kb_T >>  h_bar*omega_max. So to perform this calculation, 
#we lastly need the units of omega here:
#We are operating in units of time m^(1/2)*sigma * epsilon^(-1/2), therefore
#omega (inverse time) is in units of m^(-1/2)*sigma^-1 * epsilon^(1/2).
#See attached for the rest of this calculation. 

#Problem 6 

#Part 1
#Added code to calculate a displacement vector for different time steps. 
t = np.arange(0,nsteps)
msd = integrate(atoms_old, vels, dt)[4]
fig,ax = plt.subplots()
ax.plot(t, msd, label = 'Mean-Squared Displacement')
ax.set_xlabel("Time Step (dt = 0.01)")
ax.set_ylabel("Mean-Squared Displacement")
# ax.set_title("Mean-Squared Displacement, 3x3x3, kb_T = 0.2")
# fig.savefig("hw3_6_1.pdf")
ax.set_title("Mean-Squared Displacement, 3x3x3, kb_T = 4")
# Generated linear fit
slope, intercept, r_value, p_value, std_err = stats.linregress(t,msd)
line = slope*t + intercept
ax.plot(t, line, label = 'Linear Fit, slope = ' + str(slope))
ax.legend()
fig.savefig("hw3_6_2.pdf")

#The lower temperature case is very stable. We see steady fluctuations about
#an average displacement. The higher temperature case 
#however climbs up steadily with time in a linear fashion. As time goes on 
#our msd increases from the phenomenon of diffusion that we discussed in class. 

#Part 2
#The diffusion constant can be approximated by the long time 
#(t -> infinity) derivative of the mean-squared displacement, 
#divided by 2*d where d is the dimension of the system (3 here). 
#To estimate this, we take the derivative of the final portion of the 
#high temperature plot (approximated by the slope above),
# ~(.174) sigma^2/time unit. 
#This is approximately (.174)* sigma^2/([6.63 * 10^(-26)kg]^(1/2) * [sigma] * [epsilon]^(-1/2))
#Using wolfram alpha to calculate this with the given parameters for 
#argon (and remembering to divide by 6), we find a diffusion constant
#of approximately 1.5 * 10^-3  mm^2/s. Comparing this to the value for water,
# 1.6 * 10^-3 mm^2/s we see that particles will diffuse/information 
#will propagate about as fast in water as in Argon at the same temperature. 

#Part 3
#In a liquid vs a solid, there is higher thermal energy and more 
#available k-modes. Thus, we expect that at a temperature of 4 (liquid),
#we will see more resonant peaks. 

# fig,ax = plt.subplots()
# ax.plot(freq, auto)
# ax.set_xlim(0, 10)
# ax.set_xlabel(r"$\omega$")
# ax.set_ylabel(r"P($\omega$)")
# ax.set_title("Autocorrelation Function, 2x2x2 (dt = 0.01), Kb_T = 4")
# fig.savefig("hw3_6_3.pdf")

#See attached for the plot--turns out this is exactly what we see,
#many more resonant frequency peaks. 









