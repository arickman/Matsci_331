import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits import mplot3d
from math import ceil
import random
from tqdm import tqdm

L = 3
M = L
N = L
eps = 1
sigma = 1
rcut = 1.3 * sigma
nbasis = 4
a = sigma * 2**(2/3)
latvec = np.array([[a*L, 0, 0],[0, a*M, 0],[0,0, a*N]])
kb_T = 4
dt = 3

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
	#Requires 2*rcut < LMN
	if (2*rcut >= L): print("Requirement not met.")
	atoms, natoms = setup_cell()
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
					forces[i,:] = forces[i,:] + fac*disp
					forces[j,:] = forces[j,:] - fac*disp

	return 4*E_tot, forces

def init_vel(atoms, kb_T, dt):
	#Establish random velocities at temperature T
	temp = np.shape(atoms)[1]
	vels = np.random.uniform(-1,1, [natoms, 3])
	print(vels)
	#make center of mass velocity zero
	vel_cm = np.sum(vels, axis = 0)/natoms
	for k in range(natoms)
		vels[k,:] -= vel_cm
	KE = np.sum(np.sum(np.power(vels, 2), axis = 1))
	target_KE = 1.5*kb_T*natoms
	scale_factor = np.sqrt(target_KE/KE)
	vels *=scale_factor

	#Set previous atom positions for use with original Verlet 
	atoms_old = atoms - vels*dt
	return atoms_old, vels

atoms, natoms = setup_cell()
E_tot,forces = E_tot_and_force(atoms, natoms)
atoms_old, vels = init_vel(atoms, kb_T, dt)









