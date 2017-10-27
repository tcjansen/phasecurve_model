# A lot of the expressions and terminology come 
# from Hu et al. 2015 and Cowan & Agol 2011, whom I 
# would like to thank.
#
# Written by Tiffany Jansen
# Columbia University Astronomy
# Last updated April 2017
# 
# For any questions about the code,
# please contact Tiffany at jansent@astro.columbia.edu
# 
# For any questions about the science, please consult the 
# following papers:
# 
# Jansen, T. & Kipping, D. submitted to MNRAS
# Hu, R., Demory, B.-O., Seager, S., Lewis, N., & Showman, A. P.2015, ApJ, 802, 51
# Cowan, N. B., & Agol, E. 2011, ApJ, 726, 82 


import os, sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate
from scipy.integrate import odeint
import astropy.constants as const
from scipy.interpolate import interp1d

wls, K_response = np.genfromtxt(os.getcwd() + '/kepler_hires.dat', usecols=(0,1), unpack=True)
wls *= 1e-6
h = const.h.value
c = const.c.value
hc = h*c
k_B = const.k_B.value
R_earth = const.R_earth.value
R_sun = const.R_sun.value
au = const.au.value
numer = 2 * const.h.value * const.c.value**2

# Interpolate the planck function for the planet
int_planck_f = os.getcwd() + '/integrated_planck_table.dat'
T_eff_lib, int_planck = np.genfromtxt(int_planck_f, usecols=(0,1), unpack=True)
interpolate_planck = interp1d(T_eff_lib, int_planck)


### SYMMETRIC ###

def sym_reflection(Rp, semi_a, phase, Ab):
	"""Symmetric reflection component of the phase curve.

	Args:
		Rp (float): radius of the planet in m
		a (float): semi major axis of the planet in m
		phase (array): phase angles of the planet in radians
		Ag (float): geometric albedo of the planet

	Returns:
		array, normalized by host flux
	"""
	Ag = 2/3 * Ab # in the lambertian sphere approximation
	refl_component = (Rp / semi_a) ** 2 * Ag
	phase = abs(phase)
	phase_component = (np.sin(phase) + (np.pi - phase) * np.cos(phase)) / np.pi
	return refl_component * phase_component

### ANTI-SYMMETRIC ###

def q_Phi_arrays(alpha, xi1, xi2):
	""" Returns the phase integral and function for multiple phases.

	Args:
		alpha (float): phase angle of the planet in radians
		xi1 (float): local longitude in radians designating 
			the start of the region with low reflectivity
		xi2 (float): local longitude in radians designating 
			the end of the region with low reflectivity

	Returns: 
		array
	"""
	q = phase_integral(xi1, xi2)[0]  # excluding integral errors
	Phi = phase_function(alpha, xi1, xi2)
	return q, Phi

def plot_q_xi1xi2():
	""" Plots the phase integral as a function of xi1 and xi2
	"""
	alpha = np.linspace(-np.pi, np.pi, 50)
	xi1 = np.arange(-5 * np.pi / 10, 5 * np.pi / 10, np.pi / 10)
	for i in range(len(xi1)):
		q = []
		xi2 = np.linspace(-np.pi/2, np.pi/2, 50)
		for j in range(len(xi2)):
			q += [phase_integral(xi1[i], xi2[j])]
		plt.plot(xi2, q, color='black', lw=2)
	plt.xlabel(r'$\xi_{2}$')
	plt.ylabel('q')
	plt.ylim([0, 1.5])
	plt.xlim([-np.pi/2, np.pi/2])
	plt.show()
	return

def reflectivity(Ab, kappa, q):
	"""The reflectivity parameters determined by the Bond Albedo and the 
	phase integral for the patchy atmosphere scenerio. 

	Args:
		Ab (float): bond albedo 
		kappa (float): reflectivity boosting factor. < 1 for dark patch, 
			>1 for bright patch
		q (float): value of the phase integral at one angle

	Returns:
		tuple of floats: the reflectivity parameters r0 and r1, which
			determine the reflectivity of the patchy regions
	"""
	r0 = Ab / (1 + (2 / 3) * q * kappa)
	r1 = kappa * r0
	return r0, r1

def phase_function(alpha, xi1, xi2):
	"""Determines a value of the phase curve for a given phase angle
	considering the patchy regions in the patchy cloud scenerio.
	Equations A.3 - A.8 in Hu et al. 2015.

	Note:
		-pi/2 < xi1 <= xi2 < pi/2
		-pi <= alpha <= pi

	Args:
		alpha (float): phase angle of the planet in radians
		xi1 (float): local longitude in radians designating 
			the start of the region with low reflectivity
		xi2 (float): local longitude in radians designating 
			the end of the region with low reflectivity

	Returns:
		float
	"""
	a = -alpha

	if (-np.pi <= a) & (a <= 0):

		if (-np.pi / 2 <= a + np.pi / 2) & (a + np.pi / 2 <= xi1):
			return 1 / np.pi * (np.cos(a) * (np.pi + a) + np.sin(np.pi + a))

		elif (xi1 <= a + np.pi / 2) & (a + np.pi / 2 <= xi2):
			return 1 / np.pi * (np.cos(a) * (np.pi / 2 + xi1) \
	 		       + 0.5 * (np.sin(np.pi + a) + np.sin(2 * xi1 - a)))

		elif (xi2 <= a + np.pi / 2) & (a + np.pi / 2 <= np.pi / 2):
			return 1 / np.pi * (np.cos(a) * (np.pi + a + xi1 - xi2) \
	  		       + np.sin(np.pi + a) + np.cos(xi1 + xi2 - a) * np.sin(xi1 - xi2))

	elif (0 <= a) & (a <= np.pi):

		if (-np.pi / 2 <= a - np.pi / 2) & (a - np.pi / 2 <= xi1):
			return 1 / np.pi * (np.cos(a) * (np.pi - a + xi1 - xi2) \
	  		  	   + np.sin(a) + np.cos(xi1 + xi2 - a) * np.sin(xi1 - xi2))

		elif (xi1 <= a - np.pi / 2) & (a - np.pi / 2 <= xi2):
			return 1 / np.pi * (np.cos(a) * (np.pi / 2 - xi2) \
	 		 	   + 0.5 * (np.sin(a) - np.sin(2 * xi2 - a)))

		elif (xi2 <= a - np.pi / 2) & (a - np.pi / 2 <= np.pi / 2):
			return 1 / np.pi * (np.cos(a) * (np.pi - a) + np.sin(a))

	else:
		raise ValueError("D'oh! None of the possible conditionals passed. Check" \
						+ " that alpha and xi1,xi2 are within the right ranges.")

def phase_integral(xi1, xi2):
	"""Integrates the phase function with respect to phase angle to account for 
	the additional asymmetric reflection in the patchy cloud scenerio.

	Args:
		alpha (float): phase angle of the planet in radians
		xi1 (float): local longitude in radians designating 
			the start of the region with low reflectivity
		xi2 (float): local longitude in radians designating 
			the end of the region with low reflectivity

	Returns:
		float
	"""
	integrand = lambda alpha: phase_function(alpha, xi1, xi2) * np.sin(abs(alpha))
	result = integrate.quad(integrand, -np.pi, np.pi)
	return result

def antisym_reflection(Ag, kappa, phase, xi1, xi2, Rp, semi_a):
	"""Antisymmetric reflection component of the phase curve.

	Args: 

	Returns:
		array, normalized by host flux
	"""
	q = phase_integral(xi1, xi2)[0]
	Ab = q * Ag
	r0, r1 = reflectivity(Ab, kappa, q)
	sym_refl = sym_reflection(Rp, semi_a, phase, Ag) * 2 * r0 / 3

	Phi = []
	for alpha in phase:
		Phi += [phase_function(alpha, xi1, xi2)]

	patchy_refl = (Rp / semi_a) ** 2 * 2 * r1 / 3 * np.array(Phi)
	return sym_refl + patchy_refl

### THERMAL ###

def dP_dXi(P, xi, eps):
	"""Expression for the derivative of the thermal phase function P with respect
	to the local longitude xi. 

	Note:
		This can't be solved analytically, which is why P is solved by 
		scipy's odeint in thermal_phase_func().

	Args:
		P (array): thermal phase function
		xi (array/float): local longitude in radians
		eps (float): thermal redistribution factor

	Returns: 
		array
	"""
	return 1 / eps * (0.5 * (np.cos(xi) + abs(np.cos(xi))) - P**4)

def thermal_phase_func(eps, phase):

	xi = np.linspace(-np.pi/2, 3*np.pi/2, len(phase))

	if eps == 0.0:
		P = np.zeros(len(xi))
		nonzero_mask = (xi > -np.pi / 2) & (xi < np.pi / 2)
		P[nonzero_mask] = 0.5 * (np.cos(xi[nonzero_mask])**(1/4) + \
			abs(np.cos(xi[nonzero_mask]))**(1/4))

		return P

	g = (3 * np.pi/eps) ** 4
	P_dawn = (np.pi + g**(1/3))**(-1/4) # initial condition for P
	P = odeint(dP_dXi, P_dawn, xi, args=(eps,))
	P = np.array(P).ravel()
	
	return P

def analytic_P(phase, eps):
	"""Analytic thermal phase function from Cowan & Agol 2011"""
	xi = np.linspace(-np.pi/2, 3*np.pi/2, len(phase))

	T0 = np.pi**(-1/4)
	gamma = 4 * T0**3 / eps

	day = (xi > -np.pi / 2) & (xi < np.pi / 2)
	night = (xi > np.pi / 2) & (xi < 3 * np.pi / 2)
 
	Pday = 3 / 4 * T0 + (gamma * np.cos(xi[day]) \
		 + np.sin(xi[day]))/(eps * (1 + gamma**2)) \
		 + np.exp(-gamma * xi[day]) / \
		 (2 * eps * (1 + gamma ** 2) * np.sinh(np.pi * gamma / 2))

	Pnight = 3 / 4 * T0 + np.exp(-gamma * (xi[night] - np.pi)) \
		   / (2 * eps * (1 + gamma ** 2) * np.sinh(np.pi * gamma / 2))

	Pdawn = [(np.pi + (3 * np.pi / eps)**(4/3))**(-1/4)]

	y0 = 0.69073
	y1 = 7.5534
	Pdusk = [(np.pi**2 * (1 + y0/eps)**(-8) + y1 * eps**(-8/7))**(-1/8)]

	return Pdawn + Pday.tolist() + Pdusk + Pnight.tolist() + Pdawn

def P_phi(P, alpha, res=12):
	"""Gives the value of the thermal phase as a function of phi.

	Note:
		phi is defined by xi = phi - alpha

	Args:
		P (array): thermal phase function as a function of xi
		phi (float): local longitude transformed to the observer's frame
		alpha (float): phase angle
		res (int): long/lat resolution. splits planetary surface up into 
			[180 / res]-square degree grids. default = 15 degree^2 grids

	Returns:
		array
	"""
	xi = np.linspace(-np.pi/2, 3*np.pi/2, len(P))
	xi_start = -np.pi/2 - alpha
	xi_end = np.pi/2 - alpha

	if alpha > 0:
		condition1 = xi >= xi_start + 2 * np.pi
		condition2 = xi <= xi_end
		P_p = np.append(P[condition1], P[condition2])

	elif alpha <= 0:
		P_p = P[(xi >= xi_start) & (xi <= xi_end)]

	# interpolate to grab the value of P every 10 degrees of longitude
	phi_idx = np.arange(0, res*40 + 1, 40)
	phi_P = np.linspace(-np.pi/2, np.pi/2, len(P_p))
	phi_interp = np.linspace(-np.pi/2, np.pi/2, res*40 + 1)
	P_interp = np.interp(phi_interp, phi_P, P_p)

	return P_interp[phi_idx]

def planck_function(T_eff):
	"""Planck function convolved with the Kepler bandpass.
	
	Note:
		Must have kepler_hires.dat in your working directory.

	Args:
		T_eff (array with shape (lons, lats, 1)): effective temperature in Kelvins

	Returns:
		float: radiance (W m-2 sr-1) as detected by Kepler, i.e. value of the 
			Planck function integrated over the Kepler bandpass.
	"""
	B = numer / (wls**5)

	e = np.exp(hc / (wls * k_B * T_eff))

	integrand = B / (e - 1) * K_response
	Bk = np.trapz(integrand, wls)

	return Bk

def find_nearest_planck(T_eff):
	""" Finds the nearest value to 'value' in the array."""
	idx = np.abs(T_eff_lib-T_eff).argmin(axis=2)
	return int_planck[idx]

def T_eff(f, P, alpha, Ts, Rs, semi_a, Ab, res=12):
	""" Effective temperature of the planet as a function of 
	alpha, planetary longitude, and planetary latitude. 
	i.e., the temperature distribution across the planet's surface.

	Args:
		res (int): long/lat resolution. splits planetary surface up into 
			[180 / res]-square degree grids. default = 15 degree^2 grids

	Returns:

	"""
	P_eps_alpha = P_phi(P, alpha, res)

	theta = np.linspace(np.pi / 2, -np.pi / 2, res + 1)[:, None]
	T0_theta = Ts * np.sqrt(Rs / semi_a) * (1 - Ab)**(1/4) * np.cos(theta)**(1/4)

	return f * T0_theta * P_eps_alpha

def thermal_integrand(theta, phi, alpha, P, Ab, eps, f, Ts, Rs, semi_a, res=12):
	"""Integrand of the thermal phase dependency of the planet's
	thermal emission.

	Note:
		This function's theta is NOT the mcmc parameter tuple. See Args.

	Args:
		theta (array): planetary latitude in radians
		phi (array): planetary longitude from observer's POV
		alpha (float): phase angle of the planet in radians
		eps (float): thermal redistribution factor
		f (float): greenhouse factor
		Ts (float): effective temperature of the star
		Rs (float): radius of the star
		semi_a (float): semi major axis of the planet
		res (int): long/lat resolution. splits planetary surface up into 
			[180 / res]-square degree grids. default = 15 degree^2 grids

	Returns:
		array in units normalized by the host star
	"""
	#temperature distribution function
	T = T_eff(f, P, alpha, Ts, Rs, semi_a, Ab, res)
	# planck function for the planet integrated over Kepler bandpass
	Bk = interpolate_planck(T)

	return Bk * np.cos(theta)**2 * np.cos(phi)

def thermal(phase, Ab, eps, f, Rp, Rs, Ts, semi_a, res=12):
	"""Thermal component of the phase curve.

	Args:
		phase (array): phase angles of the planet [radians]
		Ab (float): bond albedo
		eps (float): thermal redistribution factor
		f (float): greenhouse factor
		Rp (float): radius of the planet [m]
		Rs (float): radius of the star [m]
		Ts (float): effective temperature of the star [K]
		semi_a (float): semi major axis of the planet [m]
		res (int): long/lat resolution. Must satisfy 180 mod res = 0.
			Splits planetary surface up into 
			[180 / res]-square degree grids. default = 15 degree^2 grids

	Returns:
		array, normalized by host flux
	"""
	P = thermal_phase_func(abs(eps), phase)

	# len() = 19 for a surface resolution of 10 x 10 degrees
	theta = np.linspace(np.pi / 2, -np.pi / 2, res + 1)[:, None]
	phi = np.linspace(-np.pi / 2, np.pi / 2, res + 1)

	F_T_norm = []
	for alpha in phase:

		Bk = thermal_integrand(theta, phi, alpha, P, Ab, abs(eps), f, Ts, Rs, semi_a, res)
		
		Bs = planck_function(Ts) # integrated planck function of the host

		inner = np.trapz(Bk, phi) # integrate over longitude
		F_T = np.trapz(inner, theta.ravel()[::-1]) # integrate over latitude

		F_T_norm += [F_T / (Bs * np.pi * Rs**2)]

	if eps < 0.0:
		return Rp**2 * np.array(F_T_norm[::-1])

	return Rp**2 * np.array(F_T_norm)

### BRING IT AROUND TOWN ### 

def therm_sref(n_samples, kepid, phase, Ab, eps, f, Rp, Rs, Ts, semi_a):
	therm = thermal(phase, Ab, eps, f, R_earth * Rp, R_sun * Rs, Ts, au * semi_a) * 1e6
	s_ref = sym_reflection(Rp, semi_a, phase, Ab) * 1e6
	return therm + s_ref

def run(phase, Ab, eps, f, kappa, xi1, xi2, Rp, Rs, Ts, semi_a, \
	res=12, therm=True, s_reflection=True, a_reflection=True):
	""" Returns the total phase curve model and any components of the model.
	i.e., if therm = True, s_reflection = True, a_reflection = False, this function
	will return the thermal component, symmetric component, and the total thermal + 
	symmetric model. 

	Notes:
		Returns the full model on default.
		If a_reflection==False, set kappa, xi1, xi2 to None.

	Args:
		phase (array): phase angles in radians
		eps (float): thermal redistribution factor
		f (float): greenhouse factor
		kappa (float): reflectivity boosting factor. < 1 for dark patch, 
			>1 for bright patch. Set to None if a_reflection==False
		xi1 (float): local longitude in radians designating 
			the start of the region with low reflectivity. Set to None 
			if a_reflection==False
		xi2 (float): local longitude in radians designating 
			the end of the region with low reflectivity. Set to None 
			if a_reflection==False
		Rp (float): radius of the planet [Earth radii]
		Rs (float): radius of the star [Solar radii]
		Ts (float): effective temperature of the star [K]
		semi_a (float): semi major axis of the planet [AU]
		res (int): long/lat resolution. splits planetary surface up into 
			[180 / res]-square degree grids. default = 15 degree^2 grids
		therm (bool): Returns the thermal component of the model if True
		s_reflection (bool): Returns the symmetric reflection component of 
			the model if True
		a_reflection (bool): Returns the asymmetric reflection component of 
			the model if True

	Returns:
		array or tuple of arrays
	"""
	Rp = R_earth * Rp
	Rs = R_sun * Rs
	semi_a = au * semi_a

	if [therm, s_reflection, a_reflection] == [True, False, False]:
		therm = thermal(phase, Ab, eps, f, Rp, Rs, Ts, semi_a, res=res) * 1e6
		return therm

	elif [therm, s_reflection, a_reflection] == [False, True, False]:
		s_ref = sym_reflection(Rp, semi_a, phase, Ab) * 1e6
		return s_ref

	elif [therm, s_reflection, a_reflection] == [False, False, True]:
		a_ref = antisym_reflection(Ag, kappa, phase, xi1, xi2, Rp, semi_a) * 1e6
		return a_ref

	elif [therm, s_reflection, a_reflection] == [True, True, False]:
		therm = thermal(phase, Ab, eps, f, Rp, Rs, Ts, semi_a, res=res) * 1e6
		s_ref = sym_reflection(Rp, semi_a, phase, Ab) * 1e6
		return therm, s_ref, therm + s_ref

	elif [therm, s_reflection, a_reflection] == [True, False, True]:
		therm = thermal(phase, Ab, eps, f, Rp, Rs, Ts, semi_a, res=res) * 1e6
		a_ref = antisym_reflection(Ag, kappa, phase, xi1, xi2, Rp, semi_a) * 1e6
		return therm, a_ref, therm + a_ref

	elif [therm, s_reflection, a_reflection] == [False, True, True]:
		s_ref = sym_reflection(Rp, semi_a, phase, Ab) * 1e6
		a_ref = antisym_reflection(Ag, kappa, phase, xi1, xi2, Rp, semi_a) * 1e6
		return s_ref, a_ref, s_ref + a_ref

	elif [therm, s_reflection, a_reflection] == [True, True, True]:
		therm = thermal(phase, Ab, eps, f, Rp, Rs, Ts, semi_a, res=res) * 1e6
		s_ref = sym_reflection(Rp, semi_a, phase, Ab) * 1e6
		a_ref = antisym_reflection(Ag, kappa, phase, xi1, xi2, Rp, semi_a) * 1e6
		return therm, s_ref, a_ref, therm + s_ref + a_ref

	else:
		print("You must set at least one component of the model to True")
		f.close()
		return

if __name__ == "__main__":
	assert phase_integral(np.pi/4, np.pi/4)[0] == 3/2, "Houston, we've had a problem..."




