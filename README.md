# phasecurve_model
Create your own explanet phase curve models!

You can choose to create the thermal component, symmetric reflection, and asymmetric reflection separately or you can create a sum of any of these components - it's up to you.



Example of creating a thermal + symmetric reflection phase curve with a fairly high thermal redistribution:

import forward_model as fm:

#specify the parameters you wish to use

phase = np.linspace(-np.pi, np.pi, 500)

Ab = 0.5 #bond albedo

eps = 10 #thermal redistribution

f = 1.0 #greenhouse factor

kappa = None #relative reflectivity - only for asymmetric reflection

xi1 = None #local longitude for some asymmetric cloud

xi2 = None #same as above

Rp = 1.0 #1 earth-radii

Rs = 1.0 #1 solar-radii

Ts = 5600 #temp of star in kelvins

semi_a = 1.0 #separation in au


therm, asym, combination = fm.run(phase, Ab, eps, f, kappa, xi1, xi2, Rp, Rs, Ts, semi_a, a_reflection=False)
