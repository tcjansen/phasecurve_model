## Example of using forward_model.py
Create your own exoplanet phase curve models!

> NOTE: This is specifically for modeling a noise-less phase curve as observed **in the Kepler waveband**

In this example we create a thermal + symmetric reflection phase curve model with a fairly high thermal redistribution.
You can choose to create the thermal, symmetric reflection, and asymmetric reflection components of the phase curve separately, or you can create a sum of any of these components - it's up to you.

Import the module and create a phase array. In `forward_model.py`, the phase is defined to be 0 at the midpoint of the transit and +/- pi at the midpoint of the occultation. 
```
import forward_model as fm

phase = np.linspace(-np.pi, np.pi, 500)  # transit at 0
```
Specify the values which control the thermal and symmetric reflection components of the phase curve
```
Ab = 0.5  # bond albedo
eps = 10  # thermal redistribution
f = 1.0  # greenhouse factor
```

The following parameters are used for an asymmetric reflection model, which we ignore for now by setting them equal to None
```
kappa = None  # relative reflectivity
xi1 = None  # local longitude for some asymmetric cloud
xi2 = None  # same as above
```

Finally, set the system parameters in the specified units
```
Rp = 1.0  # Earth radii
Rs = 1.0  # Solar radii
Ts = 5600  # temp of star [K]
semi_a = 1.0  # planet-star separation [AU]
```

To create the phase curve model and its atmospheric components, run the following. Note that we set `a_reflection=False` to avoid modeling an asymmetric component.
```
thermal, reflection, total = fm.run(phase, Ab, eps, f, kappa, xi1, xi2, Rp, Rs, Ts, semi_a, a_reflection=False)
```

Plot for a visual inspection
```
import matplotlib.pyplot as plt

plt.plot(phase, thermal, label='thermal component')
plt.plot(phase, reflection, label='reflection component')
plt.plot(phase, total, color='black', label='total')
plt.show()
```
