import numpy as np

R = {'g': {'coeff': [ 3.61625097, -0.08821786 ],
	  'std': 0.003983374957497041},

	 'r': {'coeff': [ 2.58602565, -0.03324773],
	  'std': 0.0010612528737322644},

	 'i': {'coeff': [ 1.90961989, -0.01281381],
	  'std': 0.00049774242478213},

	 'z': {'coeff': [ 1.50170317, -0.004548],
	  'std': 0.0014337090650275107},

	 'y': {'coeff': [ 1.25373969, -0.00321323],
	  'std': 0.000615615096694815},

	 'kep': {'coeff': [ 2.68654479, -0.26852429],
	  'std': 0.002033498130929614},

	 'tess': {'coeff': [ 0.61248297, -0.03561556],
	  'std': 0.0028265468011445124}}

def line(x, c1, c2): 
    return c1 + c2*x

def R_val(band,g=None,r=None,gr=None,ext=0):
	if (g is not None) & (r is not None):
		gr = g-r

	if (gr is None) | np.isnan(gr).all():
		Rb   = R[band]['coeff'][1]
	else:
		Rr0 = R[band]['coeff'][1]
		Rg0 = R[band]['coeff'][1]

		gr_int = gr - ext*(Rg0 - Rr0)

		vals = R[band]['coeff']
		Rb  = line(gr_int,vals[0],vals[1])
	Rb_e = R[band]['std']

	return Rb, Rb_e
