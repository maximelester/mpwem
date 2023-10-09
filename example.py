# example.py

'''
This file

'''

import numpy as np
import MPWEM as mp
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm
import symbols as sy


# ---------------------
# RAW DATA
# ---------------------
'''
The raw experimental STM data (bi_mos2_stm.txt))
is simply pulled and stored into an numpy 2D array
the size of the STM image is Lx, Ly
'''


# Path to data file

datapath = 'bimos2_rawdata.txt'

# Experimental STM image in a 2D np array

z = np.loadtxt(datapath)
z-=np.mean(z[np.where(~np.isnan(z))]) # subtracting the mean value

# Size of the STM image

Lx, Ly = 134.82, 139.72 # size of the STM image (in angstrom)
N, M = z.T.shape # number of pixels along x and y directions

extent = (0, Lx, 0, Ly)
X, Y = np.meshgrid(np.linspace(0, Lx, N), np.linspace(0, Ly, M))

dx, dy = Lx/N, Ly/M # pixel dimensions (in angstrom)

# vertical size (for optimized display:
# allows to trims the vertical scale off the 5% extreme values)

vmin, vmax = mp.GetVminVmax(z, n=0.05)

# FFT size (for proper scaling of FFT images)

rextent = mp.GetReciExtent(size=(Lx, Ly), pixels=(N, M))






# ---------------------
# SUBSTRATE STM IMAGE
# ---------------------
'''
The substrate data is generated using mp.GenerateFromLattice, which
consists of a biased 2D cosine term raised to an exponent determined
by the effective radii of the atomic-resolution protrusions. 

The data is then fit (with plane waves up to k0 = 0.48 angstrom-1)
to obtain a plane-wave parameter description (kx, ky, phi, a) of the
substrate STM image which will be used to generate the moire image

'''

# MoS2 lattice parameters (and values for STM generation)

r10, r20 = 3.158, 3.158 # angstrom
omega0 = 120*np.pi/180 # radians
theta0 = -85.43*np.pi/180 # radians
k00 = 0.48 # in angstrom-1
atoms = [[0, 0], [2/3, 1/3]]
weights = [1, 0.5]
radii = [0.7, 0.7]
offset0 = [1.72,3.9] # angstrom

# STM image generation from lattice parameters

z0 = mp.GenerateFromLattice(X=X, Y=Y, r1=r10, r2=r20, omega=omega0, theta=theta0, atoms=atoms, weights=weights, radii=radii, offset=offset0)

# Fitting with 2D plane waves up to k00 (fit0 is a dictionary, fit0['popt'] contains the optimized plane wave parameters)

fit0 = mp.FitImage(z0, X=X, Y=Y, k0=k00, r1=r10, r2=r20, omega=omega0, theta=theta0, x_shift=offset0[0], y_shift=offset0[1], print_console=True)

# Normalizing plane wave amplitudes

fit0['popt'] = mp.NormalizeAmplitudes(fit0['popt'])

# Generating the updated z image from the planewave parameters (override)

z0 = np.zeros(shape=z.shape)

for p in fit0['popt']: 
	z0+=mp.cos2D(X, Y, params=p)







# ---------------------
# TOP LAYER STM IMAGE
# ---------------------
'''
The top layer data is obtained from fitting the STM data
with plane-wave parameters (kx, ky) at the precise a-Bi location
(up to |(kx, ky)| < k01 = 0.48 angstrom-1)
the plane-wave parameters are later used to generate the moire term
'''


# alpha-Bi lattice parameters (for valid fitting of STM data)

r11, r21 = 4.4649, 4.8598 # angstrom
omega1 = 90.0*np.pi/180 # radians
theta1 = -85.85*np.pi/180 # radians
offset1 = [0, 0] # angstrom
k01 = 0.48 # angstrom-1

# Fitting the data with 2D plane waves up to k01

fit1 = mp.FitImage(z, X=X, Y=Y, k0=k01, r1=r11, r2=r21, omega=omega1, theta=theta1, x_shift=offset1[0], y_shift=offset1[1], print_console=True)

# Normalizing plane wave amplitudes

fit1['popt'] = mp.NormalizeAmplitudes(fit1['popt'])

# Generating the z image from the planewave parameters

z1 = np.zeros(shape=z.shape)

for p in fit1['popt']:
	z1+=mp.cos2D(X, Y, params=p)





# ---------------------
# GENERATING THE MOIRE
# ---------------------

# MPWEM parameters

mu = 0.643 # moire coupling parameter
tau = 0.951 # top layer opacity parameter
a0 = 0.422 # scaling factor (in angstrom)
eta = -2.18 # moire damping parameter

# Moire plane wave parameters

pM = mp.GetMoireMPWEM(params0=fit0['popt'], params1=fit1['popt'], eta=eta)

# Genrerating the moire term image

zM = np.zeros(shape=z.shape)

for p in pM:
	zM+=mp.cos2D(X, Y, params=p)




# --------------------------
# GENERATING THE TOTAL IMAGE
# --------------------------



zT = a0*((1-mu)*((1-tau)*z0 + tau*z1) + mu*zM)


# ---------------------
# DISPLAYING RESULTS
# ---------------------

# apperance/customization

CMAP = 'gist_heat' # colormap of topography images
CMAP_FFT = 'magma' # colormap of modulus fft images
GAMMA = 0.5 # FFT vertical scale low intensity enhancement: (FT)**GAMMA 
bbox = {'edgecolor':[0,0,0,0], 'facecolor':[1,1,1,1]} # text style


# Figure

fig, axs = plt.subplots(ncols=5, nrows=2, tight_layout=True, figsize=(12.5,5))

# Raw data

ax = axs[0,0] 
ax.imshow(z, extent=extent, origin='lower', cmap=CMAP, vmin=vmin, vmax=vmax)
ax.text(0.04, 0.05, s=f'Raw data', bbox=bbox, fontsize=8, transform=ax.transAxes)

ax = axs[1,0]
ax.imshow(np.abs(mp.FT(z)), extent=rextent, origin='lower', cmap=CMAP_FFT, norm=PowerNorm(gamma=GAMMA))

# Substrate data (MoS2, generated)

ax = axs[0,1]
ax.imshow(z0, extent=extent, origin='lower', cmap=CMAP)
ax.text(0.04, 0.05, s=f'MoS{sy.subscript_2}', bbox=bbox, fontsize=8, transform=ax.transAxes)

ax = axs[1,1]
ax.imshow(np.abs(mp.FT(z0)), extent=rextent, origin='lower', cmap=CMAP_FFT, norm=PowerNorm(gamma=GAMMA))
mp.AddIndices(params=fit0['popt'], ax=ax, k0=k00)


# Top layer data (Bi, from fit of STM image)

ax = axs[0,2]
ax.imshow(z1, extent=extent, origin='lower', cmap=CMAP)
ax.text(0.04, 0.05, s=f'{sy.alpha}-Bi', bbox=bbox, fontsize=8, transform=ax.transAxes)

ax = axs[1,2]
ax.imshow(np.abs(mp.FT(z1)), extent=rextent, origin='lower', cmap=CMAP_FFT, norm=PowerNorm(gamma=GAMMA))
mp.AddIndices(params=fit1['popt'], ax=ax, k0=k01)

# Moire term (zM)

ax = axs[0,3]
ax.imshow(zM, extent=extent, origin='lower', cmap=CMAP)
ax.text(0.04, 0.05, s=f'Moir{sy.e_acute} term', bbox=bbox, fontsize=8, transform=ax.transAxes)

ax = axs[1,3]
ax.imshow(np.abs(mp.FT(zM)), extent=rextent, origin='lower', cmap=CMAP_FFT, norm=PowerNorm(gamma=GAMMA))
# mp.AddIndices(params=pM, ax=ax, k0=None) # uncomment to show moire indices (beware, cluttered!)

# Total image (zT)

ax = axs[0,4]
ax.imshow(zT, extent=extent, origin='lower', cmap=CMAP)
ax.text(0.04, 0.05, s=f'Total image', bbox=bbox, fontsize=8, transform=ax.transAxes)

ax = axs[1,4]
ax.imshow(np.abs(mp.FT(zT)), extent=rextent, origin='lower', cmap=CMAP_FFT, norm=PowerNorm(gamma=GAMMA))


# Customization

for ax in axs[0]:
	mp.Scalebar(ax, size=25, unit='a', color='white')

for ax in axs[1]:
	ax.set_xlim([-0.8, 0.8])
	ax.set_ylim([-0.8, 0.8])
	mp.Scalebar(ax, size=0.25, unit='a-1', color='white')
	ax.plot(0,0, '+w')


plt.show()