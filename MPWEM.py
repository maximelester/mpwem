# MPWEM.py

'''
This python file contains the different functions of the moiré plane wave
expansion model (MPWEM). 
This library allows to generate an STM image from the two STM images of
the non-interacting substrate and top layers.

'''

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import symbols as sy
import time
from scipy.optimize import curve_fit

# Crystallography related

def GetReciExtent(size, pixels):
	'''
	Returns extent (in [size unit]-1) of FFT
	'''
	Lx, Ly = size
	N, M = pixels

	dx, dy = Lx/N, Ly/M
	dkx, dky = 1/Lx, 1/Ly
	kxmax, kymax = 1/(2*dx), 1/(2*dy)

	if N%2==1:
		Kx = np.linspace(-kxmax, kxmax, N)
	else:
		Kx = np.linspace(-kxmax-dkx/2, kxmax-dkx/2, N)
	if M%2==1:
		Ky = np.linspace(-kymax, kymax, M)
	else:
		Ky = np.linspace(-kymax-dky/2, kymax-dky/2, M)

	extent = np.min(Kx), np.max(Kx), np.min(Ky), np.max(Ky)
	return extent

def GetReciVectors(r1, r2, omega, theta):
	'''
	Returns reciprocal vectors K1, K2 for any real space lattice parameters
	if omega = 120º (hexagonal lattice), K1, K2, K1-K2 are returned
	(for symmetry reasons)
	'''

	k1 = 1/r1/np.sin(omega)
	k2 = 1/r2/np.sin(omega)

	K1 = k1*(np.sin(omega+theta) - 1j*np.cos(omega+theta))
	K2 = k2*(-np.sin(theta) + 1j*np.cos(theta))

	if np.abs(omega-120*np.pi/180) < 0.001:
		return np.array([K1, K2, K1-K2])

	return np.array([K1, K2])

# STM generation from lattice parameters

def GetExp(r, k):
	'''
	Returns exponent based on radius r and wavenumber k 
	The radius is the half-width at half-maximum near a local maximum

	in 1D:
	z(x) = [0.5 (cos(2pi*k*x) + 1)]^exp
	
	z      |<---->| = r
	|--      ----
	|  \    /    \
	|   \  /      \  /
	--------------------> x
             
	'''
	if r < 1/2/k:
		exp = 1/(1-np.log2(np.cos(2*np.pi*k*r)+1))
	else:
		exp = 2
	return exp

def GenerateFromLattice(X, Y, r1, r2, omega, theta, atoms, weights, radii, offset):
	'''
	Generates a Z layer from lattice parameter definitions. Inputs:
	- r1, r2, omega, theta: lattice vectors
	 (r1, r2: angstrom; omega, theta: radians)
	- atoms: coordinates of atoms e.g. [[0, 0], [0.666, 0.333]]
	 for a typical honeycomb lattice (fractional units)
	- weights: relative weights of atoms e.g. [1, 0.5]
	for a typical TMDC lattice (dimensionless)
	- radii: radii of atoms e.g. [0.6 0.4] (in angstrom)
	- offset: [x, y] translation with respect to origin (0, 0) (in angstrom)
	
	Output: z (2D np array)

	NOTE: in the MPWEM a table of plane wave parameters is needed, therefore
	the output of this function must then be fit using the Fit2D function

	'''
	# initialization of key parameters
	pixels = X.T.shape
	R1 = r1*(1+0*1j)*np.exp(1j*theta)
	R2 = r2*(1+0*1j)*np.exp(1j*(theta+omega))
	Ks = GetReciVectors(r1, r2, omega, theta)

	# normalize weights
	# weights = np.array(weights)/np.sum(np.array(weights))

	# initialization of z
	z = np.zeros(shape=(pixels[1], pixels[0], len(atoms)))
	z_total = np.zeros(shape=(pixels[1], pixels[0]))

	# calculation: loop over atoms
	for i, a in enumerate(atoms):

		weight = weights[i]
		exp = GetExp(r=radii[i], k=np.average(np.abs(Ks)))

		T = a[0]*R1 + a[1]*R2 + offset[0] + offset[1]*1j # shift in complex coords (angstrom)
		
		for K in Ks: # loop over the self.K
			z[:,:,i]+=np.cos(2*np.pi*(K.real*(X-T.real)+K.imag*(Y-T.imag)))
		
		z[:,:,i] = Normalize(z[:,:,i])
		z[:,:,i] = weight*z[:,:,i]**exp

	# sum of each atom-wave corresponding to every atoms in unit cell
	z_total = np.sum(z, axis=2)

	# normalize 
	z_total = Normalize(z_total)

	return z_total

# Fitting procedure

def FitImage(data, X, Y, k0, r1, r2, omega, theta, x_shift, y_shift, print_console=False):
	'''
	Returns a PW param array np.array([[m, n, kx, ky, phi, a], ...])
	such that each K = m*K0 + n*K1 is fit for |K|<k0 cut-off parameter
	'''

	K = GetReciVectors(r1=r1, r2=r2, omega=omega, theta=theta)
	extent = np.min(X[0]), np.max(X[0]), np.min(Y[:,0]), np.max(Y[:,0])
	size = extent[1]-extent[0], extent[3]-extent[2]
	rextent = GetReciExtent(size=size, pixels=X.T.shape)

	# choice of maximum value of m,n (or p,q) indices

	MAX_ORDER = 4

	# wavevector search domain

	dk = 0.002 # 'freedom' to change wavevector kx and ky by dk during the fitting

	# generation of plane-wave parameters for initial guess
	# loop over the possible K vectors under k0
	# creates a PW-like array: [m, n, kx, ky, phi, a]

	params = []
	LowerBounds, HigherBounds = [], []
	indices = []
	indices_beyond_k0 = []

	for m in range(0, MAX_ORDER+1):
		for n in range(-MAX_ORDER, MAX_ORDER+1):

			k = m*K[0] + n*K[1] # pq coordinates
			
			# considering the phase shift (initial guess is not ideal -
			# this is why we perform a fit since it depends on lateral offset (easy to handle)
			# but also on the detail of the atomic positions within the unit cell (difficult)

			phi = -2*np.pi*(k.real*x_shift+k.imag*y_shift)
			phi = FoldPlusMinusPi(phi)

			if np.abs(k)>0.00001 and indices.count([-m, -n])==0: # twins are undesirable in fitting and (0, 0) added separately after
				
				if np.abs(k)<k0:
					
					# storing indices that are being considered

					indices.append([m, n])
					
					# intializing amplitudes 

					a = 0.1/np.abs(k)**2 # Somewhat arbitrary initial amplitude. Could think of a better approach later.
					
					# appending the list of PW parameters

					params.append([m, n, k.real, k.imag, phi, a])

					# parameters bounds

					LowerBounds.append([k.real-dk, k.imag-dk, -np.inf, 0])
					HigherBounds.append([k.real+dk, k.imag+dk, np.inf, 10])
				
				else:
					
					# we keep track of the [m, n] indices ignored due to > k0
					# since we want to avoid missing [m, n] (we expect at least 1 or 2 |K|>k0)
					# because MAX_ORDER is 4, it should be fine - we will very rarely need higher order than that.
					# but leaving it as a warning in case indices_beyond_k0 is empty.

					indices_beyond_k0.append([m, n])  


	# adding a constant (plane wave such that k = [0,0])

	params.append([0, 0, 0, 0, 0, np.mean(data)]) # mean(data) is definitely an ideal start value
	LowerBounds.append([-0.000001, -0.000001, -0.001, np.min(data)]) # lower bounds
	HigherBounds.append([0.000001, 0.000001, 0.001, np.max(data)]) # higher bounds

	# format change

	params = np.array(params)
	
	# normalize amplitudes (again: not sure if absolutely necessary - depends on the nature of the image too)

	params[:,-1]/=np.sum(params[:,-1])

	# print initial parameters
	if print_console == True:
		print(f'Fitting of {len(params)} plane-waves...')
		print(f'Initial parameters:')
		PrintParams(params)
	# warning

	if len(indices_beyond_k0)<1:
		print(f'Warning: there may be points of the reciprocal lattice that were not considered. In case, increase MAX_ORDER')

	# actual fit

	if print_console==True:
		print(f'Fitting of {len(params)} plane-waves in progress...', end='\r')
	
	Dict_Fit = FitCos2D(data=data, X=X, Y=Y, InitParams=params, bounds=(LowerBounds, HigherBounds))
	
	if print_console == True:
		print(f'Fitting of {len(params)} plane-waves done (time: {Dict_Fit["time"]:.2f} s; RMS = {Dict_Fit["rms"]:.3f})\n')
		PrintParams(Dict_Fit['popt'])

	return Dict_Fit

def FitCos2D(data, X, Y, InitParams, bounds):
	'''
	Returns a convenient dictionary containing fitting outputs
	InitParams: initial guess parameters (should be reasonably close for successful fit)
	bounds: (lower, higher)
	'''
	t0 = time.time()

	# we store initial indices [m, n] in their own array for later
	# and keep InitParams to its minimum information

	Indices = np.zeros(shape=(len(InitParams), 2))
	if len(InitParams[0]) == 6:
		Indices = InitParams[:,:2]
		InitParams = InitParams[:, 2:]
	if len(InitParams[0]) == 8:
		Indices = InitParams[:,:4]
		InitParams = InitParams[:, 4:]

	# Flatten the initial guess parameter list
	p0 = [p for prms in InitParams for p in prms]

	# Flatten the bounds
	lb, hb = bounds
	lb = np.array(lb).ravel()
	hb = np.array(hb).ravel()

	# Get extent
	x0, x1, y0, y1 = np.min(X[0]), np.max(X[0]), np.min(Y[:,0]), np.max(Y[:,0])
	extent = x0, x1, y0, y1

	# Get N, M
	N, M = X.T.shape

	# Get FFT
	fft = np.fft.fftshift(np.fft.fft2(data))

	# Get rextent
	rextent = GetReciExtent(size=(x1-x0, y1-y0), pixels=(N, M))

	# Flatten X, Y data
	xdata = np.vstack((X.ravel(), Y.ravel()))

	# Fit 	
	popt, pcov = curve_fit(_cos2D, xdata, data.ravel(), p0, bounds=(lb, hb))

	# Uncertainties
	dpopt = np.sqrt(np.diag(pcov)).reshape(int(len(popt.ravel())/4), 4)

	# Create fitted data
	popt = popt.reshape(len(InitParams),4)

	# fold phi onto the [-pi, pi] domain
	for args in popt:
		args[2]=FoldPlusMinusPi(args[2])

	# Reshape dpopt
	dpopt = dpopt.reshape(popt.shape)

	# add indices to a new popt so that popt has a nice PW-like shape
	popt_new = np.zeros(shape=(len(popt), 6))
	popt_new[:,0:2] = Indices
	popt_new[:,2:] = popt
	popt = popt_new

	# generate zfit for convenience
	zfit = 0*X
	for i in range(len(InitParams)):
		zfit+=cos2D(X, Y, InitParams[i])

	# calculate additional quantities
	fft_fit = np.fft.fftshift(np.fft.fft2(zfit))
	error = data-zfit
	rms = np.sqrt(np.mean(error**2))
	rms_norm = rms/np.sqrt(np.mean(data**2))

	# pack results into a dictionary
	# the main result is 'popt' (optimized parameters in the shape of a PW 'params' [m, n, kx, ky, phi, a])
	# note: the 'popt' array does NOT contain twin plane-wave indices
	# additionally 'dpopt' contains the uncertainty on each [kx, ky, phi, a] parameter
	# 'pcov' is the covariance matrix
	# 'rms' the root mean square and 'rms_norm' the normalized rms
	# 'time' the total time of execution of the whole FitCos2D function
	# the initial ('data') and final ('data_fit') 2D arrays are also stored, aswell with respective FFTs ('fft' and 'fft_fit')
	# 'extent' and 'rextent' are also kept for convenience
	# 'init' is a copy of the InitParams
	# 'error'

	dic = {
	'data': data,
	'init': InitParams,
	'fft': fft,
	'data_fit': zfit,
	'fft_fit': fft_fit,
	'extent': extent,
	'rextent': rextent,
	'popt': popt,
	'dpopt': dpopt,
	'pcov': pcov,
	'error': error,
	'rms': rms,
	'rms_norm': rms_norm,
	'time': time.time()-t0, # in seconds
	}

	return dic

def cos2D(X, Y, params):
	'''
	Generator: a*cos(2pi*(kx*X + ky*Y) + phi)
	params can be with or without index information
	'''

	if len(params) == 4:
		kx, ky, phi, a = params
	if len(params) == 6:
		m, n, kx, ky, phi, a = params
	elif len(params) == 8:
		m, m, p, q, kx, ky, phi, a = params

	return a*np.cos(2*np.pi*(kx*X + ky*Y) + phi)

def _cos2D(M, *args):
	'''
	Generator like above
	This time with arbitrary number of plane waves
	*args must be flattened
	'''
	X, Y = M
	Z = np.zeros(X.shape)
	for i in range(len(args)//4):
		Z += cos2D(X, Y, args[i*4:i*4+4])
	return Z

# Moire term functions

def GetMoireMPWEM(params0, params1, eta):
	'''
	Returns the planewave parameters of zM
	given the planewave parameters of substrate (params0)
	and planewave parameters of the top layer (params1)
	'''
	params = []

	params0 = MakeTwinParams(params0)
	params1 = MakeTwinParams(params1)

	for p0 in params0:
		
		m, n, k0x, k0y, phi0, a0 = p0
		
		if (m,n)!=(0,0):
			
			for p1 in params1:
				
				p, q, k1x, k1y, phi1, a1 = p1
				
				if (p,q)!=(0,0):
					
					Mx = k1x-k0x # x coordinates of M_mnpq
					My = k1y-k0y # y coordinates of M_mnpq
					phiM = phi1-phi0 # phase shift of M_mnpq
					
					M = np.sqrt(Mx**2 + My**2)
					aM = M**eta # amplitude following power law
					
					params.append([m, n, p, q, Mx, My, phiM, aM])

	params = np.array(params)
	params = NormalizeAmplitudes(params)

	return params

def MakeTwinParams(params):
	'''
	Returns params with added twins if they do not already exist
	for a given parameter: [m, n, kx, ky, phi, a] without its twin
	the two twin parameters are:
	[m, n, kx, ky, phi, a/2]
	[-m, -n, -kx, -ky, -phi, a/2] (amplitudes must be shared)
	'''
	l = []

	if len(params[0]) == 8:
		return params

	indices = ListIndices(params)

	for p in params:
		
		m, n, kx, ky, phi, a = p

		# if [-m, -n] already exist, simply copy existing PW parameters (also accounts for [0, 0])

		if indices.count(([-m, -n]))>0:
			l.append([m, n, kx, ky, phi, a])

		# otherwise [-m, -n] not existing, generate the twin PW parameter

		else:
			l.append([m, n, kx, ky, phi, a/2])
			l.append([-m, -n, -kx, -ky, -phi, a/2])

	return np.array(l)

def ListIndices(params):
	'''
	returns a list of indices
	'''
	l = []
	for p in params:
		l.append([int(p[0]), int(p[1])])

	return l

# Miscellaneous math functions

def Normalize(z):
	'''
	returns a normalized version of Z where:
	min(Z) = 0, max(Z) = 1
	'''
	z-=np.min(z)
	z/=np.max(z)
	return z

def FT(z, remove_mean=True):
	if remove_mean==True:
		z0 = np.mean(z)
	else:
		z0 = 0
	return np.fft.fftshift(np.fft.fft2(z-z0))

def FoldPlusMinusPi(theta):
	'''
	Folds angles to the [-pi, ... pi] domain
	'''
	return 2*np.arctan(np.tan(theta/2))

def NormalizeAmplitudes(params):
	'''
	Normalizes all amplitudes in the planewave parameters
	such that sum(a_i) = 1
	'''
	p = params.copy()
	p[:,-1] = p[:,-1]/np.sum(p[:,-1])

	return p

# Display related

def GetVminVmax(data, n):
	'''
	get the vertical bounds of the data (trimmed by n)
	'''
	M, N = data.shape

	data_ = data.copy()

	data_[np.isnan(data)]=0

	a = np.histogram(data_.ravel(), bins=max(N,M), density=True)
	x = a[1][:-1]
	y = a[0]
	xtrim = x[y>n]
	ytrim = y[y>n]
	if len(xtrim)*len(ytrim)==0:
		return None, None
	xmin = np.min(xtrim)
	xmax = np.max(xtrim)
	return xmin, xmax

def Scalebar(ax, size=None, width=None, color='black', unit='a', fontsize=8, loc='upper right', background=False, alphabg=0.8):
	'''
	adds scalebar of size, and width and color on ax
	'''

	# hide x and y axes

	ax.get_xaxis().set_visible(False)
	ax.get_yaxis().set_visible(False)

	# size/width if unspecified

	if size is None:
		size_x = ax.get_xlim()[1]-ax.get_xlim()[0]
		size = size_x/5
		if size>1:
			size = int(size)
		else:
			size = round(size,1)
			# size = int(size)
	
	if width is None:
		size_y = ax.get_ylim()[1]-ax.get_ylim()[0]
		width = size_y/140

	# font

	fontprops = fm.FontProperties(size=fontsize, weight='heavy')

	# unit

	if unit == 'a':
		text = f'{size} {sy.angstrom}'
	elif unit == 'a-1':
		text = f'{size} {sy.angstrom}{sy.superscript_minus1}'

	bar = AnchoredSizeBar(ax.transData, size, text, loc, pad=0.5, sep=5,
		color=color, frameon=False, size_vertical=width, fontproperties=fontprops)
	bar.set(clip_on=True)

	if background==True:

		color_background = np.array([1, 1, 1, 0])-np.array(mpl.colors.to_rgba(color))
		color_background[3] = alphabg
		ax.text(0.78, 0.94, s='              ', bbox={'edgecolor':[0,0,0,0], 'facecolor': color_background}, transform=ax.transAxes, fontsize=10, clip_on=True)
	
	ax.add_artist(bar)

def PrintParams(p):
	'''
	Better print for list of plane wave parameters
	For clarity purpose
	'''

	MIN_K = 0.001

	name_param = f'{p=}'.split('=')[0]

	if len(p) == 0:
		print(f'Plane-wave parameters contains no plane-wave parameters (empty)')

	elif len(p[0]) == 4:
		print(f'i\t|\tk{sy.subscript_x}\tk{sy.subscript_y}\t{sy.phi}\ta\t|\t|k|\t{sy.lambdaa}\t{sy.delta} ({sy.degree})')
		print(f'-'*75)
		for i in range(len(p)):
			K = np.sqrt(p[i][0]**2+p[i][1]**2)
			D = np.angle(p[i][0]+p[i][1]*1j)*180/np.pi
			if K>MIN_K:
				L = 1/K
			else:
				L = np.inf
				D = np.nan
			print(f'{i}\t|\t{p[i][0]:.3f}\t{p[i][1]:.3f}\t{p[i][2]:.3f}\t{p[i][3]:.3f}\t|\t{K:.2f}\t{L:.2f}\t{D:.1f}')

	elif len(p[0]) == 6:
		print(f'i\t|\tm\tn\t|\tk{sy.subscript_x}\tk{sy.subscript_y}\t{sy.phi}\ta\t|\t|k|\t{sy.lambdaa}\t{sy.delta} ({sy.degree})')
		print(f'-'*102)
		for i in range(len(p)):
			K = np.sqrt(p[i][2]**2+p[i][3]**2)
			D = np.angle(p[i][2]+p[i][3]*1j)*180/np.pi
			if K>MIN_K:
				L = 1/K
			else:
				L = np.inf
				D = np.nan
			print(f'{i}\t|\t{int(p[i][0])}\t{int(p[i][1])}\t|\t{p[i][2]:.3f}\t{p[i][3]:.3f}\t{p[i][4]:.3f}\t{p[i][5]:.3f}\t|\t{K:.2f}\t{L:.2f}\t{D:.1f}')

	elif len(p[0]) == 8:
		print(f'i\t|\tm\tn\tp\tq\t|\tk{sy.subscript_x}\tk{sy.subscript_y}\t{sy.phi}\ta\t|\t|k|\t{sy.lambdaa}\t{sy.delta} ({sy.degree})')
		print(f'-'*119)
		for i in range(len(p)):
			K = np.sqrt(p[i][4]**2+p[i][5]**2)
			D = np.angle(p[i][4]+p[i][5]*1j)*180/np.pi
			if K>MIN_K:
				L = 1/K
			else:
				L = np.inf
				D = np.nan
			print(f'{i}\t|\t{int(p[i][0])}\t{int(p[i][1])}\t{int(p[i][2])}\t{int(p[i][3])}\t|\t{p[i][4]:.3f}\t{p[i][5]:.3f}\t{p[i][6]:.3f}\t{p[i][7]:.3f}\t|\t{K:.2f}\t{L:.2f}\t{D:.1f}')

	print()

def AddIndices(params, ax, k0):

	params = MakeTwinParams(params)

	for p in params:
		indices = p[:-4]
		kx, ky = p[-4:-2]
		phi = p[-2]
		a = p[-1]
		ax.text(kx, ky, s=f' {NiceIndex(indices)}', fontsize=8, color='white', clip_on=True)

	if k0 is not None:
		c = plt.Circle((0,0), k0, facecolor=[0,0,0,0], edgecolor=[1,1,1,0.6])
		ax.add_artist(c)
		ax.text(0.98, 0.02, s=f'k{sy.subscript_0} = {k0}', transform=ax.transAxes, ha='right')

def NiceIndex(indices):
	'''
	'''
	s = ''
	for i in indices:
		v = int(i)
		if i>=0:
			s+=str(v)
		else:
			s+=rf'$\bar{-v}$'
	return s