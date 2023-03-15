## skewed normal distribution



```Python
import numpy as np
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, basinhopping
from scipy.stats import skewnorm

def calculate_skewed_normal_params(median, lower_err, upper_err):
    '''
    Fits a screwed normal distribution via its CDF to the [16,50,84]-percentiles
    
    Inputs:
    -------
    median : float
        the median value that was reported
    lower_err : float
        the lower errorbar that was reported
    upper_err : float
        the upper errorbar that was reported
    size : int
        the number of samples to be drawn
        
    Returns:
    --------
    alpha : float
        the skewness parameter
    loc : float
        the mean of the fitted skewed normal distribution
    scale : float
        the std of the fitted skewed normal distribution
    '''
    
    lower_err = np.abs(lower_err)
    upper_err = np.abs(upper_err)
    reference = np.array([(median-lower_err), median, (median+upper_err)])
    
    
    def fake_lnlike(p):
        alpha, loc, scale = p
        
        #::: way 1: easier to read, but slower; fake log likelihood
        # eq1 = skewnorm.ppf(0.5, alpha, loc=loc, scale=scale) - median 
        # eq2 = skewnorm.ppf(0.15865, alpha, loc=loc, scale=scale) - (median-lower_err)
        # eq3 = skewnorm.ppf(0.84135, alpha, loc=loc, scale=scale) - (median+upper_err)
        # fake_lnlike = 0.5 * np.log( eq1**2 + eq2**2 + eq3**2 ) #fake log likelihod
        
        #::: way 2: pythonic; simple chi squared
        ppf = skewnorm.ppf([0.15865, 0.5, 0.84135], alpha, loc=loc, scale=scale) #np array
        fake_lnlike = np.sum( (ppf - reference)**2 ) #simple chi squared
        
        #::: way 2: pythonic; fake log likehood, just 'cause we're feeling fancy
        # ppf = skewnorm.ppf([0.15865, 0.5, 0.84135], alpha, loc=loc, scale=scale) #np array
        # fake_lnlike = 0.5 * np.log( np.sum( (ppf - reference)**2 ) ) #fake log likelihod
        
        if np.isnan(fake_lnlike): 
            return np.inf
        else:
            return fake_lnlike


    #TODO: 
    #scipy minimize is really bad because it depends so strongly on the initial guess
    #and likes to get stuck in a local minima, which is the worst possible outcome
    #for our purpose here. We only use it because it is fast. Think about replacing
    #it with something more robust in the future though, maybe a short MCMC chain or the like.
    #The alpha_guess hack below seems to get around its weakness in finding the right alpha for now.

        
    #::: initial guess for loc and scale
    loc_guess = median
    scale_guess = np.mean([lower_err, upper_err])
    # print('\n')

    
    #::: way 1: choose alpha_guess depending on the errors and hope for the best
    # if lower_err == upper_err:
    #     alpha_guess = 0
    # elif lower_err < upper_err:
    #     alpha_guess = 1
    # elif lower_err > upper_err:
    #     alpha_guess = -1
    # initial_guess = (median, sigma_guess, alpha_guess) #sigma, omega, alpha 
    # sol = minimize(fake_lnlike, initial_guess, bounds=[(None,None), (0,None), (None,None)]) 
    # sigma, omega, alpha = sol.x
    
    
    #::: way 2: choose a few different alpha_guesses and compare (similar to basinhopping optimization)
    # initial_guess1 = None #just for printing
    sol = None
    for alpha_guess in [-10,-1,0,1,10]:
        initial_guess1 = (alpha_guess, loc_guess, scale_guess)
        sol1 = minimize(fake_lnlike, initial_guess1, bounds=[(None,None), (None,None), (0,None)]) 
        # print('sol1.fun', sol1.fun)
        if (sol is None) or (sol1.fun < sol.fun):
            # initial_guess = initial_guess1 #just for printing
            sol = sol1
            
            
    # print('best initial_guess:', initial_guess)
    # print('best solution:', sol)
    
    
    alpha, loc, scale = sol.x
    return alpha, loc, scale

def simulate_PDF(median, lower_err, upper_err, size=1, plot=True):
    '''
    Simulates a draw of posterior samples from a value and asymmetric errorbars
    by assuming the underlying distribution is a skewed normal distribution.
    
    Developed to estimate PDFs from literature exoplanet parameters that did not report their MCMC chains.
    
    Inputs:
    -------
    median : float
        the median value that was reported
    lower_err : float
        the lower errorbar that was reported
    upper_err : float
        the upper errorbar that was reported
    size : int
        the number of samples to be drawn
        
    Returns:
    --------
    samples : array of float
        the samples drawn from the simulated skrewed normal distribution
    '''
    
    alpha, loc, scale = calculate_skewed_normal_params(median, lower_err, upper_err)
    samples = skewnorm.rvs(alpha, loc=loc, scale=scale, size=size)
    return samples


```


### Mutual inclination

$\cos i_{\mathrm{bc}}=\cos i_{\mathrm{b}} \cos i_{\mathrm{c}}+\sin i_{\mathrm{b}} \sin i_{\mathrm{c}} \cos \left(\Omega_{\mathrm{b}}-\Omega_{\mathrm{c}}\right)$.   

```Python
import numpy as np
i_b = np.radians(90)
i_c = np.radians(90)
Omega_b = np.radians(0)
Omega_c = np.radians(0)
cosi_bc = np.cos(i_b)*np.cos(i_c)+np.sin(i_b)*np.sin(i_c)*np.cos(Omega_b-Omega_c)
i_bc = np.degrees(np.arccos(cosi_bc))
print(i_bc)
```
Ref: 
1. https://doi.org/10.1051/0004-6361/201935944;  
2. https://ui.adsabs.harvard.edu/abs/2022arXiv220406656A/abstract.

### Convert Gaia semi-axis of photoceter to angular semimajor axis

$a_{\mathrm{phot}}^{\prime \prime}=a^{\prime \prime}(B-\beta)$

in which $B=M_{2} /\left(M_{1}+M_{2}\right) \text { and } \beta=\ell_{2} /\left(\ell_{1}+\ell_{2}\right)$ are the secondary’s fractional mass and fractional light at the wavelength of the observation (the Gaia G band). 

Ref:
1. https://iopscience.iop.org/article/10.3847/2041-8213/abdaad/pdf
2. https://www.aanda.org/articles/aa/pdf/2019/12/aa36942-19.pdf
     
     
### Get all astronomical sites

```Python
import astropy
from astropy.coordinates import EarthLocation
all_site_list = list(set(astropy.coordinates.EarthLocation.get_site_names()))
for site_name in all_site_list:
    site_info = EarthLocation.of_site(site_name)  
    print(site_name,':', site_info.info.name,site_info.geodetic.lat.degree  , site_info.geodetic.lon.degree  )#geodetic
```    
### Thiele–Innes constants and Orbital Elements

$$
\begin{array}{l}
A=a(\cos \omega \cos \Omega-\sin \omega \sin \Omega \cos i) \\
B=a(\cos \omega \sin \Omega+\sin \omega \cos \Omega \cos i) \\
F=a(-\sin \omega \cos \Omega-\cos \omega \sin \Omega \cos i) \\
G=a(-\sin \omega \sin \Omega+\cos \omega \cos \Omega \cos i)
\end{array}$$

Convert formula:

$$
\begin{array}{l}
\omega+\Omega=\arctan \left(\frac{B-F}{A+G}\right) \\
\omega-\Omega=\arctan \left(\frac{-B-F}{A-G}\right)
\end{array}
$$


$$
\begin{aligned}
k &=\frac{A^{2}+B^{2}+F^{2}+G^{2}}{2} \\
m &=A \cdot G-B \cdot F \\
j &=\sqrt{k^{2}-m^{2}}
\end{aligned}
$$

$$
\begin{array}{l}
a=\sqrt{j+k} \\
i=\arccos \left(\frac{m}{a^{2}}\right)
\end{array}
$$

Ref:
1. https://iopscience.iop.org/article/10.3847/1538-3881/aa8d6f/pdf

### Rotation period
```Python
import astropy.constants as c
import astropy.units as u
period = 12.727272727*u.d
st_rad = 0.77*c.R_sun
vrot = 2*3.1415926535/period*st_rad
print(vrot.to('km/s'))
```

### Calculate a/Rs

```Python
import astropy.constants as c
import astropy.units as u
period = 10*u.d
st_mass = 1*c.M_sun
st_rad = 1*c.R_sun
R = (c.G*st_mass*(period)**2/4/3.1415926**2 )**(1/3)
print((R/st_rad).to(''))
```
### Calculate RM Amplitude

```Python
import astropy.units as u
import astropy.constants as const
period = 8.5*u.day
st_rad = 0.8*u.Rsun
vrot = (2*np.pi*st_rad/period).to(u.km/u.s).value
print(vrot)
pl_ror = 0.07414
vsini = vrot
b = 0.251
RM_amp = 2/3*pl_ror**2*vsini*1000*(1-b**2)**0.5
print(RM_amp,'m/s')
```
### format axis

```Python
from matplotlib.ticker import MultipleLocator, \
    FormatStrFormatter, AutoMinorLocator
def ticksetax(ax, labelsize=15, ticksize=12, tickwidth=1.5, ticklength=5):
    ax.tick_params(direction='in', which='both',  width=2,colors='k', bottom='True',top='True', left='True', right='True', labelsize=15)
    ax.spines['bottom'].set_linewidth(2)
    ax.spines['left'].set_linewidth(2)
    ax.spines['right'].set_linewidth(2)
    ax.spines['top'].set_linewidth(2)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
```

### Install Rebound on M1 mac
```
conda create -n intel_env
conda activate intel_env
conda config --env --set subdir osx-64
conda install python
```




### interpolate 1d


```Python
import matplotlib.pyplot as plt
from scipy import interpolate

def get_interpolate_data(x, y, newx):
    f = interpolate.interp1d(x, y, fill_value="extrapolate", kind='quadratic')
    return f(newx)
```

### multi plot
```Python
import matplotlib.gridspec as gridspec
plt.figure(figsize=(12, 5))
nrow = 2
ncol = 1
gs = gridspec.GridSpec(nrow, ncol, width_ratios=None, height_ratios=(2,1))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0,hspace=0)
ax = plt.subplot(gs[0])

```


### linear LQ


```Python
import numpy as np
def linear_lq(x, y, yerr):
    A = np.vander(x, 2)
    C = np.diag(yerr * yerr)
    ATA = np.dot(A.T, A / (yerr**2)[:, None])
    cov = np.linalg.inv(ATA)
    w = np.linalg.solve(ATA, np.dot(A.T, y / yerr**2))
    # print("Least-squares estimates:")
    # print("m = {0:.3f} ± {1:.3f}".format(w[0], np.sqrt(cov[0, 0])))
    # print("b = {0:.3f} ± {1:.3f}".format(w[1], np.sqrt(cov[1, 1])))
    slope = w[0]; intercept = w[1]; slope_err = np.sqrt(cov[0, 0]); intercept_err = np.sqrt(cov[1, 1])
    return slope, intercept, slope_err, intercept_err

```


### update Colab python version to 3.8

```
#install python 3.8
!sudo apt-get update -y
!sudo apt-get install python3.8

#change alternatives
!sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.7 1
!sudo update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 2
!sudo apt-get install python3-pip
!python -m pip install --upgrade pip
#check python version
!python --version
#3.8

```
### allesfitter requirement

```
pip install numpy matplotlib seaborn scipy astropy statsmodels emcee corner tqdm ellc dynesty celerite h5py rebound
```

### install ellc

```

conda install --name normal -c conda-forge ellc

```

### RV amplitude

```Python
import numpy as np
import astropy.constants as constants
import astropy.units as units
pi = 3.1415926
st_mass = 1*constants.M_sun
pl_mass = 1*constants.M_jup
period = 4*units.d
sini = np.sin(np.radians(90))
e = 0
K = (2*pi*constants.G/period/(st_mass+pl_mass)**2)**(1/3)*pl_mass*sini/(1-e**2)**0.5
K = K.to('m/s')
K
```

### get single transit file from a long-candence TESS light curve
```Python
time, flux, fluxerr = np.loadtxt('TESS.TESS.csv', unpack=True, usecols=(0,1,2), delimiter=',')
period = 8.803843
t0 = 2456883.4236
epoch_min = np.ceil((time.min() - t0)/period)
epoch_max = np.floor((time.max() - t0)/period)
epochs = np.arange(epoch_min, epoch_max + 1)
i = 0
for epoch in epochs:
    mask = (time > t0 + epoch*period - 0.1*period) & (time < t0 + epoch*period + 0.1*period)
    if np.min(flux[mask]) > 0.996:
        continue
    i = i + 1
    print('Transit',i)
    plt.figure(figsize=(10, 5))
    plt.scatter(time[mask], flux[mask])
    plt.show()
    np.savetxt('TESS'+str(i)+'.TESS.csv', np.c_[time[mask], flux[mask], fluxerr[mask]], delimiter=',', fmt='%f', header='time,flux,fluxerr')
```


### GP (Matern Kernel) detrend
```Python
plt.figure(figsize=(15, 5))
plt.scatter(time, trend_lc1, s=1, c='k', alpha=1,zorder=100)

from scipy.optimize import minimize
import celerite
from celerite import terms

def get_matern_params(x, y, yerr):
    # Set up the GP model
    kernel = terms.Matern32Term(log_sigma=1., log_rho=1.)
    gp = celerite.GP(kernel, mean=np.nanmean(y)) 
    gp.compute(x, yerr)
    print("Initial log-likelihood: {0}".format(gp.log_likelihood(y)))

    # Define a cost function
    def neg_log_like(params, y, gp):
        gp.set_parameter_vector(params)
        return -gp.log_likelihood(y)

    def grad_neg_log_like(params, y, gp):
        gp.set_parameter_vector(params)
        return -gp.grad_log_likelihood(y)[1]

    # Fit for the maximum likelihood parameters
    initial_params = gp.get_parameter_vector()
    bounds = gp.get_parameter_bounds()
    soln = minimize(neg_log_like, initial_params, jac=grad_neg_log_like,
                    method="L-BFGS-B", bounds=bounds, args=(y, gp))
    gp.set_parameter_vector(soln.x)
    print("Final log-likelihood: {0}".format(-soln.fun))

    # Make the maximum likelihood prediction
    # t = np.linspace(-5, 5, 500)
    t = x
    mu, var = gp.predict(y, t, return_var=True)
    std = np.sqrt(var)

    # Plot the data
    color = "#ff7f0e"
    # plt.errorbar(x, y, yerr=yerr, fmt=".r", capsize=0,ms=1, alpha=0.5)
    plt.plot(t, mu, color=color)
    plt.fill_between(t, mu+std, mu-std, color=color, alpha=0.3, edgecolor="none")
    plt.ylabel(r"$y$")
    plt.xlabel(r"$t$")
    # plt.xlim(-5, 5)
    plt.gca().yaxis.set_major_locator(plt.MaxNLocator(5))
    plt.title("maximum likelihood prediction");
    print(soln.x)
    log_sigma = soln.x[0]
    log_rho = soln.x[1]
    soln_dict = {'log_sigma':log_sigma, 'log_rho':log_rho}
    return soln_dict


flatten_lc1, trend_lc1 = flatten(time, flux, kernel_size=5, return_trend=True, method='gp', kernel='matern',robust=True)

x = time
y = trend_lc1
yerr = 0.0001+flatten_lc1*0

get_matern_params(x, y, yerr)



```

### get_rsuma

```Python
rr = 0.1
ar = 20
rsuma = (1+rr)/ar
```
### iSpec retrun code meanings


```
from https://github.com/segasai/astrolibpy/blob/master/mpfit/mpfit.py line 754
	 .status
		An integer status code is returned.  All values greater than zero can
		represent success (however .status == 5 may indicate failure to
		converge). It can have one of the following values:
		-16
		   A parameter or function value has become infinite or an undefined
		   number.  This is usually a consequence of numerical overflow in the
		   user's model function, which must be avoided.
		-15 to -1
		   These are error codes that either MYFUNCT or iterfunct may return to
		   terminate the fitting process.  Values from -15 to -1 are reserved
		   for the user functions and will not clash with MPFIT.
		0  Improper input parameters.
		1  Both actual and predicted relative reductions in the sum of squares
		   are at most ftol.
		2  Relative error between two consecutive iterates is at most xtol
		3  Conditions for status = 1 and status = 2 both hold.
		4  The cosine of the angle between fvec and any column of the jacobian
		   is at most gtol in absolute value.
		5  The maximum number of iterations has been reached.
		6  ftol is too small. No further reduction in the sum of squares is
		   possible.
		7  xtol is too small. No further improvement in the approximate solution
		   x is possible.
		8  gtol is too small. fvec is orthogonal to the columns of the jacobian
		   to machine precision.

```

### Google Spreadsheet filter functions

```
=OR(G:G>8, ISBLANK(G:G)=TRUE)


```

### hjd_utc_to_bjd_tdb_mid

Modified from https://github.com/WarwickAstro/time-conversions/blob/master/convert_times.py

```Python
### logz 
# print(dres.logz[-1], dres.logzerr[-1])
import logging
import numpy as np
from astropy.time import Time
from astropy.coordinates import (
    SkyCoord,
    EarthLocation
    )
import astropy.units as u
def getLightTravelTimes(ra, dec, time_to_correct):
    """
    Get the light travel times to the helio- and
    barycentres

    Parameters
    ----------
    ra : str
        The Right Ascension of the target in hourangle
        e.g. 16:00:00
    dec : str
        The Declination of the target in degrees
        e.g. +20:00:00
    time_to_correct : astropy.Time object
        The time of observation to correct. The astropy.Time
        object must have been initialised with an EarthLocation

    Returns
    -------
    ltt_bary : float
        The light travel time to the barycentre
    ltt_helio : float
        The light travel time to the heliocentre

    Raises
    ------
    None
    """
    target = SkyCoord(ra, dec, unit=(u.hourangle, u.deg), frame='icrs')
    ltt_bary = time_to_correct.light_travel_time(target)
    ltt_helio = time_to_correct.light_travel_time(target, 'heliocentric')
    return ltt_bary, ltt_helio
 
logging.basicConfig(level=logging.INFO)
def hjd_utc_to_bjd_tdb_mid(ra, dec, tinp, observatory):
    location = EarthLocation.of_site(observatory)
    logging.info('Input times in HJD, removing heliocentric correction')
    time_inp = Time(tinp, format='jd', scale='utc', location=location)
    _, ltt_helio = getLightTravelTimes(ra, dec, time_inp)
    time_inp = Time(time_inp.utc - ltt_helio, format='jd', scale='utc', location=location)
    logging.info('Output set to BJD_TDB_MID, adding barycentric correction')
    ltt_bary, _ = getLightTravelTimes(ra, dec, time_inp)
    new_time = (time_inp.tdb + ltt_bary).value
    return new_time


if __name__ == '__main__':
    # input  HJD, RA, DEC, and the location of observation
    # verified with https://astroutils.astronomy.osu.edu/time/hjd2bjd.html
    # Note that it does not support any observatories, so is only accurate to 20 ms.
    ra = '16:00:00'; dec = '+20:00:00'
    observatory = 'paranal'
    tinp = np.linspace(2450000, 2458000, 1000)

    hjd_utc_to_bjd_tdb_mid(ra, dec, tinp, observatory)




```
### Latex comments
```latex
\newcommand{\xy}[1]{{\color{red}{XY: #1}}}

```


### get individual light curve from long-cadence data

```Python
data_path = 'lc.dat'
time ,flux, flux_err = np.loadtxt(data_path, usecols=(0,1,2), unpack=True)
count = 0
times = np.array([]); fluxs = np.array([]); flux_errs = np.array([])
for i in range(len(time)):
    times = np.append(times, time[i]); fluxs = np.append(fluxs, flux[i]); flux_errs = np.append(flux_errs, flux_err[i])
    
    if i+1 != len(time):
        if time[i+1] - time[i] > 0.1:
            count += 1
            plt.figure()
            plt.scatter(times, fluxs, s=1)
            plt.title(str(count))
	    np.savetxt(str(count) + '.csv', np.c_[times, fluxs, flux_errs], delimiter=',', header='BJD, Flux, Flux_err', comments='', fmt='%.7f')
            times = np.array([]); fluxs = np.array([]); flux_errs = np.array([])
```

### creat a x86 env on M1/2 Mac

```
#terminal
softwareupdate -install-rosetta -agree-to-license 
CONDA_SUBDIR=osx-64 conda create -n rosetta python
conda activate rosetta
python -c "import platform;print(platform.machine())"
```



### creat a Pymc env on M1/2 Mac

```
conda create -n pymc
conda activate pymc
conda install -c conda-forge pymc
conda install -c conda-forge aesara
conda install -c conda-forge ipykernel
```


### get sqrt(e)cosw and sqrt(e)sinw

```Python
import numpy as np
e = 0.5
w = 0
w = np.radians(w)
sqrt_e_cosw = np.sqrt(e)*np.cos(w)
sqrt_e_sinw = np.sqrt(e)*np.sin(w)
```
### transit mask (from wotan)

```
def transit_mask(time, period, duration, T0):
    half_period = 0.5 * period
    with np.errstate(invalid='ignore'):  # ignore NaN values
        return np.abs((time - T0 + half_period) % period - half_period) < 0.5 * duration

```

### Torres Relation 

```Python
def mass_torres(hd, teff, logg, feh, dteff, dlogg, dfeh):
    """Calculate masses and radii from the calibration of Torres et al. 2010"""
    # https://github.com/MariaTsantaki/mass_radius/blob/master/mass_radius_age.py
    # coefficients
    a  = [1.5689, 1.3787, 0.4243, 1.139, -0.1425,  0.01969, 0.1010]
    da = [0.058,  0.029,  0.029,  0.240, 0.011,    0.0019,  0.014]
    b  = [2.4427, 0.6679, 0.1771, 0.705, -0.21415, 0.02306, 0.04173]
    db = [0.038,  0.016,  0.027,  0.13,  0.0075,   0.0013,  0.0082]

    X = np.log10(teff) - 4.1
    dX = dteff/teff
    log_M = a[0] + (a[1]*X) + (a[2]*(X**2)) + (a[3]*(X**3)) + (a[4]*(logg**2)) + (a[5]*(logg**3)) + (a[6]*feh)
    log_R = b[0] + (b[1]*X) + (b[2]*(X**2)) + (b[3]*(X**3)) + (b[4]*(logg**2)) + (b[5]*(logg**3)) + (b[6]*feh)
    # must check the errors
    dlog_M = np.sqrt((da[0]**2) + ((a[1]*dX)**2) + ((da[1]*X)**2) + ((da[2]*(X**2))**2) + ((a[2]*2*X*dX)**2) + ((da[3]*(X**3))**2) + ((a[3]*3*X*X*dX)**2) + ((da[4]*(logg**2))**2) + ((a[4]*2*logg*dlogg)**2) + ((da[5]*(logg**3))**2) + ((a[5]*3*logg*logg*dlogg)**2) + ((da[6]*feh)**2) + ((a[6]*dfeh)**2))
    dlog_R = np.sqrt((db[0]**2) + ((b[1]*dX)**2) + ((db[1]*X)**2) + ((db[2]*(X**2))**2) + ((b[2]*2*X*dX)**2) + ((db[3]*(X**3))**2) + ((b[3]*3*X*X*dX)**2) + ((db[4]*(logg**2))**2) + ((b[4]*2*logg*dlogg)**2) + ((db[5]*(logg**3))**2) + ((b[5]*3*logg*logg*dlogg)**2) + ((db[6]*feh)**2) + ((b[6]*dfeh)**2))
    Mt = np.power(10,log_M)
    Rt = np.power(10,log_R)
    dMt = dlog_M*Mt*np.power(10,(log_M-1.0))
    dRt = dlog_R*Rt*np.power(10,(log_R-1.0))
    # Apply Santos et al. (2013) correction
    Mcal = (0.791*(Mt**2.0)) - (0.575*Mt) + 0.701
    dMcal = np.sqrt(((0.791*Mt*dMt)**2) + ((0.575*dMt)**2))

    Mcal  = np.round(Mcal,2)
    dMcal = np.round(dMcal,2)
    Rt    = np.round(Rt, 2)
    dRt   = np.round(dRt,2)
    return hd, Rt, dRt, Mcal, dMcal

hd = 0 
T = 6000; dT = 100
logg = 4.5; dlogg = 0.1
feh = 0.0; dfeh = 0.1
mass_torres(hd, T, logg, feh, dT, dlogg, dfeh)

```

### install ellc (M1)
```Python
conda create -n alles python==3.8.5
conda install gfortran
conda install pybind11
pip install numpy matplotlib seaborn scipy astropy statsmodels emcee corner tqdm dynesty celerite h5py rebound
pip install ellc==1.8.5
```
### cat slurm file

```bash
cat "$(ls -1rt *slurm* | tail -n1)"
```


