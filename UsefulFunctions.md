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
pl_ror = 0.1
vsini = 10
b = 0
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
