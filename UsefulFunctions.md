### Common Linux command

```
find . -name "*.idl" -type f -delete
find . -type f -name '*.txt' -exec rm {} \;
find . -name "*.eps" -type f -exec bash -c 'epstopdf "$0" "${0%.eps}.pdf"' {} \;
find . -name "*.ps" -type f -exec bash -c 'ps2pdf "$0" "${0%.ps}.pdf"' {} \;

```

### get TIC

```Python
from astroquery.simbad import Simbad

def get_tic_ids(star_names):
    # add the 'ids' field to the SIMBAD query object to retrieve identifiers
    Simbad.add_votable_fields('ids')

    # dictionary to hold results
    results = {}

    for name in star_names:
        # query SIMBAD for the star
        result_table = Simbad.query_object(name)
        
        # find TIC ID among identifiers, if it exists
        ids = result_table['IDS'][0].split('|')
        tic_ids = [i for i in ids if 'TIC' in i]
        
        if tic_ids:
            # if there is a TIC ID, add it to the results
            results[name] = tic_ids[0]
        else:
            # otherwise, note that no TIC ID was found
            results[name] = 'No TIC ID found'

    return results

# your list of star names
star_names = ["AU Mic", "Beta PIC", "CoRoT-2", "CoRoT-3", "CoRoT-11", "CoRoT-18"]

tic_ids = get_tic_ids(star_names)
print(tic_ids)

```


### get psi
```Python
import numpy as np
st_inc = simulate_PDF(89.4,24.5,24.5,1000)
pl_inc = simulate_PDF(87.297,0.553,0.443,1000)
lambda_ = simulate_PDF(4,11,10,1000)
cos_psi = np.cos(np.deg2rad(st_inc))*np.cos(np.deg2rad(pl_inc)) + np.sin(np.deg2rad(st_inc))*np.sin(np.deg2rad(pl_inc))*np.cos(np.deg2rad(lambda_))
psi = np.rad2deg(np.arccos(cos_psi))


q50, q16, q84 = np.percentile(psi, [50, 16, 84])
lerr, uerr= q50-q16, q84-q50

# .2f
print('psi = %.2f +%.2f -%.2f'%(q50,lerr,uerr))
# latex format
print('psi = $%.2f^{+%.2f}_{-%.2f}$'%(q50,lerr,uerr))

```



### Calc Stellar Velocity V
```Python
import astropy.units as u   
import astropy.constants as c
import uncertainties as unc

period = unc.ufloat(9.7666105,9.766610/10)

st_rad = unc.ufloat(1.47,0.017)
vrot = 2*np.pi*st_rad/period

# vrot.to(u.km/u.s)

n = vrot.n; s = vrot.s

(n*c.R_sun/u.day).to(u.km/u.s), (s*c.R_sun/u.day).to(u.km/u.s)


```


### get stellar inc

```python
# https://www.pnas.org/doi/pdf/10.1073/pnas.2017418118 A backward-spinning star with two coplanar planets
import pymc3 as pm
import numpy as np
import theano.tensor as tt


sol_radius_km = 696340  
day_to_sec = 24 * 60 * 60 
R_obs, Rerr_obs, Prot_obs, Prot_err, vsini_obs, vsini_err = 1.47, 0.017, 9.77, 0.98, 7.6, 1.3


R_obs, Rerr_obs = R_obs * sol_radius_km, Rerr_obs * sol_radius_km
Prot_obs, Prot_err = Prot_obs * day_to_sec, Prot_err * day_to_sec


with pm.Model() as model:
    R = pm.Normal('R', mu=R_obs, sd=Rerr_obs) 
    Prot = pm.Normal('Prot', mu=Prot_obs, sd=Prot_err)
    inc_rad = pm.Uniform('inc_rad', lower=0, upper=np.pi)  # inc in radians
    cos_i = pm.Deterministic('cos_i', tt.cos(inc_rad))
    
    u = pm.Deterministic('u', tt.sqrt(1 - cos_i**2))
    sin_i_squared = pm.Deterministic('sin_i_squared', 1 - cos_i**2)
    sin_i = pm.Deterministic('sin_i',tt.sqrt(sin_i_squared))
    
    inc = pm.Deterministic('inc', inc_rad * (180 / np.pi))  # convert radians to degrees

    v = 2*np.pi*R/Prot
    vu = v * u

    pm.Normal('likelihood_R', mu=R, sd=Rerr_obs, observed=R_obs)
    pm.Normal('likelihood_Prot', mu=Prot, sd=Prot_err, observed=Prot_obs)
    pm.Normal('likelihood_vsini', mu=vu, sd=vsini_err, observed=vsini_obs)
    trace = pm.sample(2000, tune=2000, chains=2, cores=2, target_accept=0.999)


import arviz as az
az.style.use('arviz-darkgrid')
az.plot_trace(trace, var_names=['R', 'Prot', 'cos_i', 'sin_i_squared', 'sin_i', 'inc'])

pm.summary(trace, var_names=['R', 'Prot', 'cos_i', 'sin_i_squared', 'sin_i', 'inc'])




```




### download TESS lc (fast)


```Python
# %%

import os
import time
import requests
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'DeJavu Serif'
plt.rcParams['font.serif'] = ['Times New Roman']
import os
import pandas as pd
import lightkurve as lk
import numpy as np
import os


# download the obliquity data from the website
target_url = 'https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv'
response = requests.get(target_url)
data = response.text
name = 'toi.csv'
date = time.strftime("%Y-%m-%d", time.localtime())
dated_name = 'toi_'+date+'.csv'
# check if the file exists
if os.path.exists(dated_name):
    print(dated_name+' exists')
else:
    print('downloading '+dated_name)
    with open(dated_name, 'w') as f:
        print(data, file=f)

def get_period(toi,dated_name):
    toi_table = pd.read_csv(dated_name, comment='#')
    tois = toi_table['TOI'].values
    periods = toi_table['Period (days)'].values
    tic_ids = toi_table['TIC ID'].values
    toi = float(toi)
    period = periods[tois==toi][0]
    tic_id = tic_ids[tois==toi][0]
    return period,tic_id

def bin_lightcurve_fast(times, fluxes, flux_errs, time_bin_size):
    df = pd.DataFrame({'time': times, 'flux': fluxes, 'flux_err': flux_errs})

    # Define root-mean-square function for flux_err
    def rmse(x):
        return np.sqrt(np.sum(np.square(x))) / len(x)

    bin_labels = (df['time'] / time_bin_size).astype(int)
    binned_df = df.groupby(bin_labels).agg({
        'time': 'mean',
        'flux': 'mean',
        'flux_err': rmse  # Use root-mean-square error for flux_err
    })
    print(bin_labels)
    return binned_df['time'].values, binned_df['flux'].values, binned_df['flux_err'].values





# %%
good_wjs = '''121.01  173.01  201.01  292.01  296.01  316.01  327.01  334.01  345.01   352.01  365.01  450.01  481.01  527.01  558.01  573.01  671.01  677.01   679.01  681.01  746.01  758.01  760.01  768.01  811.01  812.01  830.01   850.01  892.01  899.01  902.01  917.01  919.01  924.01  933.01  943.01   948.01  963.01  978.01 1058.01 1119.01 1176.01 1186.01 1232.01 1274.01  1312.01 1366.01 1406.01 1433.01 1456.01 1478.01 1642.01 1670.01 1708.01  1755.01 1764.01 1825.01 1849.01 1859.01 1874.01 1875.01 1879.01 1890.01  1897.01 1898.01 1899.01 1938.01 1958.01 1963.01 1982.01 1985.01 2005.01  2032.01 2033.01 2137.01 2145.01 2147.01 2159.01 2179.01 2202.01 2251.01  2255.01 2271.01'''
good_wjs = good_wjs.split(' ')
# remove the empty string
good_wjs = [good_wj for good_wj in good_wjs if good_wj != '']
good_wjs = [float(good_wj) for good_wj in good_wjs]
good_wjs = np.array(good_wjs)
good_wjs



def get_lc(toi_id):
    print('working on ',toi_id)
    toi_id = str(toi_id)
    

    if os.path.exists(toi_id+'.txt'):
        times, fluxes, fluxes_err = np.loadtxt(toi_id+'.txt',unpack=True,delimiter=',')
    else:
        period, tic_id = get_period(toi_id,dated_name)
        lcs = lk.search_lightcurve('TIC '+str(tic_id), mission='TESS')
        print('TIC '+str(tic_id))
        sector_uniq = np.unique(lcs.table['mission'], return_index=True)[0].tolist()

        try:
            sector_uniq.remove('TESS Sector ')
        except:
            pass

        author_list = ['SPOC', 'TESS-SPOC', 'QLP']
        times = np.asarray([]); fluxes = np.asarray([]); fluxes_err = np.asarray([])
        for sector in sector_uniq:
            lcs_tmp = lcs[lcs.table['mission'] == sector]
            for author in author_list:
                lc = lcs_tmp[lcs_tmp.author == author]
                print(sector, author, len(lc))
                try:
                    lc = lc.download().normalize().bin(10/60/24)
                except:
                    continue
                time = lc.time.value; flux = lc.flux.value; flux_err = lc.flux_err.value
                times = np.append(times, time); fluxes = np.append(fluxes, flux); fluxes_err = np.append(fluxes_err, flux_err)
                if len(lc) > 0:
                    break
        no_nan = np.isfinite(times) & np.isfinite(fluxes) & np.isfinite(fluxes_err)
        times = times[no_nan]; fluxes = fluxes[no_nan]; fluxes_err = fluxes_err[no_nan]
        np.savetxt(toi_id+'.txt', np.transpose([times, fluxes, fluxes_err]), delimiter=',', header='time,flux,flux_err', fmt='%.7f')
    time, flux,flux_err = np.loadtxt(toi_id+'.txt',delimiter=',',unpack=True)
    time, flux, flux_err = bin_lightcurve_fast(time, flux, flux_err, 10/60/24)
    
    

import multiprocessing as mp


if __name__ == "__main__":
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map_async(get_lc, good_wjs)
        results = results.get() # to fetch the results of computation


```



### fast bin pandas
```Python
import pandas as pd

def bin_lightcurve_fast(times, fluxes, flux_errs, time_bin_size):
    df = pd.DataFrame({'time': times, 'flux': fluxes, 'flux_err': flux_errs})
    bin_labels = (df['time'] / time_bin_size).astype(int)
    binned_df = df.groupby(bin_labels).mean()
    return binned_df['time'].values, binned_df['flux'].values, binned_df['flux_err'].values
```

### get ecc from rho_obs and rho_circ

```Python
import pymc3 as pm
import theano.tensor as tt
import numpy as np
import matplotlib.pyplot as plt
import astropy.units as u
import exoplanet as xo
from exoplanet_core.pymc import ops
import arviz as az
import pymc3_ext as pmx
import aesara_theano_fallback.tensor as tt
import pymc3 as pm
from celerite2.theano import terms, GaussianProcess
import pandas as pd
from fastprogress import fastprogress
fastprogress.printing = lambda: True
import corner



rho_star_observations = np.random.normal(0.435, 0.067, 1000)
rho_cir_observations = np.random.normal(0.059, 0.01, 1000)

with pm.Model() as model:

    ecs = pmx.UnitDisk("ecs", shape=(2, 1), testval=0.01 * np.ones((2, 1)))
    ecc = pm.Deterministic("ecc", tt.sum(ecs**2, axis=0))
    w = pm.Deterministic("omega", tt.arctan2(ecs[1], ecs[0]))
    xo.eccentricity.kipping13(
        "ecc_prior",  fixed=True, observed=ecc
    )   
    sinw = tt.sin(w)
    g = (1 + ecc*sinw) / tt.sqrt(1 - ecc**2)
    rho_star_model = g**(-3) * rho_cir_observations
    obs = pm.Normal('obs', mu=rho_star_model, sd=0.1, observed=rho_star_observations)
    trace = pm.sample(1000, tune=1000, cores=2, chains=2, target_accept=0.99)
e = trace['ecc']
w = trace['omega']
sqrt_e_sinw = np.sqrt(e) * np.sin(w)
sqrt_e_cosw = np.sqrt(e) * np.cos(w)


data = np.hstack([sqrt_e_cosw, sqrt_e_sinw])
figure = corner.corner(
    data,
    labels=[
        r"$\sqrt{e} \cos \omega$",
        r"$\sqrt{e} \sin \omega$",
    ],
    quantiles=[0.16, 0.5, 0.84],
    show_titles=True,
    title_kwargs={"fontsize": 12},
)

q50, q16, q84 = np.percentile(trace['ecc'], [50, 16, 84])
uerr, lerr = q84-q50, q50-q16
q50, q16, q84
print('ecc = {0:.3f} +{1:.3f} -{2:.3f}'.format(q50, uerr, lerr))



```


### agreement

```Python

import numpy as np

def calculate_agreement(value1, value_lerr1, value_uerr1, value2, value_lerr2, value_uerr2):
    if value1 > value2:
        result = (value1 - value2) / np.hypot(value_lerr1, value_uerr2)
    else:
        result = (value2 - value1) / np.hypot(value_uerr1, value_lerr2)
    agreement = round(result, 2)
    return agreement


calculate_agreement(1,0.01,0.01,1,0.01,0.01)
```






### Gaia fucntion


```Python

def dr2dr3(dr2_source_id):
    query = "SELECT * FROM gaiadr3.dr2_neighbourhood WHERE dr2_source_id = "+str(dr2_source_id)
    job = Gaia.launch_job_async(query)
    results = job.get_results()
    return results['dr3_source_id'].value[0]


def get_gaia_dr3_plx(dr3_source_id):
    query = "SELECT * FROM gaiadr3.gaia_source WHERE source_id = "+str(dr3_source_id)
    job = Gaia.launch_job_async(query)
    results = job.get_results()
    return results['parallax'].value[0], results['parallax_error'].value[0]
dr2dr3(381592313648387200),get_gaia_dr3_plx(381592313648387200)

```



### standardize tepcat name (pstable)
```Python

obliquity = pd.read_csv('obliquity_2023-06-09.csv', skiprows=[1,2])
names = obliquity['System']
names = list(set(list(names)))
names = [x.replace('_',' ') for x in names]
standardnames = []
for name in names:
    isSpace = False
    numstrs = name.split('-')
    if len(numstrs)<2:
        numstrs = name.split(' ')
        isSpace = True
    haszerostr= ''
    for i, numstr in enumerate(numstrs):
        while numstr[0] == '0':
            numstr = numstr[1:]
            haszero = i
            haszerostr = numstr
            numstrs[i] = haszerostr
    # print(numstrs, haszerostr)
    if isSpace:
        sep = ' '
    else:
        sep = '-'
    newname = sep.join(numstrs)
    
    if name[-1].isalpha():
        newname = newname.strip(name[-1])
    standardnames.append(newname.strip())
# find and replcae the names
ob_names = ['KELT-19','Kepler-89','Kepler-448','WASP-30','WASP-134','WASP-180','WASP-109','XO-2','Kepler-420','K2-93','WASP-111','K2-267','WASP-85','pi Men','Kepler-13','K2-234','TOI-1937','WASP-94','DS Tu']
ps_name = ['KELT-19 A','KOI-94','KOI-12','WASP-30','WASP-134','WASP-180 A','WASP-109','XO-2 N','KOI-1257','HIP 41378','WASP-111','EPIC 246851721','WASP-85 A','HD 39091','KOI-13','HD 89345','TOI-1937 A','WASP-94 A','DS Tuc A']

for i in range(len(ob_names)):
    try:
        standardnames[standardnames.index(ob_names[i])] = ps_name[i]
    except:
        print(ob_names[i], 'not found')
```



### measure stellar rotation period 
#### Wavelet
```Python
pip install git+https://github.com/zclaytor/prot
!git clone https://github.com/zclaytor/prot
!cp -r /content/prot/prot/misc  /usr/local/lib/python3.10/dist-packages/prot/
from typing_extensions import Unpack
import lightkurve as lk
# result = lk.search_lightcurve("TIC 149308317", author="tess-spoc", sector=range(14), cadence=1800)
# result = lk.search_lightcurve("TOI 2202", author="spoc")
# lcs = result.download_all(flux_column="sap_flux").stitch()
import numpy as np

time, flux, fluxerr = np.loadtxt('/content/toi2202.txt', unpack=True)
time = time - 2457000
lcs = lk.LightCurve(time=time, flux=flux, flux_err=fluxerr)
from prot import WaveletTransform
wt = WaveletTransform.from_lightcurve(lcs)

wt.plot_all()


wt.period_at_max_power
import matplotlib.pyplot as plt

plt.plot(wt.period, wt.gwps)
plt.xscale('log')



```



### get formated output for Latex table

```Python
def get_formated_value(value, uerr, lerr):
    if abs(float(uerr)) == abs(float(lerr)):
        line = '{:.3f}\\pm{:.3f}'.format(float(value), abs(float(uerr)))
    else:
        line = '{:.3f}^{{+{:.3f}}}_{{-{:.3f}}}'.format(float(value), abs(float(uerr)), abs(float(lerr)))
    return line

```


### AAS like CSL
```Python
#AAS like CSL
  <bibliography second-field-align="flush" et-al-min="11" et-al-use-first="3" entry-spacing="0">
    <layout suffix=".">
      <text variable="citation-number" suffix=". "/>
      <group delimiter=". ">
        <text macro="contributors"/> 
     <text macro="date"/>
      </group>
      <text macro="container-title" prefix=". "/>
          <text variable="volume" prefix=", "/>
        <text variable="page" prefix=", "/>
    </layout>
  </bibliography>
```


### Avai Month

```Python
from astroplan import Observer
from astroplan import FixedTarget
from astropy.time import Time
from astropy.coordinates import SkyCoord
kitt = Observer.at_site('Kitt Peak')
ra = toi_table['RA (deg)']
dec = toi_table['Dec (deg)']
avai_month = []
for ii in range(len(ra)):
# for ii in range(20):
    ra_ = ra[ii]; dec_ = dec[ii]
    coordinates = SkyCoord(ra_, dec_, unit='deg')
    deneb = FixedTarget(name='Deneb', coord=coordinates)
    month_avi = []
    for i in range(1,13):
        time = Time('2023-{:02d}-01'.format(i))
        is_up = kitt.target_is_up(time, deneb)
        alt_ = kitt.altaz(time, deneb).alt.value; az_ = kitt.altaz(time, deneb).az.value
        if alt_>-20 and az_<90:
            # print('2023-{:02d}-01'.format(i), alt_, az_)
            month_avi.append(i)
    month_avi_str = ' '.join(['{:02d}'.format(i) for i in month_avi])+' *'
    avai_month.append(month_avi_str)
    # print(ii,month_avi_str)
toi_table['Available Month'] = avai_month
```


### skewed normal distribution

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

### get emcee mcmc save file


```Python
import sys
import scipy.stats as stats
import math
import matplotlib.gridspec as gridspec
import gzip, pickle
from scipy.stats import norm
from shutil import copyfile
import emcee
import numpy as np
copyfile('mcmc_save.h5', 'mcmc_save_tmp.h5')
sampler = emcee.backends.HDFBackend( 'mcmc_save_tmp.h5', read_only=True )
samples = sampler.get_chain(flat=True)#, discard=int(1.*1000/1))
```

### get_model (ellc)

```Python
import numpy as np
import matplotlib.pyplot as plt
import ellc

def get_model(x_lin, rr, rsuma, cosi, epoch, period, f_c, f_s, vsini, lambda_, q1, q2,q,st_rad):
    ar = (1+rr)/rsuma
    a = ar*st_rad
    radius_1 = rsuma/(1+rr)
    radius_2 = rsuma/(1+1/rr)
    lambda_1 = lambda_
    f_c = f_c; f_s = f_s
    q = q
    u1 = 2*np.sqrt(q1)*q2
    u2 = np.sqrt(q1)*(1-2*q2)
    idegree = np.arccos(cosi)*180/np.pi
    period = period
    epoch = epoch
    rm,_ = ellc.rv(x_lin,radius_1=radius_1,radius_2=radius_2,sbratio=0, vsini_1=vsini,lambda_1=lambda_1,f_c=f_c,f_s=f_s,
                   incl=idegree,a=a,q=q,ld_1='quad',ldc_1=[u1,u2],t_zero=epoch, period=period)
    rv,_ = ellc.rv(x_lin,radius_1=radius_1,radius_2=radius_2,sbratio=0, vsini_1=0,lambda_1=lambda_1,f_c=f_c,f_s=f_s,
                   incl=idegree,a=a,q=q,ld_1='quad',ldc_1=[u1,u2],t_zero=epoch, period=period,flux_weighted=False)
    transit = ellc.lc(x_lin,radius_1=radius_1,radius_2=radius_2,sbratio=0, vsini_1=vsini,lambda_1=lambda_1,f_c=f_c,f_s=f_s,
                    incl=idegree,a=a,q=q,ld_1='quad',ldc_1=[u1,u2],t_zero=epoch, period=period)
    return rm, rv, transit 

```
### get public data
```Python
import os
import time
import requests
# download the obliquity data from the website
target_url = 'https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv'
response = requests.get(target_url)
data = response.text
name = 'toi.csv'
date = time.strftime("%Y-%m-%d", time.localtime())
dated_name = 'toi_'+date+'.csv'
# check if the file exists
if os.path.exists(dated_name):
    print(dated_name+' exists')
else:
    print('downloading '+dated_name)
    with open(dated_name, 'w') as f:
        print(data, file=f)

    
import requests
# download the obliquity data from the website
target_url = 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+ps&format=csv'
response = requests.get(target_url)
data = response.text

name = 'pstable.csv'
date = time.strftime("%Y-%m-%d", time.localtime())
dated_name = 'pstable_'+date+'.csv'
# check if the file exists
if os.path.exists(dated_name):
    print(dated_name+' exists')
else:
    print('downloading '+dated_name)
    with open(dated_name, 'w') as f:
        print(data, file=f)


import requests
# download the obliquity data from the website
target_url = 'https://www.astro.keele.ac.uk/jkt/tepcat/obliquity.csv'
response = requests.get(target_url)
data = response.text

name = 'obliquity.csv'
date = time.strftime("%Y-%m-%d", time.localtime())
dated_name = 'obliquity_'+date+'.csv'
# check if the file exists
if os.path.exists(dated_name):
    print(dated_name+' exists')
else:
    print('downloading '+dated_name)
    with open(dated_name, 'w') as f:
        print(data, file=f)
	
	
toi_table = pd.read_csv('toi_'+date+'.csv', comment='#')
pstable = pd.read_csv('pstable_'+date+'.csv', comment='#')
obliquity_table = pd.read_csv('obliquity_'+date+'.csv', comment='#')

```

### GPT prompt for midjourney

```
Here is a MidJourney Prompt Formula:
(image we're prompting), (5 descriptive keywords), (camera type), (camera lens type), (time of day), (style of photograph), (type of film)

Please respond with "yes" if you understand the formula
```


### fetch S&F 2011 upper-limit AV
```Python
import requests
import xml.etree.ElementTree as ET

def fetch_and_parse_dust_data(ra_deg, dec_deg, name=None):
    
    if name is not None:
        url = "https://irsa.ipac.caltech.edu/cgi-bin/DUST/nph-dust?locstr="+name+"&regSize=2.0"
    else:
        url = "https://irsa.ipac.caltech.edu/cgi-bin/DUST/nph-dust?locstr="+str(ra_deg)+"+"+str(dec_deg)+"+equ+j2000&regSize=2.0"
    response = requests.get(url)

    if response.status_code == 200:
        content = response.content.decode("utf-8")
        root = ET.fromstring(content)
        return root
    else:
        print(f"Error: Unable to fetch data from the URL. Status code: {response.status_code}")
if __name__ == "__main__":
    ra_deg = 180
    dec_deg = 30
    root = fetch_and_parse_dust_data(ra_deg, dec_deg, name='HAT-P-7')
    for i in range(len(root)):
        for j in range(len(root[i])):
            if root[i][j].tag == 'statistics':
                for k in range(len(root[i][j])):
                    if root[i][j][k].tag == 'maxValueSandF':
                        maxValueSandF = root[i][j][k].text.strip().replace('(mag)', '')
                    # meanValueSandF
                    if root[i][j][k].tag == 'meanValueSandF':
                        meanValueSandF = root[i][j][k].text.strip().replace('(mag)', '')
                    # stdSandF
                    if root[i][j][k].tag == 'stdSandF':
                        stdSandF = root[i][j][k].text.strip().replace('(mag)', '')
    maxValueSandF, meanValueSandF, stdSandF = float(maxValueSandF), float(meanValueSandF), float(stdSandF)

    maxAv, meanAv, stdAv = maxValueSandF*3.1, meanValueSandF*3.1, stdSandF*3.1

    print("maxAv, meanAv, stdAv")
    print(maxAv, meanAv, stdAv)

```

### make latex ouput

```Python
import math
import numpy as np


def get_params_from_samples(samples):
    '''
    read MCMC or NS results and update params
    '''
    theta_median = np.nanpercentile(samples, 50, axis=0)
    theta_ul = np.nanpercentile(samples, 84, axis=0) - theta_median
    theta_ll = theta_median - np.nanpercentile(samples, 16, axis=0)

    return theta_median, theta_ll,theta_ul


#test
number = 0.0000214546
rounded_number = round(number, 2 - int(math.floor(math.log10(abs(number)))) - 1)
deci_num = 2 - int(math.floor(math.log10(abs(number)))) - 1
print(f"%.{deci_num}f" % rounded_number)

start_idx = 0
for key in data.columns:
    if 'b_rr' in key:
        start_idx = 1
    if start_idx == 1:
        linedata = data[key].to_numpy()
        linedata = linedata[int(len(linedata)/2):]
        theta_median, theta_ll, theta_ul = get_params_from_samples(linedata)
        # print(key,theta_median, theta_ll, theta_ul)
        number = np.min([theta_ll, theta_ul])
        rounded_number = round(number, 2 - int(math.floor(math.log10(abs(number)))) - 1)
        deci_num = 2 - int(math.floor(math.log10(abs(number)))) - 1
        formatted_string = f"{theta_median:.{deci_num}f}^{{+{theta_ul:.{deci_num}f}}}_{{-{theta_ll:.{deci_num}f}}}"
        print(key,formatted_string)
```
### Obs Month
```Python
from astroplan import Observer, FixedTarget
from astropy.time import Time
from astropy.coordinates import SkyCoord, EarthLocation
import astropy.units as u
import pandas as pd
import numpy as np
import warnings
from astroquery.gaia import Gaia

warnings.filterwarnings('ignore')


def observability_table_example(observer_location, star_data, start_time, end_time, time_resolution=31):
    observer = Observer(location=observer_location)
    targets = [FixedTarget(coord=SkyCoord(ra=ra, dec=dec, unit=('hourangle', 'deg')), name=name)
               for name, ra, dec in star_data]

    start_time = Time(start_time)
    end_time = Time(end_time)
    times = start_time + np.arange(0, (end_time - start_time).to(u.d).value, time_resolution) * u.d

    obs_table = observer.target_is_up(times, targets, grid_times_targets=True)

    obs_df = pd.DataFrame(obs_table.T, columns=[target.name for target in targets], index=times)

    return obs_df


def get_star_coordinates_from_gaia_id(gaia_id):
    job = Gaia.launch_job_async("SELECT * FROM gaiadr2.gaia_source WHERE source_id = " + str(gaia_id))
    results = job.get_results()
    ra = results['ra'][0]
    dec = results['dec'][0]
    return ra, dec


from astroplan import Observer

obs_observer = Observer.at_site("Kitt Peak")
observer_location = obs_observer.location


gaia_ids = [111111111111111]

for gaia_id in gaia_ids:
    ra, dec = get_star_coordinates_from_gaia_id(gaia_id)
    star_data = [('Star', ra * u.deg, dec * u.deg)]

    this_year = Time.now().decimalyear
    this_year = str(int(np.round(this_year, 0)))

    start_time = this_year + '-01-01 00:00:00'
    end_time = this_year + '-12-31 00:00:00'

    obs_df = observability_table_example(observer_location, star_data, start_time, end_time)
    obs = obs_df.to_numpy()
    obs = np.transpose(obs)

    month_ = np.arange(1, 13, 1)
    # print(this_year)

    obs_mon = month_[obs[0] == True]

    print(gaia_id, obs_mon)


```

### make new priors from mcmc table

```Python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from shutil import copyfile
from datetime import datetime

def mk_prior(work_folder):

    mcmc_save_table = pd.read_csv( os.path.join(work_folder, 'results/mcmc_table.csv') )
    mcmc_save_table['#name'].to_list()
    mcmc_save_table
    new_priors = []
    with open( os.path.join(work_folder, 'params.csv'), 'r') as f:
        for line in f.readlines():
            line = line.strip()
            name = line.split(',')[0]
            if '#' in name or name == '':
                # print(line)
                new_priors.append(line)
                continue
            new_table = mcmc_save_table[mcmc_save_table['#name'] == name]
            fit = 1
            if 'fixed' in new_table['lower_error'].values[0]:
                fit = 0
                new_priors.append(line)
            else:
                err = np.max([float(new_table['upper_error'].values[0]), float(new_table['lower_error'].values[0])])
                if '_rr' in name or '_rsuma' in name or 'q1' in name or 'q2' in name or 'cosi' in name:
                    # print(name)
                    new_priors.append(name + ',' + str(new_table['median'].values[0]) + ',' + str(fit) + ',' + 'uniform 0 1,' + ','.join(line.split(',')[4:]))
                elif 'f_c' in name or 'f_s' in name:
                    new_priors.append(name + ',' + str(new_table['median'].values[0]) + ',' + str(fit) + ',' + 'uniform -1 1,' + ','.join(line.split(',')[4:]))
                elif 'ln_err' in name or 'ln_jitter' in name:
                    new_priors.append(name + ',' + str(new_table['median'].values[0]) + ',' + str(fit) + ',' + 'uniform -15 0,' + ','.join(line.split(',')[4:]))
                elif 'lambda' in name:
                    new_priors.append(name + ',' + str(new_table['median'].values[0]) + ',' + str(fit) + ',' + 'uniform -360 360,' + ','.join(line.split(',')[4:]))
                elif '_K' in name:
                    new_priors.append(name + ',' + str(new_table['median'].values[0]) + ',' + str(fit) + ',' + 'uniform 0 10,' + ','.join(line.split(',')[4:]))                    
                elif '_period' in name or '_epoch' in name:
                    new_priors.append(name + ',' + str(new_table['median'].values[0]) + ',' + str(fit) + ',' + 'uniform '+str(new_table['median'].values[0]-1)+\
                                      ' '+str(new_table['median'].values[0]+1)+',' + ','.join(line.split(',')[4:]))  
                elif 'vsini' in name:
                    new_priors.append(name + ',' + str(new_table['median'].values[0]) + ',' + str(fit) + ',' + 'uniform 0 '+str(new_table['median'].values[0]+5)+',' + ','.join(line.split(',')[4:]))
                else:
                    new_priors.append(name + ',' + str(new_table['median'].values[0]) + ',' + str(fit) + ',' + 'normal '+str(new_table['median'].values[0])\
                    +' '+ str(err) + ',' + ','.join(line.split(',')[4:]))
    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d-%H-%M-%S"); current_time = current_time.replace('-','_')
    old_prior = os.path.join(work_folder, 'params.csv')
    cp_old_prior = os.path.join(work_folder, 'params_'+current_time+'.csv')
    copyfile(old_prior, cp_old_prior)

    with open(old_prior, 'w') as f:
        for line in new_priors:
            f.write(line+'\n')  
      
if __name__ == '__main__':
    pass
    work_folder = '..'
    mk_prior(work_folder)
```

### plot arrow for fig 

```Python
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.transforms as mtransforms

x0 = -0.1

arrow_style="simple,head_length=15,head_width=30,tail_width=10"
rect_style="simple,tail_width=25"
line_style="simple,tail_width=1"

fig, ax = plt.subplots()

# the x coords of this transformation are axes, and the y coord are data
trans = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)

y_tail = 5
y_head = 15
arrow1 = mpatches.FancyArrowPatch((x0, y_tail), (x0, y_head), arrowstyle=arrow_style, transform=trans)
arrow1.set_clip_on(False)
ax.add_patch(arrow1)

y_tail = 40
y_head = 60
arrow2 = mpatches.FancyArrowPatch((x0, y_tail), (x0, y_head), arrowstyle=arrow_style, facecolor='gold', edgecolor='black', linewidth=1, transform=trans)
arrow2.set_clip_on(False)
ax.add_patch(arrow2)

y_tail = 20
y_head = 40
rect_backgr = mpatches.FancyArrowPatch((x0, y_tail), (x0, y_head), arrowstyle=rect_style, color='white', zorder=0, transform=trans)
rect_backgr.set_clip_on(False)
rect = mpatches.FancyArrowPatch((x0, y_tail), (x0, y_head), arrowstyle=rect_style, fill=False, color='orange', hatch='///', transform=trans)
rect.set_clip_on(False)
ax.add_patch(rect_backgr)
ax.add_patch(rect)

line = mpatches.FancyArrowPatch((x0, 0), (x0, 80), arrowstyle=line_style, color='orange', transform=trans, zorder=-1)
line.set_clip_on(False)
ax.add_patch(line)

ax.set_xlim(0, 30)
ax.set_ylim(0, 80)
plt.show()
```

