### comment block generator
```
https://blocks.jkniest.dev/
```


### Time Stamp
```
HIERARCH TNG DRS BJD: BJD UTC
HIERARCH TNG QC BJD: BJD TDB
HIERARCH ESO DRS BJD: BJD UTC

```


### Common Linux command

```
rsync -av --max-size=500k a/ b/

find . -type f \( -name "*.txt" -o -name "*.sbatch" \) -exec sed -i 's/13fits/14fitsvsini/g' {} +


find . -type f -name 'quartzrun.sbatch' -exec grep -l 'mail-type' {} \; | while read file; do
    sed -i '/mail-type/c\#SBATCH --mail-type=FAIL' "$file"
done



find . -type f -name 'xxx' -exec sed -i '3i#SBATCH --mem=200GB' {} \;

find . -type f -name '*.eps' -exec convert {} {}.png \;

find . -type d | while read -r dir; do echo "$dir: $(find "$dir" -maxdepth 1 -type f | wc -l)"; done | sort -t: -k2 -nr | head -n 20
grep -v 'nan' input.txt > output.txt # remove nan
find . -name "*.idl" -type f -delete
find . -type f -name '*.txt' -exec rm {} \;
find . -name "*.eps" -type f -exec bash -c 'epstopdf "$0" "${0%.eps}.pdf"' {} \;
find . -name "*.ps" -type f -exec bash -c 'ps2pdf "$0" "${0%.ps}.pdf"' {} \;

```

### Font Style

```Python
# Set Times New Roman as the base font and override math text fonts
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'  # Roman (normal) font
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'  # Italic font
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'  # Bold font
```

### Plot TTV
```Python
# %%
import sys
import scipy.stats as stats
import math
import matplotlib.gridspec as gridspec
import gzip, pickle
from scipy.stats import norm
from shutil import copyfile
import emcee
import numpy as np

# %%
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

# %%
# %%file midtimes.txt
# 2460155.87091544,0.0005629751176654147,BJD_TDB,TESS,1
# 2460158.182084075,0.0006478729692925299,BJD_TDB,TESS,1
# 2460160.4927127236,0.0006271762322530238,BJD_TDB,TESS,1
# 2460162.8043466494,0.0017837301621797712,BJD_TDB,TESS,1
# 2460169.736747653,0.0005201690547712917,BJD_TDB,TESS,1
# 2460172.047833772,0.0005780620460642341,BJD_TDB,TESS,1
# 2460174.3591026776,0.0006689517338831407,BJD_TDB,TESS,1
# 2460176.6683759075,0.0006557685951709537,BJD_TDB,TESS,1
# 2459062.7819910124,0.0005235938637403539,BJD_TDB,TESS,1
# 2459065.0928612784,0.0005591888543168892,BJD_TDB,TESS,1
# 2459067.4036947577,0.0006386582555319323,BJD_TDB,TESS,1
# 2459069.7152420655,0.0006409538898839312,BJD_TDB,TESS,1
# 2459076.6470739846,0.0005920823323588776,BJD_TDB,TESS,1
# 2459078.956878636,0.003528929139763574,BJD_TDB,TESS,1
# 2459081.270241118,0.0006440552732835145,BJD_TDB,TESS,1
# 2459083.58036514,0.0006365612535308274,BJD_TDB,TESS,1
# 2458325.5848541623,0.005280898502174849,BJD_TDB,TESS,1
# 2458327.893462148,0.0006085523243624557,BJD_TDB,TESS,1
# 2458330.2039186475,0.0006206349242419661,BJD_TDB,TESS,1
# 2458332.515316971,0.0005434501296873212,BJD_TDB,TESS,1
# 2458334.8254373507,0.0005546215690210428,BJD_TDB,TESS,1
# 2458337.1375653213,0.0005862716645603745,BJD_TDB,TESS,1
# 2458341.7594227656,0.0005381643759153594,BJD_TDB,TESS,1
# 2458344.07032893,0.0006169268639943809,BJD_TDB,TESS,1
# 2458346.38130746,0.0005412695374159326,BJD_TDB,TESS,1
# 2458351.002872908,0.0008729693315132663,BJD_TDB,TESS,1



data = np.loadtxt('midtimes.txt', delimiter=',', dtype=str)
sort_idx = np.argsort(data[:,0].astype(float))
data = data[sort_idx]
np.savetxt('midtimes.txt', data, delimiter=',', fmt='%s')
filename = 'midtimes.txt'

period = 2.3109650
midtimes = data[:,0].astype(float)
midtimes_err = data[:,1].astype(float)
sort = np.argsort(midtimes)
midtimes = midtimes[sort]
midtimes_err = midtimes_err[sort]
t0_init = midtimes[int(len(midtimes)/2)]
t0_err_init = midtimes_err[int(len(midtimes)/2)]
epoch = np.round((midtimes - t0_init)/period)
epoch = np.array(epoch); time = np.array(midtimes); err = np.array(midtimes_err)

# %%
x = epoch
y = time
yerr = err
A = np.vander(x, 2)
C = np.diag(yerr * yerr)
ATA = np.dot(A.T, A / (yerr**2)[:, None])
cov = np.linalg.inv(ATA)
w = np.linalg.solve(ATA, np.dot(A.T, y / yerr**2))
print("Least-squares estimates:")
print("m = {0:.7f} ± {1:.7f}".format(w[0], np.sqrt(cov[0, 0])))
print("b = {0:.7f} ± {1:.7f}".format(w[1], np.sqrt(cov[1, 1])))


# %%
def covar(x,y):
    mx = x.mean(); my = y.mean()
    stdx = x.std();stdy = y.std()
    covxy = np.cov(x, y)
    return covxy[0, 1]

def read_data(filename):
    data = np.loadtxt(filename, delimiter=',', dtype=str)
    midtimes = data[:, 0].astype(float)
    midtimeserr = data[:, 1].astype(float)
    return midtimes, midtimeserr

def least_square(x, y, yerr):
    A = np.vander(x, 2)
    C = np.diag(yerr * yerr)
    ATA = np.dot(A.T, A / (yerr ** 2)[:, None])
    cov = np.linalg.inv(ATA)
    w = np.linalg.solve(ATA, np.dot(A.T, y / yerr ** 2))
    m = w[0]; merr = np.sqrt(cov[0, 0])
    b = w[1]; berr = np.sqrt(cov[1, 1])
    residual = y - (x*m+b)
    covar_value = cov[0, 1]
    y_model = m*x+b

    sum_ = 0
    for j in range(len(y)):
        sum_ = sum_ + (y[j] - y_model[j])**2 / yerr[j]**2

    reduced_chisq = sum_ / (len(y)-2)
    return m, merr, b, berr, covar_value,reduced_chisq


def find_optimal_epoch(ref_period, filename):
    y, yerr = read_data(filename)
    x = np.round((y - y[0])/ref_period, 0)
    covar_list = []
    optimal_epoch = 0
    minimum_covar= 1e4
    optimal_chisq = 0
    for i in range(-2000,2000):
        x = np.round((y - y[0])/ref_period, 0)
        x = x + i
        m, merr, b, berr, covar_value,reduced_chisq = least_square(x, y ,yerr)
        if abs(covar_value) < abs(minimum_covar):
            minimum_covar = covar_value
            optimal_epoch = x
            optimal_chisq = reduced_chisq
        covar_list.append(covar_value)
    return optimal_epoch, y, yerr, optimal_chisq


ref_period = period
x, y, yerr,optimal_chisq = find_optimal_epoch(ref_period, 'midtimes.txt')
yerr = yerr*optimal_chisq**0.5
m, merr, b, berr, covar_value,reduced_chisq = least_square(x, y, yerr)
oc = y - (x*m + b)
idx = np.argsort(x)

length = x[-1] - x[0]
xseq = np.linspace(np.min(x)-1/4*length, np.max(x)+1/4*length, 1000)
best_t0 = b
best_epoch = x
period,best_epoch


import numpy as np
import emcee

def linear_model(x, m, b):
    return m * x + b
def log_likelihood(params, x, y, yerr):
    m, b = params
    model = linear_model(x, m, b)
    inv_sigma2 = 1.0 / yerr**2
    return -0.5 * np.sum((y - model)**2 * inv_sigma2 + np.log(inv_sigma2))
def log_prior(params):
    m, b = params
    if period-1 < m < period+1 and best_t0-1 < b < best_t0+1:
        return 0.0
    return -np.inf
def log_probability(params, x, y, yerr):
    lp = log_prior(params)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(params, x, y, yerr)
true_m, true_b = period, best_t0

x = best_epoch
y = time
yerr = err

nwalkers, ndim = 32, 2
pos = np.array([true_m, true_b]) + 1e-4 * np.random.randn(nwalkers, ndim)
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(best_epoch, y, yerr))
sampler.run_mcmc(pos, 50000, progress=True)
samples = sampler.get_chain(discard=200, thin=15, flat=True)
m_perc = np.percentile(samples[:, 0], [16, 50, 84])
b_perc = np.percentile(samples[:, 1], [16, 50, 84])
m_err = np.diff(m_perc)
b_err = np.diff(b_perc)
print(f"Best-fit parameters: m = {m_perc[1]:.17f} +{m_err[1]:.17f} -{m_err[0]:.17f}, b = {b_perc[1]:.17f} +{b_err[1]:.17f} -{b_err[0]:.17f}")


import matplotlib.pyplot as plt
ms = samples[:,0]; bs = samples[:,1]

newepoch = np.linspace(best_epoch[0], best_epoch[-1], 1000)
med_pred_time = np.median(ms[:,None])*newepoch[None,:] + np.median(bs[:,None])
med_pred_time_obs = np.median(ms[:,None])*best_epoch[None,:] + np.median(bs[:,None])

med_pred_time[0]
pred_mid = ms[:,None]*newepoch[None,:] + bs[:,None]
plt.figure(figsize=(12,6))
ax = plt.gca()

datas = []
for i in range(len(pred_mid)):
    # ax.plot(newepoch, (pred_mid[i] - med_pred_time[0])*24*60, "C1", alpha=0.1, zorder=0)
    datas.append((pred_mid[i] - med_pred_time[0])*24*60)
datas = np.asarray(datas)
confs = [68.3, 95.4, 99.7]
for conf in confs:

    pred_data_array = np.percentile(
            datas,
            [50-conf/2, 50, 50+conf/2],
            axis=(0),
        )
    art = ax.fill_between(
        newepoch, pred_data_array[0], pred_data_array[2], color="C0", alpha=0.3,zorder=10
    )
    art.set_edgecolor("none")
ax.errorbar(best_epoch, (time-med_pred_time_obs)[0]*24*60, yerr=err*24*60, fmt='o', color='#f27100', label='observed', zorder=20,\
            capsize=7.5, elinewidth=2, ms=7, ecolor='k', mec='k', mew=1)
ticksetax(ax)
ax.set_xlabel('Epoch', fontsize=20)
ax.set_ylabel('O-C (min)', fontsize=20)
# ax.set_xlim(np.min(newepoch), np.max(newepoch))
plt.savefig('oc.pdf',bbox_inches='tight')
oc = (time-med_pred_time_obs)[0]*24*60

for i in range(len(best_epoch)):
    print(int(best_epoch[i]), '%.7f'%time[i],oc[i],err[i]*24*60)

```


### Clean Warm Jupiters Sample

```Python
import os
import time
import requests
import pandas as pd
import numpy as np
import astropy.units as u
import astropy.constants as c
import matplotlib.pyplot as plt

name = 'toi.csv'
date = time.strftime("%Y-%m-%d", time.localtime())
dated_name = 'toi_'+date+'.csv'
if os.path.exists(dated_name):
    print(dated_name+' exists')
else:
    target_url = 'https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv'
    response = requests.get(target_url)
    data = response.text
    print('downloading '+dated_name)
    with open(dated_name, 'w') as f:
        print(data, file=f)
toi_table = pd.read_csv(dated_name)
os.system('rm '+dated_name)

toi =toi_table



toi = toi[toi['TESS Mag'] < 13.5]
toi = toi[toi['Planet Radius (R_Earth)'] > 11.208981*0.8]
toi = toi[(toi['Period (days)'] > 8) & (toi['Period (days)'] < 200)]


st_mass = toi['Stellar Mass (M_Sun)'].values
period = toi['Period (days)'].values

period = period*u.day
st_mass = st_mass*u.Msun
##############################################
## Keep only the planets that have high probability
##############################################
np.unique(toi['TESS Disposition'].values,return_counts=True)
toi = toi[(toi['TFOPWG Disposition'] == 'CP' ) | (toi['TFOPWG Disposition'] == 'PC') | (toi['TFOPWG Disposition'] == 'APC')]
toi = toi[(toi['TESS Disposition'] == 'CP' ) | (toi['TESS Disposition'] == 'PC') ]
toi = toi[toi['Planet SNR'] > 20]


toi = toi[toi['Period (days)']<200]
idx = []
for i in range(len(toi)):
    if 'eb' in  str(toi['Comments'].values[i]).lower() or 'sb' in  str(toi['Comments'].values[i]).lower() or 'v-shape' in str(toi['Comments'].values[i]).lower():
        pass
    else:
        idx.append(i)
        pass
toi_wj = toi.iloc[idx]
tic_id = toi_wj['TIC ID'].values
planet_snr = toi_wj['Planet SNR'].values
tois = toi_wj['TOI'].values
toi_wj.to_csv('toi_wj.csv',index=False)
tic_uniq = np.unique(tic_id)

```


### Python Opening

```
import numpy as np
import pandas as pd
import os
import sys
from astroquery.simbad import Simbad
from astroquery.gaia import Gaia
import astropy
from wotan import flatten, transit_mask
import lightkurve as lk
import matplotlib.pyplot as plt
import warnings
```

### tic2teffmet
```
import json
import requests
import numpy as np

def tic2teffmet(tic):
    url = 'https://exofop.ipac.caltech.edu/tess/target.php?id='+str(tic)+'&json'
    response = requests.get(url)
    data = json.loads(response.text)
    stellar_parameters = data['stellar_parameters']
    teffs = []
    mets = []
    for i in range(len(stellar_parameters)):
        try:
            tel = stellar_parameters[i]['tel']
        except:
            continue
        if tel =='':
            continue
        teff = stellar_parameters[i]['teff']
        teff_e = stellar_parameters[i]['teff_e']
        if teff_e == None:
            teff_e = 100.123    
        logg = stellar_parameters[i]['logg']
        if logg == None:
            logg = 4.5123
        logg_e = stellar_parameters[i]['logg_e']
        vsini = stellar_parameters[i]['vsini']
        vsini_e = stellar_parameters[i]['vsini_e']
        if vsini_e == None:
            vsini_e = 0.123
        met = stellar_parameters[i]['met']
        met_e = stellar_parameters[i]['met_e']
        if met_e == None:
            met_e = 0.1
        teffs = teffs + list(np.random.normal(float(teff), float(teff_e), 1000))
        mets = mets + list(np.random.normal(float(met), float(met_e), 1000))
        
    if len(teffs) == 0:
        return None, None, None, None
    else:
        teffs = np.asarray(teffs).flatten()
        mets = np.asarray(mets).flatten()
        return np.median(teffs), np.std(teffs), np.median(mets), np.std(mets)
    
    
tic = 337217173
teff, teff_e, met, met_e = tic2teffmet(tic)
print(teff, teff_e, met, met_e)
```



### fit.pro2vmarg
```
vm_str = '''
exofastv2,nplanets=1,rvpath='/N/slate/xwa5/xianyuwangfolder/wjs/XO3_EXOFAST/DT/n*RV.rv', tranpath='/N/slate/xwa5/xianyuwangfolder/wjs/XO3_EXOFAST/DT/n*flux',ttvpath='/N/slate/xwa5/xianyuwangfolder/wjs/XO3_EXOFAST/DT/n20240817.b.ttv',$
          priorfile='/N/slate/xwa5/xianyuwangfolder/wjs/XO3_EXOFAST/DT/8400842.priors',prefix='/N/slate/xwa5/xianyuwangfolder/wjs/XO3_EXOFAST/DT/fitresults/8400842.',$
          mistsedfile='/N/slate/xwa5/xianyuwangfolder/wjs/XO3_EXOFAST/DT/8400842.sed', $
		  dtpath='/N/slate/xwa5/xianyuwangfolder/wjs/XO3_EXOFAST/DT/*.fits',$
          maxsteps=100000,nthin=10, fittran=[1],fitrv=[1],$
          debug=debug, nthreads=63, restorebest=1,verbose=verbose,maxgr=1.1, mintz=200,$
          stopnow=0,maxtime=43200,OPTMETHOD='de',optcriteria=1,skiptt=1
exit




'''
vm_str = vm_str.replace('exofastv2', '').replace('exit', '').replace('$', '').replace('\n', '').replace('\'', '').replace(' ', '').replace('","', '|||').replace('\t', '')

for line in vm_str.split(','):
    if len(line) > 0:
        if 'debug' in line:
            continue
        if 'verbose' in line:
            continue
        print(line)



```


### tic 2 tefff
```Python
import os
import time
import requests
import pandas as pd
# download the obliquity data from the website


name = 'pscomppars.csv'
date = time.strftime("%Y-%m-%d", time.localtime())
dated_name = 'pscomppars_'+date+'.csv'
# check if the file exists
if os.path.exists(dated_name):
    print(dated_name+' exists')
else:
    target_url = 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+pscomppars&format=csv'
    response = requests.get(target_url)
    data = response.text
    print('downloading '+dated_name)
    with open(dated_name, 'w') as f:
        print(data, file=f)
pscomppars_table = pd.read_csv('pscomppars_'+date+'.csv', comment='#')
import os
os.system('rm pscomppars_'+date+'.csv')
def tic2teff(tic_id):
    this_row = pscomppars_table[pscomppars_table['tic_id'] == tic_id]
    st_teff = this_row['st_teff'].values[0]
    st_tefferr1 = this_row['st_tefferr1'].values[0]
    st_tefferr2 = this_row['st_tefferr2'].values[0]
    return st_teff, st_tefferr1, st_tefferr2
```


Read HARPS and HARPS_N

```Python

import os
import astropy.io.fits as fits

path = '/Volumes/TESS_GO/TOI-892'


fits_files = [x for x in os.listdir(path) if x.endswith('.fits.gz') and 'G2_A' in x and '._' not in x]

times = np.asarray([]); rvs = np.asarray([]); rvs_err = np.asarray([])
for fits_file in fits_files:
    header = fits.open(os.path.join(path, fits_file))[0].header
    rv = header['HIERARCH TNG DRS CCF RV']
    rv_err = header['HIERARCH TNG DRS CCF NOISE']
    mjd = header['MJD-OBS']
    times = np.append(times, mjd); rvs = np.append(rvs, rv); rvs_err = np.append(rvs_err, rv_err)
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 5))
plt.errorbar(times, rvs, yerr=rvs_err, fmt='o')

# ccf_G2_A.fits
# 
root_paht = '/Volumes/TESS_GO/WASP_87/archive/'

# find all files ending with ccf_G2_A.fits

times = np.asarray([]); rvs = np.asarray([]); rvs_err = np.asarray([])
for root, dirs, files in os.walk(root_paht):
    for file in files:
        if file.endswith('ccf_G2_A.fits') and '._' not in file:
            print(file)
            header = fits.open(os.path.join(root, file))[0].header
            try:
                rv = header['HIERARCH ESO DRS CCF RV']
                rv_err = header['HIERARCH ESO DRS CCF NOISE']
                mjd = header['MJD-OBS']
                times = np.append(times, mjd); rvs = np.append(rvs, rv); rvs_err = np.append(rvs_err, rv_err)
            except:
                pass
            
plt.figure(figsize=(10, 5))
plt.errorbar(times%1.6827, rvs, yerr=rvs_err, fmt='o')




```



Clean SG2

```Python
import os
import time
import requests
import pandas as pd
# download the obliquity data from the website


name = 'pscomppars.csv'
date = time.strftime("%Y-%m-%d", time.localtime())
dated_name = 'pscomppars_'+date+'.csv'
# check if the file exists
if os.path.exists(dated_name):
    print(dated_name+' exists')
else:
    target_url = 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+pscomppars&format=csv'
    response = requests.get(target_url)
    data = response.text
    print('downloading '+dated_name)
    with open(dated_name, 'w') as f:
        print(data, file=f)
pscomppars_table = pd.read_csv('pscomppars_'+date+'.csv', comment='#')



import numpy as np
ps_tics = pscomppars_table['tic_id']
ps_tics_list = []
for tic in ps_tics:
    if 'TIC' in str(tic):
        ps_tics_list.append(tic.split('TIC ')[1])
ps_tics_list = np.array(ps_tics_list).astype(int)





SG2_SG4_Aug21 = pd.read_csv('SG2_SG4_Aug21.csv')
allowed_keys = ['KP', 'P', 'CPC', 'APC', 'VP', 'CPC-', 'VPC',  'VPC+',  'PC',  'NPC', 'VPC-',
                'PPC', 'VPC+?', 'VPC?', 'VPC-+', 'CPC?',  'VPC-?',  'NPC?', 'CP']
SG2_SG4_Aug21 = SG2_SG4_Aug21[SG2_SG4_Aug21['SG1 Disposition'].isin(allowed_keys)]
SG2_SG4_Aug21 = SG2_SG4_Aug21[~SG2_SG4_Aug21['TIC'].isin(ps_tics_list)]

remove_idx = []
for i in range(len(SG2_SG4_Aug21)):
    this_sg2_note = SG2_SG4_Aug21['Facilities/Teams Planning SG2 Observations \n(spectra in hand)'].iloc[i]
    if 'sb' in str(this_sg2_note).lower() or 'bd' in str(this_sg2_note).lower():
        remove_idx.append(i)
        
SG2_SG4_Aug21 = SG2_SG4_Aug21.drop(SG2_SG4_Aug21.index[remove_idx])
SG2_SG4_Aug21['Rp']

import matplotlib.pyplot as plt

_ = plt.hist(SG2_SG4_Aug21['Rp'], bins=2000)
plt.xlim(0, 50)
plt.xlabel('Rp (R_Earth)')
plt.ylabel('Number of planets')
plt.axvline(2.5, color='r', linestyle='--')
plt.axvline(12.5, color='r', linestyle='--')
plt.show()
```


### ar 2 period

```
# ar to period


def period2ar(st_mass, st_rad, period):
    import astropy.units as u
    import astropy.constants as const

    # host star mass
    st_mass = st_mass * u.Msun
    st_rad = st_rad * u.Rsun

    pl_orbperiod = period * u.day
    pl_orbsmax = (const.G * st_mass * pl_orbperiod**2 / (4 * np.pi**2))**(1/3)
    pl_ratdor = pl_orbsmax / st_rad
    value = pl_ratdor.to(u.dimensionless_unscaled)
    return value


def ar2period(st_mass, st_rad, ar):
    import astropy.units as u
    import astropy.constants as const

    # host star mass
    st_mass = st_mass * u.Msun
    st_rad = st_rad * u.Rsun
    
    pl_ratdor = ar
    pl_orbsmax = st_rad * pl_ratdor
    pl_orbperiod = np.sqrt(4 * np.pi**2 * pl_orbsmax**3 / (const.G * st_mass))
    
    value = pl_orbperiod.to(u.day)
    
    return value
```



### change starry image color to black

```
### ~L520
            if ax is None:
                fig, ax = plt.subplots(1, figsize=figsize)
                fig.patch.set_facecolor('k')
                # fig.patch.set_alpha(0.7)
            else:
                fig = ax.figure
            ax.axis("off")
            ax.set_facecolor("k")
            
            ax.set_xlim(-1.05, 1.05)
            ax.set_ylim(-1.05, 1.05)
            
            dx = 2.0 / image.shape[1]
            extent = (-1 - dx, 1, -1 - dx, 1)

            # Anti-aliasing at the edges
            xp, yp, xm, ym = self._get_ortho_borders()
            borders += [
                ax.fill_between(xp, 1.1 * yp, yp, color="k", zorder=-1)
            ]
            borders += [
                ax.fill_between(xm, 1.1 * ym, ym, color="k", zorder=-1)
            ]

```


### stitch NEID spectra
```
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# minimize
import os
from scipy.optimize import minimize
from astropy.io import fits


def residual(norm_factor, flux_order, blaze_order):
    resi = flux_order / (blaze_order / norm_factor)
    median_resi = np.median(resi)
    sigma_resi = np.std(resi)
    sigma_1_idx = np.where((resi > median_resi - 2 * sigma_resi) & (resi < median_resi + 2 * sigma_resi))
    sigma_1_idx = np.where((resi > median_resi - 0.1 * sigma_resi))
    return ((resi[sigma_1_idx] - 1) ** 2).sum()


fits_files = [x for x in os.listdir() if x.endswith('.fits')]

for j, filename in enumerate(fits_files):
    print(j, len(fits_files), filename)
    hdul = fits.open(filename)

    flux = hdul['SCIFLUX'].data
    wave = hdul['SCIWAVE'].data
    SCIBLAZE = hdul['SCIBLAZE'].data
    SKYBLAZE = hdul['SKYBLAZE'].data

    plt.figure(figsize=(10, 5))
    norm_factor = 10
    waves = np.asarray([])
    fluxes = np.asarray([])



    norm_factor_50 = 0
    norm_factor_78 = 0
    for order in range(47,84):
    # for order in range(47,49):
        wave_order = wave[order]; flux_order = flux[order]
        idx = np.where(flux_order < 1e99)
        wave_order = wave_order[idx]; flux_order = flux_order[idx]
        blaze_order = SCIBLAZE[order][idx]
        half_space = 2300
        start_idx = int(4608-half_space-500-(order-46)*10)
        end_idx = int(4608+half_space-1000+(order-46)*30)
        res = minimize(residual, norm_factor, args=(flux_order[start_idx:end_idx], blaze_order[start_idx:end_idx]), method='nelder-mead', options={'xatol': 1e-8, 'disp': False})
        norm_factor =  res.x[0]
        if order==50:
            norm_factor_50 = norm_factor
        if order==78:
            norm_factor_78 = norm_factor
    # print(norm_factor_50, norm_factor_78)
    tmp_wave_77 = None
    tmp_wave_78 = None
    offset = 0
    for order in range(45,84):
    # for order in range(47,49):
        wave_order = wave[order]; flux_order = flux[order]
        idx = np.where(flux_order < 1e99)
        wave_order = wave_order[idx]; flux_order = flux_order[idx]
        blaze_order = SCIBLAZE[order][idx]
        half_space = 2300
        start_idx = int(4608-half_space-500-(order-46)*10)
        end_idx = int(4608+half_space-1000+(order-46)*30)
        res = minimize(residual, norm_factor, args=(flux_order[start_idx:end_idx], blaze_order[start_idx:end_idx]), method='nelder-mead', options={'xatol': 1e-8, 'disp': False})
        norm_factor =  res.x[0]
        norm_flux = flux_order/blaze_order*norm_factor


        wave_start_end = wave_order[start_idx:end_idx]
        norm_flux_start_end = norm_flux[start_idx:end_idx]
        wave_start_end_fit = wave_start_end - np.mean(wave_start_end)
        wave_start_end = wave_order[start_idx:end_idx]
        norm_flux_start_end = flux_order[start_idx:end_idx]/( blaze_order[start_idx:end_idx]/norm_factor_50)
        
        if order==77:
            idx_6400_6420 = np.where((wave_start_end > 6416) & (wave_start_end < 6420))
            tmp_wave_77 = wave_start_end[idx_6400_6420]
            tmp_flux_77 = norm_flux_start_end[idx_6400_6420]
            tmp_wave_77_resample = np.arange(6416, 6420, 0.001)
            tmp_flux_77_resample = np.interp(tmp_wave_77_resample, tmp_wave_77, tmp_flux_77)
        if order==78:
            idx_6400_6420 = np.where((wave_start_end > 6416) & (wave_start_end < 6420))
            tmp_wave_78 = wave_start_end[idx_6400_6420]
            tmp_flux_78 = norm_flux_start_end[idx_6400_6420]
            tmp_wave_78_resample = np.arange(6416, 6420, 0.001)
            tmp_flux_78_resample = np.interp(tmp_wave_78_resample, tmp_wave_78, tmp_flux_78)
            offset = np.nanmedian(tmp_flux_77_resample/tmp_flux_78_resample)
            print(offset)
        if order > 77:
            norm_flux_start_end = norm_flux_start_end*offset

        idx = np.where(norm_flux_start_end < 3)
        plt.text(np.mean(wave_start_end[idx]), 1.5, str(order), fontsize=8)
        plt.plot(wave_start_end[idx], norm_flux_start_end[idx],zorder=100,linewidth=0.5)
        waves = np.append(waves, wave_start_end[idx])
        fluxes = np.append(fluxes, norm_flux_start_end[idx])
    plt.axhline(1, color='black', linewidth=0.5)
    plt.xlim(6416, 6424)
    # plt.ylim(0.7, 1.1)
    plt.savefig(filename.strip('.fits')+'.png', dpi=300)
    np.savetxt(filename.strip('.fits')+'.txt', np.transpose([waves/10, fluxes, fluxes*0]), fmt='%.6f', delimiter='\t', header='waveobs flux err', comments='')
```

### get_exoclock_emp

```python
import urllib
import json

def get_exoclock_emp(name):
    exoclock_planets = json.loads(urllib.request.urlopen('https://www.exoclock.space/database/planets_json').read())
    try:
        ephem_mid_time = exoclock_planets[name.replace(' ', '')]['ephem_mid_time']
        ephem_mid_time_e1 = exoclock_planets[name.replace(' ', '')]['ephem_mid_time_e1']
        ephem_mid_time_e2 = exoclock_planets[name.replace(' ', '')]['ephem_mid_time_e2']
        ephem_period = exoclock_planets[name.replace(' ', '')]['ephem_period']
        ephem_period_e1 = exoclock_planets[name.replace(' ', '')]['ephem_period_e1']
        ephem_period_e2 = exoclock_planets[name.replace(' ', '')]['ephem_period_e2']
        ephem_parameters_ref = exoclock_planets[name.replace(' ', '')]['ephem_parameters_ref']
        return ephem_mid_time, ephem_mid_time_e1, ephem_mid_time_e2, ephem_period, ephem_period_e1, ephem_period_e2, ephem_parameters_ref
    except:
        print(name, 'is not in', exoclock_planets.keys())

```



### Unfinished Matlab Hirano2011; still slow

```Python
############2nd
beta = 1;
gamma = 1;
u1 = 0.3;
u2 = 0.2;
vsini = 3;
vp = 1; % km/s
zeta = 1;
sigma = 1;  
t = 1;      
f = 1;
costheta = 0;
sintheta = 0;
pi_squared = pi^2;


phi_arr = 0:.01:2*pi;
t_arr = 0:.01:1;
sigma_arr = 0:.01:10;
[phi, t, sigma] = meshgrid(phi_arr, t_arr, sigma_arr);

F = exp(-2*pi_squared*beta.^2*sigma.^2 - 4*pi.*gamma.*sigma) .* ...
                     (1 - u1 * (1 - sqrt(1 - t.^2)) - u2 * (1 - sqrt(1 - t.^2)).^2) ./ ...
                     (1 - u1 / 3 - u2 / 6) .* ...
                     (exp(-pi_squared * zeta.^2 .* sigma.^2 .* (1 - t.^2)) + exp(-pi.^2 * zeta.^2 .* sigma.^2 .* t.^2)) .* ...
                     cos(2 * pi .* sigma .* vsini .* t .* cos(phi)) .* ...
                     sin(2 * pi .* sigma .* vp) .* ...
                     0.5 .*  (exp(-(pi*zeta*costheta).^2 * sigma.^2) + exp(-(pi*zeta*sintheta).^2 * sigma.^2));


disp(Iz);

tic
% 1:y; 3; z; 2 x
Ix = trapz(phi_arr, F, 2);
Iy = trapz(t_arr, Ix, 1);
Iz = trapz(sigma_arr, Iy, 3);
toc







############# 1st
beta = 1;
gamma = 1;
u1 = 0.3;
u2 = 0.2;
vsini = 3;
vp = 1; % km/s
zeta = 1;
sigma = 1;  
t = 1;      
f = 1;
costheta = 0;
sintheta = 0;
pi_squared = pi^2;


f_upper = @(phi, t, sigma) exp(-2*pi_squared*beta.^2*sigma.^2 - 4*pi.*gamma.*sigma) .* ...
                     (1 - u1 * (1 - sqrt(1 - t.^2)) - u2 * (1 - sqrt(1 - t.^2)).^2) ./ ...
                     (1 - u1 / 3 - u2 / 6) .* ...
                     (exp(-pi_squared * zeta.^2 .* sigma.^2 .* (1 - t.^2)) + exp(-pi.^2 * zeta.^2 .* sigma.^2 .* t.^2)) .* ...
                     cos(2 * pi .* sigma .* vsini .* t .* cos(phi)) .* ...
                     sin(2 * pi .* sigma .* vp) .* ...
                     0.5 .*  (exp(-(pi*zeta*costheta).^2 * sigma.^2) + exp(-(pi*zeta*sintheta).^2 * sigma.^2));
f_lower = @(phi, t, sigma) exp(-2*pi_squared*beta.^2*sigma.^2 - 4*pi.*gamma.*sigma) .* ...
                     (1 - u1 * (1 - sqrt(1 - t.^2)) - u2 * (1 - sqrt(1 - t.^2)).^2) ./ ...
                     (1 - u1 / 3 - u2 / 6) .* ...
                     (exp(-pi.^2 * zeta.^2 .* sigma.^2 .* (1 - t.^2)) + exp(-pi.^2 * zeta.^2 .* sigma.^2 .* t.^2)) .* ...
                     cos(2 * pi .* sigma .* vsini .* t .* cos(phi)) .* ...
                     ((1 - u1 * (1 - sqrt(1 - t.^2)) - u2 * (1 - sqrt(1 - t.^2)).^2) ./ ...
                     (1 - u1 / 3 - u2 / 6) .* ...
                     (exp(-pi_squared * zeta.^2 .* sigma.^2 .* (1 - t.^2)) + exp(-pi_squared * zeta.^2 .* sigma.^2 .* t.^2)) .* ...
                     cos(2 * pi .* sigma .* vsini .* t .* cos(phi)) - f*0.5 .*  (exp(-(pi*zeta*costheta).^2 * sigma.^2) ...
                     + exp(-(pi*zeta*sintheta).^2 * sigma.^2)) .* cos(2 * pi .* sigma .* vp)  );
                     

tic
I_upper = integral3(f_upper, 0, 2*pi, 0, 1, 0, 5, 'AbsTol',1e-3, 'RelTol', 1e-3,'Method','tiled'); 
I_lower = integral3(f_lower, 0, 2*pi, 0, 1, 0, 5, 'AbsTol',1e-3, 'RelTol', 1e-3,'Method','tiled');
ratio = I_upper/ I_lower
toc

disp('The integral result is:');
disp(I);


% correct
% f = @(phi) cos(2 * pi * sigma * vsini * t * cos(phi));
% I = integral(f,0,2*pi)

% f = @(r,theta,phi,xi) r.^3 .* sin(theta).^2 .* sin(phi);
% Q = @(r) integral3(@(theta,phi,xi) f(r,theta,phi,xi),0,pi,0,pi,0,2*pi);
% I = integral(Q,0,2,'ArrayValued',true)

```



### Photoeccentric effect 


```Python
import numpy as np
import corner


ecc_list = np.asarray([])
weight_list = np.asarray([])
omega_list = np.asarray([])


rho_cir_median = 9
rho_cir_err = 0.01

for i in range(1000):
    N = 1000
    rho_circ = np.random.normal(rho_cir_median, rho_cir_err, N)
    ecc = np.random.uniform(0, 1, N)
    omega = np.random.uniform(-np.pi, np.pi, N)
    g = (1 + ecc * np.sin(omega)) / np.sqrt(1 - ecc**2)
    rho = rho_circ / g[:, None] ** 3

    log_weights = -0.5 * ((rho - 1) / 0.001) ** 2
    weights = np.exp(log_weights[:, 0] - np.max(log_weights[:, 0]))
    ecc_list = np.append(ecc_list, ecc)
    weight_list = np.append(weight_list, weights)
    omega_list = np.append(omega_list, omega)

q = corner.quantile(ecc_list, [0.16, 0.5, 0.84], weights=weight_list)
q50, q16, q84 = q

print(f"ecc = {q50:.2f} +{q84 - q50:.2f} {q50 - q16:.2f}")

# idx = np.where(weight_list > 0)
# ecc_list = ecc_list[idx]
# omega_list = omega_list[idx]
data = np.vstack([ecc_list, omega_list]).T

figure = corner.corner(
    data,
    labels=[r"$e$", r"$\omega$"],
    quantiles=[0.16, 0.5, 0.84],
    weights=weight_list,
)

```



### Get HARPS RV from ESO Archive


```Python
# Download HARPSSTAR .tar data

path = '/Volumes/TESS_GO/ESO_Data/HD189733/archive/'
tars = [x for x in os.listdir(path) if x.endswith('.tar') and not x.startswith('.')]
import glob

rvs = np.asarray([]); rv_errs = np.asarray([]); bjds = np.asarray([])
for tar in tars:
    os.system('tar -zxvf '+path+tar)
    fits_files = glob.glob(os.path.join('.', '**', '*ccf*.fits'), recursive=True)
    print(tar)
    hdu = fits.open(fits_files[0])
    hdu[0].header

    rv = hdu[0].header['HIERARCH ESO DRS CCF RVC']
    rv_err = hdu[0].header['HIERARCH ESO DRS CCF NOISE']
    bjd = hdu[0].header['HIERARCH ESO DRS BJD']
    # print(rv, rv_err, bjd
    rvs = np.append(rvs, rv); rv_errs = np.append(rv_errs, rv_err); bjds = np.append(bjds, bjd)
    os.system('rm -rf data')
    


```

### download HARPS RV 2003 to 2023
```Python
import astropy.io.fits as fits
import numpy as np
import os, sys
# https://arxiv.org/abs/2312.06586

os.system('wget https://dataportal.eso.org/dataPortal/file/ADP.2023-12-04T15:16:53.464 && mv ADP.2023-12-04T15:16:53.464 HARPS_RV.fits')
hdu = fits.open('HARPS_RV.fits')

time = np.asarray([])
rv = np.asarray([])
rv_err = np.asarray([])
for i in range(len(csv)):
    if 'KELT-10' in csv['tel_object'][i]:
        # print(csv['tel_object'][i])
        drs_bjd = csv['drs_bjd'][i]
        drs_ccf_rvc = csv['drs_ccf_rvc'][i]
        drs_ccf_rv = csv['drs_ccf_rv'][i]
        drs_ccf_noise = csv['drs_ccf_noise'][i]
        time = np.append(time, drs_bjd)
        rv = np.append(rv, drs_ccf_rvc)
        rv_err = np.append(rv_err, drs_ccf_noise)
import matplotlib.pyplot as plt
%matplotlib widget
plt.figure()
plt.errorbar(time, rv, yerr=rv_err, fmt='o')

```

### Time Converter

```Python

from astropy.time import Time
# Define your BJD in UTC
def bjdutc2bjdtbd(bjd_utc):
    time_bjd_utc = Time(bjd_utc, format='jd', scale='utc')
    time_bjd_tdb = time_bjd_utc.tdb
    return time_bjd_tdb
	
	
```


### NEID Proposal, Targets Info submission
```python
import os
import time
import requests
import pandas as pd
import os
import pandas as pd
import numpy as np
name = 'toi.csv'
date = time.strftime("%Y-%m-%d", time.localtime())
dated_name = 'toi_'+date+'.csv'
if os.path.exists(dated_name):
    print(dated_name+' exists')
else:
    target_url = 'https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv'
    response = requests.get(target_url)
    data = response.text
    print('downloading '+dated_name)
    with open(dated_name, 'w') as f:
        print(data, file=f)

toi_ids = '''111
222
333
444'''
toi_ids = [int(i) for i in toi_ids.split('\n')]
toi_ids = list(set(toi_ids)
toi_table = pd.read_csv('toi_'+date+'.csv', comment='#')
TOI = toi_table['TOI'].to_numpy()
TIC_ID = toi_table['TIC ID'].to_numpy()
ra = toi_table['RA'].to_numpy()
dec = toi_table['Dec'].to_numpy()
print('id,objectName,ra,dec,epoch,magnitude,filter,exposureTimeSeconds,numberOfExposures,skyCondition,seeing,comment')
for i in range(len(toi_ids)):
    idx = np.where(TOI == float(toi_ids[i])+0.01)[0]
    print(i+1, 'TOI'+str(toi_ids[i]),  ra[idx][0], dec[idx][0], '', '', '', '', '','','','', sep=',')
```

### download TFOP transits
```Python
import requests
import json

tic_id = '394050135'


if not os.path.exists(tic_id):
    os.makedirs(tic_id)

json_data = requests.get('https://exofop.ipac.caltech.edu/tess/target.php?id='+tic_id+'&json').json()


for ind_file in json_data['files']:
    if ind_file['ftype'] == 'Light_Curve':
        fname = ind_file['fname']
        fid = ind_file['fid']
        os.system('wget https://exofop.ipac.caltech.edu/tess/get_file.php?id='+str(fid)+' -O '+tic_id+'/'+fname)
```


### plots and tables 
```python
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


# data for the tables
data1 = np.ones((3,7))
data2 = np.ones((3,7))
data4 = np.ones((3,7))
data_all = [data1,data2,data4]


fig = plt.figure(constrained_layout = True, figsize = (11,7))
plt.rcParams["figure.autolayout"] = True

# grid for the tables
grid = gridspec.GridSpec(ncols=3, nrows=4, figure = fig, height_ratios=[1,1,1,2])
ax0 = fig.add_subplot(grid[0,:])
ax1 = fig.add_subplot(grid[1,:]) 
ax2 = fig.add_subplot(grid[2,:]) 

# grid for the plots
subgrid_spec = gridspec.GridSpecFromSubplotSpec(1, 3, subplot_spec=grid[3,:], wspace=0.1)
ax = [ax0,ax1,ax2]
ax00 =[plt.subplot(subgrid_spec[0, 0]),plt.subplot(subgrid_spec[0, 1]),plt.subplot(subgrid_spec[0, 2])]


# column labels and settings 
titles = ['title1','title2','title3']
columns = ("column1","column2","column3","column4","column5","column6","column7")
row = (' g ',' r ',' i ')
color = [['w','w','w','bisque','w','skyblue','w'],['w','w','w','bisque','w','w','w'],['w','w','w','bisque','w','w','w']]


# make the table
fig.suptitle("\n\nDe-red spec Template",fontsize = 14)
for i in range(len(ax)):
    ax[i].table(cellText=data_all[i],fontsize=13,colLabels=columns,rowLabels=row,cellLoc ='center',loc='center',cellColours=color,colWidths=colwidth1)
    ax[i].axis('tight')
    ax[i].axis('off')
    ax[i].set_title(titles[i],fontsize= 12)

# make the plots
for i in range(len(ax00)):
    # placeholder
    pass



```


### make TOI table with Gaia DR2/3 RUWE

```
import time
import pandas
import os
import requests
import matplotlib.pyplot as plt
import pandas as pd
name = 'toi.csv'
date = time.strftime("%Y-%m-%d", time.localtime())
dated_name = 'toi_'+date+'.csv'
# check if the file exists
if os.path.exists(dated_name):
    print(dated_name+' exists')
else:
    target_url = 'https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv'
    response = requests.get(target_url)
    data = response.text
    print('downloading '+dated_name)
    with open(dated_name, 'w') as f:
        print(data, file=f)

	
toi_table = pd.read_csv('toi_'+date+'.csv', comment='#')
TESS_Disposition = toi_table['TESS Disposition']
TFOPWG_Disposition = toi_table['TFOPWG Disposition']
unique_TESS_Disposition = TESS_Disposition.unique() #KP', 'CP', 'PC',
unique_TFOPWG_Disposition = TFOPWG_Disposition.unique() #KP', 'CP', 'PC',
# print(unique_TESS_Disposition, unique_TFOPWG_Disposition)
print('We only keep the ones with TESS Disposition == "KP" or "CP" or "PC" and TFOPWG Disposition == "KP" or "CP" or "PC"')
# toi_table = toi_table.query('`TESS Disposition` == "KP" or `TESS Disposition` == "CP" or `TESS Disposition` == "PC"')
# toi_table = toi_table.query('`TFOPWG Disposition` == "KP" or `TFOPWG Disposition` == "CP" or `TFOPWG Disposition` == "PC"')
print('The number of targets is', len(toi_table))   


#  -- output format : csv
#  SELECT "IV/39/tic82".TIC,  "IV/39/tic82".RAJ2000,  "IV/39/tic82".DEJ2000,  "IV/39/tic82".HIP,  "IV/39/tic82".TYC,  "IV/39/tic82".UCAC4,  "IV/39/tic82"."2MASS", 
#  "IV/39/tic82".objID,  "IV/39/tic82".WISEA,  "IV/39/tic82".GAIA,  "IV/39/tic82".KIC,  "IV/39/tic82"."S/G",  "IV/39/tic82".Ref,  "IV/39/tic82".pmRA,  "IV/39/tic82".e_pmRA, 
#  "IV/39/tic82".pmDE,  "IV/39/tic82".e_pmDE,  "IV/39/tic82".r_pm,  "IV/39/tic82".Teff,  "IV/39/tic82".s_Teff,  "IV/39/tic82".logg,  "IV/39/tic82".s_logg,  "IV/39/tic82"."[M/H]", 
#  "IV/39/tic82"."e_[M/H]",  "IV/39/tic82".Rad,  "IV/39/tic82".s_Rad,  "IV/39/tic82".Mass,  "IV/39/tic82".s_Mass,  "IV/39/tic82".LClass,  "IV/39/tic82".Dist,  "IV/39/tic82".s_Dist, 
#  "IV/39/tic82".Ncont,  "IV/39/tic82".Rcont,  "IV/39/tic82".r_Dist,    "IV/39/tic82".r_Teff,  "IV/39/tic82".e_RAJ2000,  "IV/39/tic82".e_DEJ2000, 
#  "IV/39/tic82".RAOdeg,  "IV/39/tic82".DEOdeg,  "IV/39/tic82".e_RAOdeg,  "IV/39/tic82".e_DEOdeg
#  FROM "IV/39/tic82" WHERE TIC = 231663901

adql_string = '''-- output format : csv
SELECT "IV/39/tic82".TIC,   "IV/39/tic82".GAIA
FROM "IV/39/tic82" WHERE xxxx'''
 
 
tics = toi_table['TIC ID']
adql_file = open('adql.txt', 'w')


tic_str = ''
for i in range(len(tics)):
    if i == len(tics)-1:
        # print(f'(TIC = {tics[i]})' , end=' ')
        tic_str = tic_str + f'(TIC = {tics[i]})'
    else:
        # print(f'(TIC = {tics[i]}) OR' , end=' ')
        tic_str = tic_str + f'(TIC = {tics[i]}) OR'
        
        
print(adql_string.replace('xxxx', tic_str), file=adql_file)
https://tapvizier.u-strasbg.fr/adql/
```


### Get RA DEC

```
def get_ra_dec(name):
    import requests

    get_str = 'https://cds.unistra.fr/cgi-bin/nph-sesame/-oxp/SNVA?'
    results = requests.get(get_str+name)

    for line in results.text.split('\n'):
        if '<jradeg>' in line:
            ra = float(line.split('>')[1].split('<')[0])
        if '<jdedeg>' in line:
            dec = float(line.split('>')[1].split('<')[0])
    return ra, dec

```

### RA DEC 2 EDR3 ID

```
def radec2edr3(ra, dec):
    from astroquery.gaia import Gaia

    ra = ra
    dec = dec

    query = f"SELECT * FROM gaiaedr3.gaia_source WHERE CONTAINS(POINT('ICRS', gaiaedr3.gaia_source.ra, gaiaedr3.gaia_source.dec), CIRCLE('ICRS', {ra}, {dec}, 0.001389))=1;"

    job = Gaia.launch_job(query)
    result = job.get_results()


    return result['source_id'][0]
```

### PyTransit Installation on M1/M2 Macbook/Mac Mini
Note: Also works for https://github.com/hippke/tls;https://johannesbuchner.github.io/UltraNest/example-line.html#
```
git clone https://github.com/hpparvi/PyTransit.git
cd pytransit
conda create -n pytransit python==3.10
conda activate pytransit

conda install numpy scipy matplotlib ipython jupyter pandas sympy nose -y
pip install emcee corner arviz
conda install astropy tqdm -y
conda install -c conda-forge celerite -y
conda install anaconda::pytables -y
conda install scipy pandas -y
conda install conda-forge::pyopencl
conda install conda-forge::semantic_version
conda install ocl_icd_wrapper_apple -y
python setup.py install
```


### Vmic and Vmac (iSpec)

```Python
import numpy as np
def _estimate_vmac_doyle2014(teff, logg, feh):
    """
    Estimate Macroturbulence velocity (Vmac) by using an empirical relation
    considering the effective temperature, surface gravity and metallicity.

    The relation was constructed by Doyle et al. (2014), which is only valid
    for the Teff range 5200 to 6400 K, and the log g range 4.0 to 4.6 dex.
    """
    t0 = 5777
    g0 = 4.44

    if logg >= 3.5:
        if teff >= 5000:
            # main sequence and subgiants (RGB)
            vmac = 3.21 + 2.33e-3*(teff-t0) + 2e-6*(teff-t0)**2 - 2*(logg-g0)
        else:
            # main sequence
            vmac = 3.21 + 2.33e-3*(teff-t0) + 2e-6*(teff-t0)**2 - 2*(logg-g0)
    else:
        # Out of the calibrated limits
        vmac = 0.

    return vmac

def _estimate_vmac_ges(teff, logg, feh):
    """
    Estimate Microturbulence velocity (Vmic) by using an empirical relation
    considering the effective temperature, surface gravity and metallicity.

    The relation was constructed by Maria Bergemann for the Gaia ESO Survey.
    """
    t0 = 5500
    g0 = 4.0

    if logg >= 3.5:
        if teff >= 5000:
            # main sequence and subgiants (RGB)
            vmac = 3*(1.15 + 7e-4*(teff-t0) + 1.2e-6*(teff-t0)**2 - 0.13*(logg-g0) + 0.13*(logg-g0)**2 - 0.37*feh - 0.07*feh**2)
        else:
            # main sequence
            vmac = 3*(1.15 + 2e-4*(teff-t0) + 3.95e-7*(teff-t0)**2 - 0.13*(logg-g0) + 0.13*(logg-g0)**2)
    else:
        # giants (RGB/AGB)
        vmac = 3*(1.15 + 2.2e-5*(teff-t0) - 0.5e-7*(teff-t0)**2 - 0.1*(logg-g0) + 0.04*(logg-g0)**2 - 0.37*feh - 0.07*feh**2)

    return vmac

def estimate_vmac(teff, logg, feh, relation='GES'):
    """
    Estimate Microturbulence velocity (Vmic) by using an empirical relation
    considering the effective temperature, surface gravity and metallicity.

    By default, the selected relation was constructed by Maria Bergemann
    for the Gaia ESO Survey. Alternatively, "relation='Doyle2014'" implements
    a relation for dwrafs (Doyle et al, 2014).
    """
    if relation == 'Doyle2014':
        vmac = _estimate_vmac_doyle2014(teff, logg, feh)
    else:
        vmac = _estimate_vmac_ges(teff, logg, feh)
    vmac = float("%.2f" % vmac)
    return vmac

### vmic
def _estimate_vmic_ges(teff, logg, feh):
    """
    Estimate Microturbulence velocity (Vmic) by using an empirical relation
    considering the effective temperature, surface gravity and metallicity.

    The relation was constructed based on the UVES Gaia ESO Survey iDR1 data,
    results for the benchmark stars (Jofre et al. 2013),
    and globular cluster data from external literature sources.

    Source: http://great.ast.cam.ac.uk/GESwiki/GesWg/GesWg11/Microturbulence
    """
    t0 = 5500
    g0 = 4.0

    if logg >= 3.5:
        if teff >= 5000:
            # main sequence and subgiants (RGB)
            vmic = 1.05 + 2.51e-4*(teff-t0) + 1.5e-7*(teff-t0)**2 - 0.14*(logg-g0) - 0.05e-1*(logg-g0)**2 + 0.05*feh + 0.01*feh**2
        else:
            # main sequence
            vmic = 1.05 + 2.51e-4*(5000-t0) + 1.5e-7*(5000-t0)**2 - 0.14*(logg-g0) - 0.05e-1*(logg-g0)**2 + 0.05*feh + 0.01*feh**2
    else:
        # giants (RGB/AGB)
        vmic = 1.25 + 4.01e-4*(teff-t0) + 3.1e-7*(teff-t0)**2 - 0.14*(logg-g0) - 0.05e-1*(logg-g0)**2 + 0.05*feh + 0.01*feh**2
    vmic = float("%.2f" % vmic)
    return vmic

def _estimate_vmic_Bruntt2010(teff, logg, feh):
    # https://ui.adsabs.harvard.edu/abs/2010MNRAS.405.1907B/abstract
    t0 = 5700
    g0 = 4.0

    if logg < 4 or teff < 5000 or teff > 6500:
        return np.nan

    vmic = 1.01 + 4.5610e-4*(teff-t0) + 2.75e-7*(teff-t0)**2
    
    vmic = float("%.2f" % vmic)
    return vmic

def estimate_vmic(teff, logg, feh, relation='GES'):
    """
    Estimate Microturbulence velocity (Vmic) by using an empirical relation
    considering the effective temperature, surface gravity and metallicity.

    By default, the selected relation was constructed by Maria Bergemann
    for the Gaia ESO Survey. Alternatively, "relation='Doyle2014'" implements
    a relation for dwrafs (Doyle et al, 2014).
    """
    if relation == 'Bruntt2010':
        vmac = _estimate_vmic_Bruntt2010(teff, logg, feh)
    else:
        vmac = _estimate_vmic_ges(teff, logg, feh)
    vmac = float("%.2f" % vmac)
    return vmac


teff = 5500
logg = 4.5
feh = 0.0

#vmac = estimate_vmac(teff, logg, feh)
vmic = estimate_vmic(teff, logg, feh)

#print(vmac,vmic)

vmac = estimate_vmac(teff, logg, feh, relation='Doyle2014')
#vmic = estimate_vmic(teff, logg, feh, relation='Bruntt2010')

print(vmac,vmic)
```




### get RV from ESO data; ESP

```Python
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import interpolate
# x = np.arange(0, 10)
# y = np.exp(-x/3.0)
# f = interpolate.interp1d(x, y)

from astropy.io import fits

# open the FITS file
import glob
import numpy as np
file_list = glob.glob('/Users/wangxianyu/Downloads/Downloads/archive (5)/*.fits')
times = np.asarray([]); rvs = np.asarray([]); rverrs = np.asarray([])
for file in file_list:
    hdul = fits.open(file)
    try:
        time = hdul[0].header["HIERARCH ESO QC BJD"] # IN TDB; see https://openaccess.inaf.it/bitstream/20.500.12386/31285/4/TNG-MAN-HARPN-0005_HARPS-N_DRS_Manual_i1r2.pdf
        rv = hdul[0].header['HIERARCH ESO QC CCF RV']
        rverr = hdul[0].header['HIERARCH ESO QC CCF RV ERROR']
        times = np.append(times, time); rvs = np.append(rvs, rv); rverrs = np.append(rverrs, rverr)
    except:
        pass
import matplotlib.pyplot as plt
plt.errorbar(times, rvs, yerr=rverrs, fmt='o')


```


### find all file path including
```
import pandas as pd
import numpy as np

import os

def find_files(filename, search_path):
    result = []

    for root, dirs, files in os.walk(search_path):
        if filename in files:
            result.append(os.path.join(root, filename))

    return result
filename = "params_star.csv"
search_path = os.getcwd()
```



### q 2 u

```Python

def get_u1u2_from_q1q2(q1, q2, Nsamples=10000):
    '''
    q1, q2: float or list of form [median, lower_err, upper_err]
    '''
    if type(q1)==float and type(q2)==float:
        u1 = 2.*np.sqrt(q1)*q2
        u2 = np.sqrt(q1) * (1. - 2.*q2)
        return u1, u2
    else:
        q1 = spdf(q1[0], q1[1], q1[2], size=Nsamples, plot=False)
        q2 = spdf(q2[0], q2[1], q2[2], size=Nsamples, plot=False)
        ind_good = np.where( (q1>=0) & (q1<=1) & (q2>=0) & (q2<=1) )[0]
        q1 = q1[ind_good]
        q2 = q2[ind_good]
        u1 = 2.*np.sqrt(q1)*q2
        u2 = np.sqrt(q1) * (1. - 2.*q2)
        u1_ll, u1_median, u1_ul = np.percentile(u1, [16,50,84])
        u2_ll, u2_median, u2_ul = np.percentile(u2, [16,50,84])
        return (u1_median, u1_median-u1_ll, u1_ul-u1_median), (u2_median, u2_median-u2_ll, u2_ul-u2_median)
    


def get_q1q2_from_u1u2(u1, u2, Nsamples=10000):
    '''
    u1, u2: float or list of form [median, lower_err, upper_err]
    '''
    if type(u1)==float and type(u2)==float:
        q1 = (u1 + u2)**2
        q2 = 0.5*u1/(u1+u2)
        return q1, q2
    else:
        u1 = spdf(u1[0], u1[1], u1[2], size=Nsamples, plot=True)
        u2 = spdf(u2[0], u2[1], u2[2], size=Nsamples, plot=True)
        ind_good = np.where( (u1>=0) & (u1<=1) & (u2>=0) & (u2<=1) )[0]
        u1 = u1[ind_good]
        u2 = u2[ind_good]
        q1 = (u1 + u2)**2
        q2 = 0.5*u1/(u1+u2)
        q1_ll, q1_median, q1_ul = np.percentile(q1, [16,50,84])
        q2_ll, q2_median, q2_ul = np.percentile(q2, [16,50,84])
        return (q1_median, q1_median-q1_ll, q1_ul-q1_median), (q2_median, q2_median-q2_ll, q2_ul-q2_median)

```



### get RV from ESPRESSO
```Python
# fits
from astropy.io import fits
import numpy as np
import os

path = '/Users/wangxianyu/Library/CloudStorage/OneDrive-SharedLibraries-onedrive/每日科研/2023/20230928_RM_Refit/add_planets/HD209485/archive2'
fit_files = [x for x in os.listdir(path) if x.endswith('.fits')]
rvs = []
rverrs = []
times = []
for i in fit_files:
    hdu = fits.open(path+'/'+i)
    time_bjd_tdb = hdu[0].header['HIERARCH ESO QC BJD']
    RV = hdu[0].header['HIERARCH ESO QC CCF RV']
    RVerr = hdu[0].header['HIERARCH ESO QC CCF RV ERROR']
    hdu.close()
    rvs.append(RV)
    rverrs.append(RVerr)
    times.append(time_bjd_tdb)
    
    
    
times = np.array(times)
rvs = np.array(rvs)
rverrs = np.array(rverrs)
```

### Manim Resources

```
Manim Web: https://manim-web.hugos29.dev/
https://eertmans.be/manim-slides/
https://pypi.org/project/manim-pptx/
https://plugins.manim.community/
```


### Manim, HJs migration
```Python
# https://docs.manim.community/en/stable/installation/jupyter.html#google-colaboratory
# !sudo apt update
# !sudo apt install libcairo2-dev ffmpeg \
#     texlive texlive-latex-extra texlive-fonts-extra \
#     texlive-latex-recommended texlive-science \
#     tipa libpango1.0-dev
# !pip install manim
# !pip install IPython --upgrade


%%manim -qm -v WARNING HeatDiagramPlot

from manim import *
import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt

def smoothstep(x, x_min=0, x_max=1, y_min=0, y_max=1, N=1):
    x = np.clip((x - x_min) / (x_max - x_min), 0, 1)
    
    result = 0
    for n in range(0, N + 1):
         result += comb(N + n, n) * comb(2 * N + 1, N - n) * (-x) ** n

    result *= x ** (N + 1)
    
    # Transforming the output to desired range [y_min, y_max]
    result = y_min + result * (y_max - y_min)
    
    return result


import matplotlib.pyplot as plt


class HeatDiagramPlot(Scene):
    def construct(self):
        ax = Axes(
            x_range=[0.00001, 4.1, 1],
            y_range=[0, 12.1, 1],
            x_length=9,
            y_length=6,
            y_axis_config={"numbers_to_include": np.arange(0, 12, 1)},
            x_axis_config={"scaling": LogBase(custom_labels=True),"numbers_to_include": np.arange(1, 4, 1)},
            tips=False,
        )
        labels = ax.get_axis_labels(
            x_label=Tex("Period (days)"), y_label=Tex("Occurrence rate ($\%$)")
        )

        x_vals = np.linspace(0, 4, 1000)
        y_min = 5
        y_max = 10
        cut = 300
        log_cut = np.log10(300)
        half_width = 0.5
        y_vals = smoothstep(x_vals, x_min=(log_cut - half_width), x_max=(log_cut + half_width), y_min=y_min, y_max=y_max, N=5)
        x_vals = 10**x_vals

        lines = VGroup(
            *[Line(ax.c2p(x_vals[i], y_vals[i]), ax.c2p(x_vals[i+1], y_vals[i+1])) for i in range(len(x_vals)-1)]
        )
        # Creating Random Dots
        dots = VGroup()

        x_randoms = []
        y_randoms = []

        for _ in range(100):  # Creating 50 dots as an example
            x_random = np.random.uniform(0, 4)
            y_random = np.random.uniform(0, smoothstep(x_random, x_min=(log_cut - half_width), x_max=(log_cut + half_width), y_min=y_min, y_max=y_max, N=5))
            x_random = 10**x_random
            dot = Dot(ax.c2p(x_random, y_random), radius=0.04, color=BLUE)
            dots.add(dot)
            x_randoms.append(x_random)
            y_randoms.append(y_random)


        # self.play(Create(ax))
        # self.play(Write(labels))
        # self.play(Create(lines))
        # self.play(Create(dots))


        red_probability = 0.1  # baseline probability to turn red
        red_probability_boost = 0.5  # boosted probability if x > 300

  # [Your existing code above this point ...]

        dots_to_remove = VGroup()
        dots_to_add = VGroup()

        # Changing color and size of some dots
        for i, dot in enumerate(dots):
            x_value = x_randoms[i]
            prob = red_probability_boost if x_value > 300 else red_probability
            if np.random.uniform(0, 1) < prob:  
                dot.set_color(RED).scale(2)
                if x_value > 300:
                    dots_to_remove.add(dot)

        self.play(Create(ax), Write(labels), Create(lines), Create(dots),run_time=10)

        dots_to_remove_animations = []
        dots_to_add_animations = []

        # Prepare animations for removing dots
        for dot in dots_to_remove:
            dots_to_remove_animations.append(dot.animate.scale(0).set_opacity(0))
            dots.remove(dot)  # if you want to actually delete them from the dots VGroup

        num_dots_to_add = len(dots_to_remove)

        # Prepare animations for adding new red dots under the curve at x < 300
        for _ in range(num_dots_to_add):
            x_new = np.random.uniform(0, np.log10(300))  # new x in log scale
            x_new = 10**x_new  # convert back to linear scale
            y_new = np.random.uniform(0, smoothstep(np.log10(x_new), x_min=(log_cut - half_width), x_max=(log_cut + half_width), y_min=y_min, y_max=y_max, N=5))
            dot = Dot(ax.c2p(x_new, y_new), radius=0.08, color=RED)  # bigger red dots
            dots.add(dot)
            dots_to_add.add(dot)
            dots_to_add_animations.append(FadeIn(dot))

        # Play animations simultaneously
        self.play(*dots_to_remove_animations, *dots_to_add_animations, run_time=5)

        self.wait()



```



### better print for pandas 
```Python
for i,ele in enumerate(obliquity_all.columns):
    ending = ' '
    if i%15 == 0 and i>0:
        ending = '\n'
    print(ele, end=ending)
```


### smoothstep function

```Python
import numpy as np
from scipy.special import comb
import matplotlib.pyplot as plt

def smoothstep(x, x_min=0, x_max=1, y_min=0, y_max=1, N=1):
    x = np.clip((x - x_min) / (x_max - x_min), 0, 1)
    
    result = 0
    for n in range(0, N + 1):
         result += comb(N + n, n) * comb(2 * N + 1, N - n) * (-x) ** n

    result *= x ** (N + 1)
    
    # Transforming the output to desired range [y_min, y_max]
    result = y_min + result * (y_max - y_min)
    
    return result

# Example usage
x = np.linspace(-2, 2, 1000)
y_min, y_max = 10, 15

plt.figure(figsize=(10, 6))
for N in range(4, 5):
    y = smoothstep(x, x_min=-1, x_max=1, y_min=y_min, y_max=y_max, N=N)
    plt.plot(x, y, label=f"N={N}")

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.title("Smoothstep function with varying N")
plt.show()
```



### merge pstable to get all parameters

```Python
import pandas as pd
import numpy as np

df = pstable
unique_pl_names = np.unique(df['pl_name'])
def select_best_data(group):
    default_index = group[group['default_flag'] == 1].index[0]
    default_row = group.loc[default_index].copy()
    
    for col in group.columns:
        if pd.isnull(default_row[col]):
            non_default_rows = group.loc[group.index != default_index]
            non_nan_values = non_default_rows[~pd.isnull(non_default_rows[col])]
            if not non_nan_values.empty:
                best_alternate_value = non_nan_values.sort_values(by='pl_pubdate', ascending=False)[col].iloc[0]
                default_row[col] = best_alternate_value
    return default_row


results = pd.DataFrame()
for i in range(len(unique_pl_names)):
    print(len(unique_pl_names), i, unique_pl_names[i],  end='\r')    
    tmp_pstable = df.query('pl_name == "'+unique_pl_names[i]+'"')
    result = tmp_pstable.groupby('pl_name').apply(select_best_data).reset_index(drop=True)
    results = pd.concat([results, result], ignore_index=True)
results.to_csv('pstable_merge.csv', index=False)
```



### get stellar parameters from 175 million
```Python
import numpy as np
dr3s = tepcat_table['gaiadr3_id'].values
tic = dr3s.astype(str)
import pandas as pd

filename = '/Users/wangxianyu/Downloads/Program/175millionStellarPar/table_1_catwise.csv.gz'
chunk_size = 50000  # Adjust depending on your available memory
chunks = pd.read_csv(filename, compression='gzip', chunksize=chunk_size)

# Assuming you have a list of target IDs
target_ids = dr3s  # Example list
target_ids = [int(t) for t in target_ids]
target_ids
results = {}  # Store results for each target_id

for chunk in chunks:
    # Check for each target_id
    for tid in target_ids:
        if tid in results:  # If already found, skip
            continue
        filtered_chunk = chunk[chunk['source_id'] == tid]
        if not filtered_chunk.empty:
            mh_xgboost = filtered_chunk['mh_xgboost'].values[0]
            teff_xgboost = filtered_chunk['teff_xgboost'].values[0]
            logg_xgboost = filtered_chunk['logg_xgboost'].values[0]
            results[tid] = (mh_xgboost, teff_xgboost, logg_xgboost)
            print(f'For target_id {tid}: XGBoost: MH={mh_xgboost:.3f}, Teff={teff_xgboost:.1f}, logg={logg_xgboost:.2f}')
            # print
    
    # If all target_ids are found, break
    if set(target_ids).issubset(set(results.keys())):
        break

# All results are stored in 'results' dictionary
for tid, (mh, teff, logg) in results.items():
    # print(f'For target_id {tid}: XGBoost: MH={mh:.3f}, Teff={teff:.1f}, logg={logg:.2f}')
    print(tid, mh, teff, logg, sep=',')
fehs = []
teffs = []
loggs = []


for i in range(len(tepcat_table)):
    name = tepcat_table['name'][i]
    gaiadr2_id = tepcat_table['gaiadr2_id'][i]
    gaiadr3_id = tepcat_table['gaiadr3_id'][i]
    # print('Working on '+name)
    try:
        mh, teff, logg = results[gaiadr3_id]
        fehs.append(mh)
        teffs.append(teff)
        loggs.append(logg)
    except:
        fehs.append(np.nan)
        teffs.append(np.nan)
        loggs.append(np.nan)
        print(name, 'not found')
        pass

```


### get stellar rho from period and a/r


```Python
rho = 3 * pi * (a/R)**3 / (G * P**2)
```


### CDPP snr calculation

```Python
# Note that: this is incorrect. But Xian-Yu confirmed that the paper's calculation is correct.
# And the results agree with TOI catalog. Need time to fix. 2024 Nov 10 03:27 PM
import pandas as pd


# constants from https://iopscience.iop.org/article/10.3847/1538-3881/ac68e3/pdf
c0 = 50.2/1e6
c1 = 97.4/1e6
c2 = 92.9/1e6


# import target list
target_list = pd.read_csv('https://raw.githubusercontent.com/transit-timing/tt/master/3_database/1_target_list.csv')
System = target_list['System']

duration = target_list['duration']
R_b_over_R_A_squared = target_list['R_b_over_R_A_squared']
TESSMag = target_list['TESSMag']
SNR = target_list['SNR']

headers = ['System','My SNR','My SNR * 2','TT SNR','duration','TESSMag']
data = []
for i in range(len(System)):
    cdpp = c0 + c1*10**(0.2*(TESSMag[i]-10)) + c2*10**(0.4*(TESSMag[i]-10))
    snr = R_b_over_R_A_squared[i]/(cdpp/np.sqrt(duration[i]*24))
    data = np.append(data, [System[i],snr, snr*2,SNR[i],duration[i]*24,TESSMag[i]])
# to pandas
data = data.reshape(-1, len(headers))
df = pd.DataFrame(data, columns=headers)
df
```




### sub sub spec grid

```Python
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(10, 8))
grid_spec = gridspec.GridSpec(2, 2, figure=fig)

subgrid_spec = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=grid_spec[0, 0], hspace=0, height_ratios=[2, 1])
ax1 = plt.subplot(subgrid_spec[0, 0])
ax2 = plt.subplot(subgrid_spec[1, 0], sharex=ax1)
ax1.set_xticklabels([])


subgrid_spec = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=grid_spec[0, 1], hspace=0, height_ratios=[2, 1])
ax1 = plt.subplot(subgrid_spec[0, 0])
ax2 = plt.subplot(subgrid_spec[1, 0], sharex=ax1)
ax1.set_xticklabels([])




subgrid_spec = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=grid_spec[1, 0], hspace=0, height_ratios=[2, 1])
ax1 = plt.subplot(subgrid_spec[0, 0])
ax2 = plt.subplot(subgrid_spec[1, 0], sharex=ax1)
ax1.set_xticklabels([])


subgrid_spec = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=grid_spec[1, 1], hspace=0, height_ratios=[2, 1])
ax1 = plt.subplot(subgrid_spec[0, 0])
ax2 = plt.subplot(subgrid_spec[1, 0], sharex=ax1)
ax1.set_xticklabels([])


plt.show()


```



### get dilution factor for TESS and TESS-SPOC

```Python


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import lightkurve as lk
name = 'WASP-148'
import lightkurve as lk
lcs= lk.search_lightcurvefile(name, mission='TESS')



lc = lcs[3].download()
lc.meta['CROWDSAP'],lcs
```


### get Av from Bayestar

```Python

# Example usage
star_name = "K2-232"
ra, dec, parallax, parallax_error = query_star_data(star_name)



from astropy.coordinates import SkyCoord
import astropy.units as u

ra_deg = ra * u.deg
dec_deg = dec * u.deg
parallax_mas = parallax * u.mas
distance = parallax_mas.to(u.pc, equivalencies=u.parallax())
coords = SkyCoord(ra=ra_deg, dec=dec_deg, distance=distance)
bayestar = BayestarQuery(max_samples=1)
Av_bayestar = 2.742 * bayestar(coords)
Av_bayestar
```


### name to ra, dec, parallax

```Python
from astroquery.simbad import Simbad
from astroquery.gaia import Gaia

def query_star_data(star_name):
    # Initialize Simbad and add Gaia DR3 to the fields
    Simbad.add_votable_fields('ids')
    
    # Query Simbad by star name to get Gaia DR3 ID
    result = Simbad.query_object(star_name)
    ids = result['IDS'][0].split('|')
    gaia_dr3_id = next((id for id in ids if 'Gaia DR3' in id), None)
    
    if not gaia_dr3_id:
        print(f"No Gaia DR3 ID found for star {star_name}.")
        return
    
    # Query the Gaia DR3 archive using the retrieved ID
    query = f"""
    SELECT ra, dec, parallax, parallax_error 
    FROM gaiaedr3.gaia_source 
    WHERE source_id = {gaia_dr3_id.split()[-1]}
    """

    job = Gaia.launch_job(query)
    result = job.get_results()

    # Extract and return the values
    ra = result['ra'][0]
    dec = result['dec'][0]
    parallax = result['parallax'][0]
    parallax_error = result['parallax_error'][0]

    return ra, dec, parallax, parallax_error

# Example usage
star_name = "K2-232"
ra, dec, parallax, parallax_error = query_star_data(star_name)
print(f"RA: {ra} deg")
print(f"Dec: {dec} deg")
print(f"Parallax: {parallax} mas")
print(f"Parallax Error: {parallax_error} mas")

```


### RV Completeness Contours

```Python
# https://california-planet-search.github.io/rvsearch/tutorials/Completeness_Contours.html
import os

import numpy as np
import pylab as pl

import rvsearch
from rvsearch.inject import Completeness
from rvsearch.plots import CompletenessPlots

recfile = os.path.join(rvsearch.DATADIR, 'recoveries.csv')

comp = Completeness.from_csv(recfile, 'inj_au', 'inj_msini', mstar=1.1)
xi, yi, zi = comp.completeness_grid(xlim=(0.05, 100), ylim=(1.0, 3e4), resolution=25)

cp = CompletenessPlots(comp)
fig = cp.completeness_plot(xlabel='$a$ [AU]', ylabel=r'M$\sin{i}$ [M$_{\oplus}$]')

```



### find boundary and calc the difference
```Python

import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)
teff = np.random.normal(6200, 100, 100)
lambda_ = np.random.uniform(0, 180, 100) * (teff > 6300)
lambda_ += np.random.normal(0, 8, 100)* (teff <= 6300)

with pm.Model() as model:
    # t = pm.Normal('t', mu=6500., sigma=250.)
    t = pm.Uniform('t', lower=teff.min(), upper=teff.max(), testval=6250.)
    
    mu_high = pm.Normal('mu_high', mu=0., sigma=10.)
    sigma_high = pm.HalfNormal('sigma_high', sigma=10.)
    mu_low = pm.Normal('mu_low', mu=0., sigma=10.)
    sigma_low = pm.HalfNormal('sigma_low', sigma=10.)

    weight = pm.math.sigmoid(teff - t)

    mu_ = weight * mu_high + (1 - weight) * mu_low
    sigma_ = weight * sigma_high + (1 - weight) * sigma_low

    likelihood = pm.Normal('likelihood', mu=mu_, sigma=sigma_, observed=lambda_)

    testval = pm.find_MAP(model=model)
    trace = pm.sample(tune=4000, draws=4000, chains=2, cores=2, random_seed=0, return_inferencedata=True, target_accept=0.999, start=testval)

az.style.use('arviz-darkgrid')
az.plot_trace(trace, var_names=['t', 'mu_high', 'sigma_high', 'mu_low', 'sigma_low'])
mu_high = trace.posterior['mu_high'].values.flatten()
mu_low = trace.posterior['mu_low'].values.flatten()

median1 = np.median(mu_high)
median2 = np.median(mu_low)

err = np.std(mu_high)
err2 = np.std(mu_low)

agreements = abs(median1 - median2) / np.sqrt(err**2 + err2**2)
agreements
```




### make fits
```Python
def mkfits(time, flux, flux_err, path):
    time = np.round(time, 7); flux = np.round(flux, 7); flux_err = np.round(flux_err, 7)
    col1 = fits.Column(name='TIME', array=time, format='D')
    col2 = fits.Column(name='FLUX', array=flux, format='D')
    col3 = fits.Column(name='FLUX_ERR', array=flux_err, format='D')
    cols = fits.ColDefs([col1, col2, col3])
    hdu = fits.BinTableHDU.from_columns(cols)
    hdu.writeto(path, overwrite=True)
    
```


### tic to dr2 to dr3 / get Gaia stellar pars


```python
from astroquery.vizier import Vizier
import astropy.units as u
import numpy as np
from astroquery.gaia import Gaia

def get_gaia_dr2_id(tic_num):
    # Configure Vizier to retrieve all columns from the TIC catalog
    v = Vizier(columns=["**"], catalog=["IV/39/tic82"])

    # Query the catalog for the object with the given TIC number
    result = v.query_object("TIC "+str(tic_num), radius=0.0001*u.deg)

    # Get the table of results
    tic_table = result[0]

    # Find the row with the closest TIC number
    idx_tic = np.argmin(np.abs(tic_table['TIC'] - tic_num))
    qtic = tic_table[idx_tic]

    # Extract and return the Gaia DR2 ID
    gaia_dr2_id = qtic['GAIA']
    return gaia_dr2_id


from astroquery.gaia import Gaia

def dr2_to_dr3(dr2_id):
    # Define the ADQL query
    query = f"SELECT * FROM gaiadr3.dr2_neighbourhood WHERE dr2_source_id={dr2_id}"

    # Execute the query
    job = Gaia.launch_job_async(query)
    result = job.get_results()

    # Extract and return the Gaia DR3 ID
    # Note: This assumes that there is only one matching entry in the Gaia DR3 catalog.
    # If there might be multiple matches, you would need to add some logic to handle that.
    dr3_id = result['dr3_source_id'][0]
    return dr3_id


def get_star_parameters(gaia_dr3_id):
    # Run the query
    job = Gaia.launch_job_async("SELECT * FROM gaiadr3.astrophysical_parameters WHERE source_id={}".format(gaia_dr3_id))
    result = job.get_results()
    # Return the results
    return result

tic_num = 441739020  # Replace with your TIC number
gaia_dr2_id = get_gaia_dr2_id(tic_num)
dr3_id = dr2_to_dr3(gaia_dr2_id)
star_parameters = get_star_parameters(dr3_id)  
mass_flame = star_parameters['mass_flame'][0];mass_flame_lower = star_parameters['mass_flame_lower'][0];mass_flame_upper = star_parameters['mass_flame_upper'][0]
mass_flame_uerr, mass_flame_lerr = mass_flame_upper - mass_flame, mass_flame - mass_flame_lower
radius_flame = star_parameters['radius_flame'][0];radius_flame_lower = star_parameters['radius_flame_lower'][0];radius_flame_upper = star_parameters['radius_flame_upper'][0]
radius_flame_uerr, radius_flame_lerr = radius_flame_upper - radius_flame, radius_flame - radius_flame_lower
age_flame = star_parameters['age_flame'][0];age_flame_lower = star_parameters['age_flame_lower'][0];age_flame_upper = star_parameters['age_flame_upper'][0]
age_flame_uerr, age_flame_lerr = age_flame_upper - age_flame, age_flame - age_flame_lower
teff_gspspec = star_parameters['teff_gspspec'][0];teff_gspspec_lower = star_parameters['teff_gspspec_lower'][0];teff_gspspec_upper = star_parameters['teff_gspspec_upper'][0]
teff_gspspec_uerr, teff_gspspec_lerr = teff_gspspec_upper - teff_gspspec, teff_gspspec - teff_gspspec_lower
logg_gspspec = star_parameters['logg_gspspec'][0];logg_gspspec_lower = star_parameters['logg_gspspec_lower'][0];logg_gspspec_upper = star_parameters['logg_gspspec_upper'][0]
logg_gspspec_uerr, logg_gspspec_lerr = logg_gspspec_upper - logg_gspspec, logg_gspspec - logg_gspspec_lower
mh_gspspec = star_parameters['mh_gspspec'][0];mh_gspspec_lower = star_parameters['mh_gspspec_lower'][0];mh_gspspec_upper = star_parameters['mh_gspspec_upper'][0]
mh_gspspec_uerr, mh_gspspec_lerr = mh_gspspec_upper - mh_gspspec, mh_gspspec - mh_gspspec_lower
print(mh_gspspec, mh_gspspec_uerr, mh_gspspec_lerr)




```



### get files with certain suffix

```Python
import os
import re

def find_files(path, pattern):
    files_found = []
    
    for root, dirs, files in os.walk(path):
        for file in files:
            if re.search(pattern, file):
                file_path = os.path.join(root, file)
                files_found.append(file_path)
                
    return files_found

path = "."  # Use your path
pattern = "TOI-2025.*\.csv$"  # Use your regex
print(find_files(path, pattern))


```


### get Filtered WJs candidates (Need to mark confirmed manually)

```Python
import pandas as pd
 
toi = pd.read_csv(dated_name, comment='#')

# toi = toi.query(' TESS_Mag < 13.5')

toi = toi[toi['TESS Mag'] < 13.5]
toi = toi[toi['Planet Radius (R_Earth)'] > 11.208981*0.8]



st_mass = toi['Stellar Mass (M_Sun)'].values
period = toi['Period (days)'].values




period = period*u.day
st_mass = st_mass*u.Msun

a = (period**2 * c.G * st_mass / (4*np.pi**2))**(1/3)
a = a.to(u.AU)
a_r = a/(toi['Stellar Radius (R_Sun)'].values*u.Rsun)
a_r = a_r.to('')
toi = toi[a_r>12]

np.unique(toi['TESS Disposition'].values,return_counts=True)


# toi = toi[(toi['TFOPWG Disposition'] == 'CP') or (toi['TFOPWG Disposition'] == 'PC')]
toi = toi[(toi['TFOPWG Disposition'] == 'CP' ) | (toi['TFOPWG Disposition'] == 'PC') | (toi['TFOPWG Disposition'] == 'APC')]
toi = toi[(toi['TESS Disposition'] == 'CP' ) | (toi['TESS Disposition'] == 'PC') ]

toi = toi[toi['Period (days)']<100]



# if Comments has EB and V-shape; remove
idx = []
for i in range(len(toi)):
    # if 'eb' in  str(toi['Comments'].values[i]).lower() or 'v-shape' in str(toi['Comments'].values[i]).lower() or 'found in faint-star' in str(toi['Comments'].values[i]).lower():
    if 'eb' in  str(toi['Comments'].values[i]).lower() or 'sb' in  str(toi['Comments'].values[i]).lower() or 'v-shape' in str(toi['Comments'].values[i]).lower():
        pass
    else:
        # print(i)
        idx.append(i)
        pass
    
    
toi = toi.iloc[idx]

toi.to_csv('toi.csv',index=False)


```


### get K2

```Python
import astropy.io.fits as fits
from astropy.io import fits
import numpy as np
# K2-140
# K2-232
# K2-25
# K2-29
# K2-290
# K2-34
# K2-93

name = 'K2-93'
lcs = lk.search_lightcurve(name, cadence='long', mission='K2', author='EVEREST')
for lc in lcs:
    try:
        mission = lc.mission[0].strip('K2 Campaign ')
        lc = lc.download()
        time = lc.time.value; flux = lc.flux.value; flux_err = lc.flux_err.value
        time = 2454833.0 + time
        datett = np.round(time, 7); flux = np.round(flux, 7); flux_err = np.round(flux_err, 7)
        col1 = fits.Column(name='TIME', array=datett, format='D')
        col2 = fits.Column(name='FLUX', array=flux, format='D')
        col3 = fits.Column(name='FLUX_ERR', array=flux_err, format='D')
        cols = fits.ColDefs([col1, col2, col3])
        hdu = fits.BinTableHDU.from_columns(cols)
        hdu.writeto(name+'_'+str(mission)+'.fits', overwrite=True)
        print(name+'_'+str(mission)+'.fits saved (long)')
    except:
        pass
lcs = lk.search_lightcurve(name, cadence='short', mission='K2')
unique_mission = np.unique(lcs.table['mission'])
for mission in unique_mission:
    try:
        lc_cl = lcs[lcs.table['mission'] == mission]
        lc = lc_cl.download_all()
        times = np.asarray([]); fluxes = np.asarray([]); flux_errs = np.asarray([])
        for i,ind_lc in enumerate(lc):
            mission = lc_cl[i].mission[0].strip('K2 Campaign ')
            time = ind_lc.time.value; flux = ind_lc.flux.value; flux_err = ind_lc.flux_err.value
            times = np.append(times, time); fluxes = np.append(fluxes, flux); flux_errs = np.append(flux_errs, flux_err)
        times = 2454833.0 + times
        datett = np.round(times, 7); flux = np.round(fluxes, 7); flux_err = np.round(flux_err, 7)
        col1 = fits.Column(name='TIME', array=datett, format='D')
        col2 = fits.Column(name='FLUX', array=flux, format='D')
        col3 = fits.Column(name='FLUX_ERR', array=flux_err, format='D')
        cols = fits.ColDefs([col1, col2, col3])
        hdu = fits.BinTableHDU.from_columns(cols)
        hdu.writeto(name+'_'+str(mission)+'.fits', overwrite=True)
        print(name+'_'+str(mission)+'.fits saved (short)')
    except:
        pass
```




### get Kepler

```Python
import astropy.io.fits as fits
from astropy.io import fits
import numpy as np
name = 'Kepler-420'
lcs = lk.search_lightcurve(name, cadence='long', mission='Kepler')

for lc in lcs:
    try:
        mission = lc.mission[0].strip('Kepler Quarter ')
        lc = lc.download()
        time = lc.time.value; flux = lc.flux.value; flux_err = lc.flux_err.value
        time = 2454833.0 + time
        datett = np.round(time, 7); flux = np.round(flux, 7); flux_err = np.round(flux_err, 7)
        col1 = fits.Column(name='TIME', array=datett, format='D')
        col2 = fits.Column(name='FLUX', array=flux, format='D')
        col3 = fits.Column(name='FLUX_ERR', array=flux_err, format='D')
        cols = fits.ColDefs([col1, col2, col3])
        hdu = fits.BinTableHDU.from_columns(cols)
        hdu.writeto(name+'_'+str(mission)+'.fits', overwrite=True)
        print(name+'_'+str(mission)+'.fits saved (long)')
    except:
        pass
lcs = lk.search_lightcurve(name, cadence='short', mission='Kepler')
unique_mission = np.unique(lcs.table['mission'])
for mission in unique_mission:
    try:
        lc_cl = lcs[lcs.table['mission'] == mission]
        lc = lc_cl.download_all()
        times = np.asarray([]); fluxes = np.asarray([]); flux_errs = np.asarray([])
        for i,ind_lc in enumerate(lc):
            mission = lc_cl[i].mission[0].strip('Kepler Quarter ')
            time = ind_lc.time.value; flux = ind_lc.flux.value; flux_err = ind_lc.flux_err.value
            times = np.append(times, time); fluxes = np.append(fluxes, flux); flux_errs = np.append(flux_errs, flux_err)
        times = 2454833.0 + times
        datett = np.round(times, 7); flux = np.round(fluxes, 7); flux_err = np.round(flux_err, 7)
        col1 = fits.Column(name='TIME', array=datett, format='D')
        col2 = fits.Column(name='FLUX', array=flux, format='D')
        col3 = fits.Column(name='FLUX_ERR', array=flux_err, format='D')
        cols = fits.ColDefs([col1, col2, col3])
        hdu = fits.BinTableHDU.from_columns(cols)
        hdu.writeto(name+'_'+str(mission)+'.fits', overwrite=True)
        print(name+'_'+str(mission)+'.fits saved (short)')
    except:
        pass
```


### get CoRoT :C

```Python
# http://idoc-corot.ias.u-psud.fr/sitools/client-user/COROT_N2_PUBLIC_DATA/project-index.html
# HJD
import astropy.io.fits as fits
from astropy.io import fits
import numpy as np

data = fits.open('/Users/wangxianyu/Downloads/Downloads/EN2_STAR_CHR_0315211361_20100305T001525_20100329T065610.fits')
datett = data[1].data['DATETT']
flux = data[1].data['WHITEFLUX']
datett = data[1].data['DATEBARTT']+2400000
flux = data[1].data['WHITEFLUX']/np.median(data[1].data['WHITEFLUX'])
flux_err =  0.003+0*flux


idx = (flux > 0.95) & (flux < 1.05)
datett = datett[idx]; flux = flux[idx]; flux_err = flux_err[idx]

datett = np.round(datett, 7); flux = np.round(flux, 7); flux_err = np.round(flux_err, 7)

col1 = fits.Column(name='TIME', array=datett, format='D')
col2 = fits.Column(name='FLUX', array=flux, format='D')
col3 = fits.Column(name='FLUX_ERR', array=flux_err, format='D')
cols = fits.ColDefs([col1, col2, col3])
hdu = fits.BinTableHDU.from_columns(cols)
hdu.writeto('CoRoT-18_6.fits', overwrite=True)
plt.scatter(datett, flux/np.median(flux), s=0.1)
plt.axvline(2455260.925, color='r')
plt.xlim(datett[0], datett[0]+1)
```


### Get RM from PyAstronomy


```Python
# Import some unrelated modules
from numpy import arange, pi, random
import matplotlib.pylab as plt
import astropy.constants as const
import astropy.units as u
from PyAstronomy import modelSuite as ms
rmcl = ms.RmcL_Hirano()



def get_rm(time,rr,ar,period,t0,inc,st_rad,vsini,u1,u2):
    t0 = t0-period/2
    st_r_m = st_rad*(const.R_sun.to(u.m)).value
    Omega = -vsini*1000/st_r_m
    rmcl.assignValue({"a": ar, "lambda": lambda_r, 
                      "P": period, "T0": t0, "i": inc,
                      "Is": 90/180*np.pi  , "Omega": Omega, "gamma": rr, "linLimb": u1, "quadLimb": u2, "vbeta": 0, "vSurf": vsini})
    rv = rmcl.evaluate(time)
    return rv*st_r_m/1000


a_r = 16.7
lambda_r = 0/180.0*pi
period = 9.457
t0 = 0
inc = 87.8/180.*pi
st_rad = 1.0  
rr = 0.05  
u1 = 0.3 
u2 = 0.2  
vsini = 9.5 

time = np.linspace(-1, 2, 10000)


rm = get_rm(time,rr,a_r,period,t0,inc,st_rad,vsini,u1,u2)

plt.plot(time, rm)
plt.xlim(-0.2, 0.2)

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
### get stellar inc (pymc version)

```Python
# https://www.pnas.org/doi/pdf/10.1073/pnas.2017418118 A backward-spinning star with two coplanar planets
import pymc as pm
import numpy as np
import pytensor.tensor as tt


sol_radius_km = 696340  
day_to_sec = 24 * 60 * 60 
R_obs, Rerr_obs, Prot_obs, Prot_err, vsini_obs, vsini_err = 1.338, 0.038, 9.34, 0.97, 10.2, 0.64


R_obs, Rerr_obs = R_obs * sol_radius_km, Rerr_obs * sol_radius_km
Prot_obs, Prot_err = Prot_obs * day_to_sec, Prot_err * day_to_sec


with pm.Model() as model:
    R = pm.Normal('R', mu=R_obs, sigma=Rerr_obs) 
    Prot = pm.Normal('Prot', mu=Prot_obs, sigma=Prot_err)
    inc_rad = pm.Uniform('inc_rad', lower=0, upper=np.pi)  # inc in radians
    cos_i = pm.Deterministic('cos_i', tt.cos(inc_rad))
    
    u = pm.Deterministic('u', tt.sqrt(1 - cos_i**2))
    sin_i_squared = pm.Deterministic('sin_i_squared', 1 - cos_i**2)
    sin_i = pm.Deterministic('sin_i',tt.sqrt(sin_i_squared))
    
    inc = pm.Deterministic('inc', inc_rad * (180 / np.pi))  # convert radians to degrees

    v = 2*np.pi*R/Prot
    vu = v * u

    pm.Normal('likelihood_R', mu=R, sigma=Rerr_obs, observed=R_obs)
    pm.Normal('likelihood_Prot', mu=Prot, sigma=Prot_err, observed=Prot_obs)
    pm.Normal('likelihood_vsini', mu=vu, sigma=vsini_err, observed=vsini_obs)
    trace = pm.sample(2000, tune=2000, chains=2, cores=2, target_accept=0.999)


import arviz as az
az.style.use('arviz-darkgrid')
az.plot_trace(trace, var_names=['R', 'Prot', 'cos_i', 'sin_i_squared', 'sin_i', 'inc'])

pm.summary(trace, var_names=['R', 'Prot', 'cos_i', 'sin_i_squared', 'sin_i', 'inc'])





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
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
import matplotlib
matplotlib.rcParams['mathtext.fontset'] = 'custom'
matplotlib.rcParams['mathtext.rm'] = 'Times New Roman'
matplotlib.rcParams['mathtext.it'] = 'Times New Roman:italic'

matplotlib.rc('text.latex', preamble=r'\usepackage{sfmath}')
from matplotlib.ticker import MultipleLocator, \
    FormatStrFormatter, AutoMinorLocator
def ticksetax(ax, labelsize=15, ticksize=12, tickwidth=1.5, ticklength=5):
    linewidth = 1.5
    ax.tick_params(direction='in', which='both', width=linewidth,colors='k', bottom='True',top='True', left='True', right='True', labelsize=15)
    ax.spines['bottom'].set_linewidth(linewidth)
    ax.spines['left'].set_linewidth(linewidth)
    ax.spines['right'].set_linewidth(linewidth)
    ax.spines['top'].set_linewidth(linewidth)
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
import pandas as pd
# download the obliquity data from the website

name = 'toi.csv'
date = time.strftime("%Y-%m-%d", time.localtime())
dated_name = 'toi_'+date+'.csv'
# check if the file exists
if os.path.exists(dated_name):
    print(dated_name+' exists')
else:
    target_url = 'https://exofop.ipac.caltech.edu/tess/download_toi.php?sort=toi&output=csv'
    response = requests.get(target_url)
    data = response.text
    print('downloading '+dated_name)
    with open(dated_name, 'w') as f:
        print(data, file=f)

    
import requests
# download the obliquity data from the website


name = 'pstable.csv'
date = time.strftime("%Y-%m-%d", time.localtime())
dated_name = 'pstable_'+date+'.csv'
# check if the file exists
if os.path.exists(dated_name):
    print(dated_name+' exists')
else:
    target_url = 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+ps&format=csv'
    response = requests.get(target_url)
    data = response.text
    print('downloading '+dated_name)
    with open(dated_name, 'w') as f:
        print(data, file=f)


name = 'pscomppars.csv'
date = time.strftime("%Y-%m-%d", time.localtime())
dated_name = 'pscomppars_'+date+'.csv'
# check if the file exists
if os.path.exists(dated_name):
    print(dated_name+' exists')
else:
    target_url = 'https://exoplanetarchive.ipac.caltech.edu/TAP/sync?query=select+*+from+pscomppars&format=csv'
    response = requests.get(target_url)
    data = response.text
    print('downloading '+dated_name)
    with open(dated_name, 'w') as f:
        print(data, file=f)

# download the obliquity data from the website


name = 'obliquity.csv'
date = time.strftime("%Y-%m-%d", time.localtime())
dated_name = 'obliquity_'+date+'.csv'
# check if the file exists
if os.path.exists(dated_name):
    print(dated_name+' exists')
else:
    target_url = 'https://www.astro.keele.ac.uk/jkt/tepcat/obliquity.csv'
    response = requests.get(target_url)
    data = response.text
    print('downloading '+dated_name)
    with open(dated_name, 'w') as f:
        print(data, file=f)
	
	
toi_table = pd.read_csv('toi_'+date+'.csv', comment='#')
pstable = pd.read_csv('pstable_'+date+'.csv', comment='#')
obliquity_table = pd.read_csv('obliquity_'+date+'.csv', comment='#')
pscomppars_table = pd.read_csv('pscomppars_'+date+'.csv', comment='#')
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
import pandas as pd
import os
import numpy as np

work_folder = '.'

priors_table = pd.read_csv(os.path.join(work_folder, 'params.csv'))
priors_table = priors_table[~priors_table['#name'].str.startswith('#')]
mcmc_save_table = pd.read_csv(os.path.join(work_folder, 'results/mcmc_table.csv'))
mcmc_save_table = mcmc_save_table[~mcmc_save_table['#name'].str.startswith('#')]
mcmc_name_list = mcmc_save_table['#name'].to_list()
priors_columns_list = priors_table.columns.to_list()
valid_names = priors_table.loc[~priors_table['value'].isna(), '#name'].to_list()

for i, valid_name in enumerate(valid_names):
    mcmc_value = mcmc_save_table.loc[mcmc_save_table['#name'] == valid_name, 'median'].values[0]
    priors_table.loc[priors_table['#name'] == valid_name, 'value'] = mcmc_value
    
# make a copy of the old priors file
import time
time_str = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
old_priors_file = os.path.join(work_folder, 'params.csv')
new_priors_file = os.path.join(work_folder, 'params_'+time_str+'.csv')
os.system('cp '+old_priors_file+' '+new_priors_file)
# make a new priors file
priors_table.to_csv(old_priors_file, index=False)
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

