### Mutual inclination

$\cos i_{\mathrm{bc}}=\cos i_{\mathrm{b}} \cos i_{\mathrm{c}}+\sin i_{\mathrm{b}} \sin i_{\mathrm{c}} \cos \left(\Omega_{\mathrm{b}}-\Omega_{\mathrm{c}}\right)$.   

```
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

```
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
```
import astropy.constants as c
import astropy.units as u
period = 12.727272727*u.d
st_rad = 0.77*c.R_sun
vrot = 2*3.1415926535/period*st_rad
print(vrot.to('km/s'))
```

### Calculate a/Rs

```
import astropy.constants as c
import astropy.units as u
period = 10*u.d
st_mass = 1*c.M_sun
st_rad = 1*c.R_sun
R = (c.G*st_mass*(period)**2/4/3.1415926**2 )**(1/3)
print((R/st_rad).to(''))
```
### Calculate RM Amplitude

```
pl_ror = 0.1
vsini = 10
b = 0
RM_amp = 2/3*pl_ror**2*vsini*1000*(1-b**2)**0.5
print(RM_amp,'m/s')
```
### format axis

```
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


```
import matplotlib.pyplot as plt
from scipy import interpolate

def get_interpolate_data(x, y, newx):
    f = interpolate.interp1d(x, y, fill_value="extrapolate", kind='quadratic')
    return f(newx)
```

### multi plot

```
import matplotlib.gridspec as gridspec
plt.figure(figsize=(12, 5))
nrow = 2
ncol = 1
gs = gridspec.GridSpec(nrow, ncol, width_ratios=None, height_ratios=(2,1))
plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0,hspace=0)
ax = plt.subplot(gs[0])

```
