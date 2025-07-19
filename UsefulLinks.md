### MR ploter
- https://github.com/castro-gzlz/mr-plotter

### IDL download page
- https://portal.nv5geospatialsoftware.com/

###  Transiting BDs list
- https://arxiv.org/pdf/2502.19940
### ETCs
- https://www.astro.physik.uni-goettingen.de/research/rvprecision/
- https://www.eso.org/observing/etc/bin/gen/form?INS.NAME=ESPRESSO+INS.MODE=spectro
- https://neid-etc.tuc.noirlab.edu/calc_shell/calculate_rv
- https://etc.eso.org/
- https://research.iac.es/OOCC/observing-tools/exposure-time-calculators/
- HARPS-N official: https://www.astro.unige.ch/~buchschn/

### Stellar Inclination (Masuda & Winn 2020)
- https://github.com/mjfields/cosi
- https://github.com/emilknudstrup/coPsi


### Stellar Rotation
- https://github.com/zclaytor/prot
- https://github.com/RuthAngus/starspot
- https://github.com/ramstojh/kanchay?tab=readme-ov-file
- https://github.com/rae-holcomb/SpinSpotter

### Poster Galleries
- TESS Sci Con II: https://zenodo.org/communities/tsc2/records?q=&l=list&p=1&s=10&sort=newest
- https://exoplanets5.org/posterspreview/
- https://www.astrobetter.com/wiki/Presentation%2bSkills

### SED fit
- VOSA: http://svo2.cab.inta-csic.es/theory/vosa/
- isoclassify: https://github.com/danxhuber/isoclassify/tree/master/isoclassify
- isochrones: https://github.com/timothydmorton/isochrones 
- MINESweeper: https://github.com/pacargile/MINESweeper
- ARIADNE: https://github.com/jvines/astroARIADNE/tree/master/astroARIADNE
- EXOFASTv2: https://github.com/jdeast/EXOFASTv2
- 
### Transmission Spectrum fit
- https://github.com/esedagha/molecfit_lecture/tree/main

### R-M fitting tools
- EXOSAM, https://github.com/baddison2005/ExOSAM/tree/master/R-M_modules
- tracit, https://github.com/emilknudstrup/tracit
- ARoME, https://cdsarc.u-strasbg.fr/viz-bin/qcat?J/A+A/550/A53; https://github.com/andres-jordan/PyARoME; https://github.com/esedagha/ARoMEpy
- Ellc, https://github.com/pmaxted/ellc
- RMfit, private repo
- RM Revolutions https://github.com/rodluger/rmrevolutions/tree/master
- Covino 2013,https://github.com/LucaMalavolta/PyORBIT/blob/main/pyorbit/models/rossitermclaughlin_precise.py
### Database
#### RV database
- http://atlas.obs-hp.fr/ ELODIE/SOPHIE database
- https://ui.adsabs.harvard.edu/abs/2017AJ....153..208B/abstract LSK, HIRES
- https://dace.unige.ch/radialVelocities/ from Paul
- https://arxiv.org/abs/2001.05942 A public HARPS radial velocity database corrected for systematic errors
- https://ui.adsabs.harvard.edu/abs/2014ApJ...785..126K/abstract friend of hot jupiters, HIRES

#### TTV database
- https://transit-timing.github.io/  8,667 transit timing measurements for 382 systems

### K2 light curve
- https://lweb.cfa.harvard.edu/~avanderb/k2.html

### Programming languages
#### Fortran
- http://micro.ustc.edu.cn/Fortran/ZJDing/ (in Chinese)
- https://github.com/jacobwilliams/Fortran-Astrodynamics-Toolkit Astrodynamics code in Fortran
- https://github.com/jacobwilliams/pyplot-fortran Use pyplot in Fortran

### Spectrum analysis


#### Lamost Spec
- https://github.com/hypergravity/laspec

#### APOGEE DR16 IR Spec
- https://dr16.sdss.org/infrared/spectrum/search
- https://data.sdss.org/datamodel/files/APOGEE_ASPCAP/APRED_VERS/ASPCAP_VERS/TELESCOPE/FIELD/aspcapStar.html data format

#### Least-Squares deconvolution (LSD)
- https://github.com/TVanReeth/Least-squares-deconvolution
- https://github.com/IvS-KULeuven/IvSPythonRepository (recommanded)

#### Keck Spec archive
- https://koa.ipac.caltech.edu/cgi-bin/KOA/nph-KOAlogin

#### Keck data reduction Tools
- https://www2.keck.hawaii.edu/koa/public/drp.html

#### Normalization
- https://github.com/arpita308/NEID_Tutorials
- https://github.com/RozanskiT/HANDY.  interactiva GUI supported

#### iSpec status
- https://github.com/marblestation/iSpec/blob/a513ad6e5ef84709518d11369241aef59b2f183d/ispec/modeling/mpfit.py


### Stellar modelling

#### 3D dust
- https://github.com/jobovy/mwdust

#### Correct routine to install isochrones
- https://github.com/timothydmorton/isochrones/issues/138 
#### LDC calculation
- https://github.com/ucl-exoplanets/ExoTETHyS/

### K2 detrend

- https://github.com/rodluger/everest

### Validation
#### VESPA install yml
- https://github.com/alexteachey/MoonPy/blob/master/env_setup_files/vespa_for_linux.yml 
  Note: if conda can find vespa=0.5.1, you can comment out it and install it munually.  
  The representation keyword/property name is deprecated in favor of representation_type 26 
  https://github.com/timothydmorton/VESPA/issues/26
#### The Kepler DR25 Robovetter
- https://github.com/nasa/kepler-robovetter


### TTV modelling & Photodynamical fit
#### Systemicv2
- https://github.com/stefano-meschiari/Systemic2
- https://research.iac.es/sieinvens/siepedia/pmwiki.php?n=Tutorials.SystemiconMac
#### exostriker
- https://github.com/3fon3fonov/exostriker

#### nauyaca using ptemcee (effective!)
- https://github.com/EliabCanul/nauyaca
#### NbodyGradient
- https://github.com/ericagol/NbodyGradient.jl

#### TTVMCMC (test some mcmc performance for TTV fitting)
- https://github.com/nwtuchow/TTVMCMC

#### TTVFaster with gradient calculation
- https://github.com/nwtuchow/TTVFaster
- Will be very helpful when do HMC in the TTV + RV fit.
#### TIDYMESS (simualtion with tidy effect, may useful for simulation in the near future)
- arXiv:2209.03955


### Nbody Simulation 
#### Gravitational Encounters in N-body simulations with GPU Acceleration
- https://arxiv.org/abs/1404.2324
- https://genga.readthedocs.io/en/latest/InitialConditions.html
#### Integration WIHT GPU
- https://www.cise.ufl.edu/research/SurfLab/swarmng/TutorialIntegrator.html



### Find period from data 
#### (online) generalised Lomb-Scargle periodogram
- http://www.astro.physik.uni-goettingen.de/~zechmeister/GLS/gls.html  (online)
- https://pyastronomy.readthedocs.io/en/latest/pyTimingDoc/pyPeriodDoc/gls.html (offline)

#### Understanding the Lomb-Scargle Periodogram
- https://arxiv.org/pdf/1703.09824.pdf

#### Wavelet
- https://github.com/alsauve/scaleogram 

#### ACF
- https://github.com/rae-holcomb/SpinSpotter

### MCMC & Nested sampling
#### The Markov-chain Monte Carlo Interactive Gallery
- https://chi-feng.github.io/mcmc-demo/
#### PINTS (Probabilistic Inference on Noisy Time-Series)
- https://github.com/pints-team/pints
#### PyCBC (includes ptemcee)
 - https://pycbc.org/pycbc/latest/html/index.html
#### bilby (includes ptemcee)
- https://lscsoft.docs.ligo.org/bilby/index.html

#### Dynesty examples
- https://notebook.community/joshspeagle/dynesty/demos/Examples%20--%20Linear%20Regression

### Schedule Tools

#### Object visibility
- http://catserver.ing.iac.es/staralt/index.php
#### definition of bright, dark, gray night
- https://www.ing.iac.es//PR/newsletter/news6/tel1.html
#### Moon Position and Distance Calculation
- https://pyastronomy.readthedocs.io/en/latest/pyaslDoc/aslDoc/moon.html


### Proposal Tools
#### NEID exopsure calculator
- http://neid-etc.tuc.noirlab.edu/calc_shell/calculate_rv

### WYIN time table
- https://noirlab.edu/science/observing-noirlab/scheduling/mso-telescopes

#### PFS
- NEID RV precision/(1.5~2.0)


### MSO Telescope Schedules including NEID
https://noirlab.edu/science/observing-noirlab/scheduling/mso-telescopes

### Object Visibility
- http://catserver.ing.iac.es/staralt/  staralt

- https://airmass.org/chart/obsid:kpno/date:2022-11-04/object:IC%201805/ra:38.175000/dec:61.450000 airmass

### TESS data quality flag

- https://outerspace.stsci.edu/display/TESS/2.0+-+Data+Product+Overview

### TESS Software Tools
- https://heasarc.gsfc.nasa.gov/docs/tess/software.html

### weather website
- https://www.meteoblue.com/en/weather/week/kitt-peak_united-states_5301146

### filters
- http://svo2.cab.inta-csic.es/svo/theory/fps3/index.php?mode=browse&gname=2MASS


### How to identify divergences in pymc3 chain using arviz
- https://stackoverflow.com/questions/66895673/how-to-identify-divergences-in-pymc3-chain-using-arviz
- https://www.pymc.io/projects/docs/en/v3/pymc-examples/examples/diagnostics_and_criticism/Diagnosing_biased_Inference_with_Divergences.html
- simple way to solve divergences: 1) use large accept rate; 2) reparameterize
