### Mutual inclination

$\cos i_{\mathrm{bc}}=\cos i_{\mathrm{b}} \cos i_{\mathrm{c}}+\sin i_{\mathrm{b}} \sin i_{\mathrm{c}} \cos \left(\Omega_{\mathrm{b}}-\Omega_{\mathrm{c}}\right)$.   

```
i_b = np.radians(90)
i_c = np.radians(90)
Omega_b = np.radians(0)
Omega_c = np.radians(0)
cosi_bc = np.cos(i_b)*np.cos(i_c)+np.sin(i_b)*np.sin(i_c)*np.cos(Omega_b-Omega_c)
i_bc = np.degrees(np.arccos(cosi_bc))

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

