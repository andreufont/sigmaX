import numpy as np
import matplotlib.pyplot as plt
from sigmaX import cosmology

def get_galP_Mpc(kt,kp,cosmo,params={}):
    """Compute galaxy power spectrum, in Mpc.
        - cosmo will be used as a template for P(z,k).
        - params can modify the values of 'f sigma_8' and 'b sigma_8'.
    """
    
    z=cosmo['z']
    b=cosmo['b']
    
    # compute (k,mu) from (kt,kp)
    k=np.sqrt(kt**2+kp**2)
    mu=kp/k
    # interpolate to input k [1/Mpc]
    P=cosmo['cambP'].P(z,k)
    
    # compute P / sigma_8^2
    P_sig8 = P/cosmo['sig8']**2
    
    # compute b sig8
    if 'bsig8' in params:
        bsig8 = params['bsig8']
    else:
        bsig8 = b*cosmo['sig8']
    #print('b sigma_8 = {}'.format(bsig8))
    
    # compute f sig8
    if 'fsig8' in params:
        fsig8 = params['fsig8']
    else:
        fsig8 = cosmo['f']*cosmo['sig8']
    #print('f sigma_8 = {}'.format(fsig8))
    
    # apply Kaiser model
    galP=(bsig8+fsig8*mu**2)**2*P_sig8
    
    return galP


def get_galP_obs(qt,qp,cosmo_coord,cosmo_temp,params={}):
    """Given observed coordinates, transform to comoving and compute
	galaxy power spectrum.
        - (qt,qp) are wavenumbers in dimensionless, observable units.
        - cosmo_coord is used to transform these to comoving wavenumbers.
        - cosmo_temp is used to compute the P(k) template.
        - params can modify 'f sig8', 'b sig8', alpha_par, alpha_perp.
    """
    
    z=cosmo_coord['z']
    assert z==cosmo_temp['z'], print('z should be the same in both cosmologies')
    b=cosmo_temp['b']
    
    # compute coordinate transformations
    DA=cosmo_coord['DA']
    DH=cosmo_coord['DH']
    # rescale coordinates with alpha parameters
    if 'at' in params:
        DA *= params['at']
    if 'ap' in params: 
        DH *= params['ap']
    #print('D_A = {}'.format(DA))
    #print('D_H = {}'.format(DH))
    kt=qt/DA/(1+z)
    kp=qp/DH/(1+z)
    
    # call galaxy power spectrum in comoving coordinates
    galP=get_galP_Mpc(kt,kp,cosmo=cosmo_temp,params=params)
    
    # normalize with volume factor (DV is volume, nothing to do with iso BAO)
    DV=DH*DA*DA*(1+z)**3
    
    return galP/DV


def plot_galP_Mpc(cosmo_temp,k=None):
    """Plot galaxy power spectrum, in 1/Mpc, for mu=0 and mu=1"""

    # define wavenumbers
    if k is None:
        k=np.linspace(0.001,1.0,1000)

    plt.loglog(k,cosmology.get_linP(cosmo_temp,k),label='linear')
    plt.loglog(k,get_galP_Mpc(k,0.0,cosmo_temp),label=r'$\mu=0$')
    plt.loglog(k,get_galP_Mpc(0.0,k,cosmo_temp),label=r'$\mu=1$')
    plt.legend()
    plt.xlabel('k [1/Mpc]')
    plt.ylabel(r'$P(k) [Mpc^3]$')
    plt.show()


def plot_galP_obs(qt,qp,cosmo_coord,cosmo_temp):
    """Plot model of galaxy power spectrum, in observed coordinates"""

    # compute true model for galaxy power
    galP_obs=get_galP_obs(qt,qp,cosmo_coord,cosmo_temp) 

    plt.yscale('log')
    plt.scatter(qt,galP_obs,c=qp)
    cbar=plt.colorbar()
    cbar.set_label(r'$q_\parallel$', labelpad=+1)
    plt.xlabel(r'$q_\perp$')
    plt.ylabel(r'$P_g(q)$')
    plt.show()


