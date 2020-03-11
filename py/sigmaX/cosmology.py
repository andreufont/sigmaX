import numpy as np
import camb

def get_pars(H0=67.0,ombh2=0.022,omch2=0.12,mnu=0.0,As=2e-9,ns=0.96):
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=H0,ombh2=ombh2,omch2=omch2,mnu=mnu)
    pars.InitPower.set_params(As=As, ns=ns)
    return pars

def get_f(z,pars):
    Om=pars.omegam
    z1=1+z
    Omz=Om*z1**3/(Om*z1**3-Om+1)
    f=Omz**(6/11)
    return f

def get_results(z,pars):
    pars.set_matter_power(redshifts=[z])
    results = camb.get_results(pars)
    return results

def get_cosmo(z,b=2.0,H0=67.0,ombh2=0.022,omch2=0.12,mnu=0.0,As=2e-9,ns=0.96):
    
    # setup CAMB object from input cosmological parameters
    pars = get_pars(H0=H0,ombh2=ombh2,omch2=omch2,mnu=mnu,As=As,ns=ns)
    # compute logarithmic growth rate under GR
    f = get_f(z,pars)
    #print('f(z={}) = {}'.format(z,f))
    # run Boltzman code
    results = get_results(z,pars)
    
    # compute sigma_8 in Mpc/h, at input z
    sig8 = results.get_sigmaR(R=8,hubble_units=True)
    #print('sigma_8(z={}) = {}'.format(z,sig8))
    # compute sigma_12 in Mpc, at input z
    sig12 = results.get_sigmaR(R=12,hubble_units=False)
    #print('sigma_12(z={}) = {}'.format(z,sig12))
    # interpolator object for linear matter power (in 1/Mpc)
    cambP = results.get_matter_power_interpolator(nonlinear=False,
                hubble_units=False,k_hunit=False)
    
    # compute coordinate transformations
    DA = results.angular_diameter_distance(z)
    # This is equivalent to DH_true=2.998e5/results.hubble_parameter(z)
    DH = 1.0/results.h_of_z(z)
    #print('D_A(z={}) = {}'.format(z,DA))
    #print('D_H(z={}) = {}'.format(z,DH))
    cosmo = {'z':z, 'b':b, 'pars':pars, 'f':f, 'results':results,
             'sig8':sig8, 'sig12':sig12, 'cambP':cambP, 'DA':DA, 'DH':DH}

    return cosmo


def get_linP(cosmo,k_Mpc):
    """Compute linear power spectrum at input wavenumbers."""

    z=cosmo['z']
    return cosmo['cambP'].P(z,k_Mpc)


def print_cosmo_info(cosmo,keys=['f','sig8','sig12','DA','DH']):
    for key in keys:
        print(key,'=',cosmo[key])


