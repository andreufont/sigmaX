import numpy as np
from sigmaX import galaxy_power

def get_survey(cosmo,Nk=5,dk=0.03,V_Mpc3=1.e9,N_gal=1.e6):
    """Setup galaxy survey, and galaxy power measurement."""

    # get information from true cosmology
    z=cosmo['z']
    DA=cosmo['DA']
    DH=cosmo['DH']
    # bin width of power spectra
    dqt=dk*DA*(1+z)
    dqp=dk*DH*(1+z)
    # total number of bins
    Ntot=Nk*Nk
    qt=np.empty(Ntot)
    qp=np.empty(Ntot)
    for it in range(Nk):
    	iqt=dqt*(it+1)
    	for ip in range(Nk):
        	iqp=dqp*(ip+1)
        	i=it*Nk+ip
        	qt[i]=iqt
        	qp[i]=iqp

    # survey volume in observed coordinates (in stereoradians*dv/c)
    DV = DH*DA**2*(1+z)**3 
    V_obs = V_Mpc3 / DV
    # density of galaxies (in stereoradians*dv/c)
    N_gal = 1e6
    ng_obs= N_gal / V_obs

    # density times signal (at true cosmology)
    nP = ng_obs * galaxy_power.get_galP_obs(qt,qp,cosmo,cosmo)
    # fractional error on band power
    sigP_P = 2 * np.pi * np.sqrt(2/(V_obs*qt*dqt*dqp)) * (1+nP / nP)

    survey={'z':z,'DA':DA,'DH':DH,'DV':DV,
            'Nk':Nk,'dqt':dqt,'dqp':dqp,'qt':qt,'qp':qp,
            'V_Mpc3':V_Mpc3,'V_obs':V_obs,'N_gal':N_gal,'ng_obs':ng_obs,
            'nP':nP,'sigP_P':sigP_P}

    return survey


def get_galP_rand(survey,cosmo):
    """Generate random measurement of observed galaxy power given cosmology"""

    # get prediction for each observed wavenumber
    qt=survey['qt']
    qp=survey['qp']
    galP_obs=galaxy_power.get_galP_obs(qt,qp,cosmo,cosmo) 

    # get randomness given diagonal covariance matrix
    galP_err=galP_obs*survey['sigP_P']
    galP_real=galP_obs+galP_err*np.random.normal(size=len(qt))

    return galP_real, galP_err

