'''
Created on 2020-05-05 16:14:14
Last modified on 2020-05-07 16:19:54
Python 3.7.3
v0.1

@author: L. F. Pereira (lfpereira@fe.up.pt)

Main goal
---------

'''


#%% imports

# third-party
import numpy as np
from scipy import interpolate, integrate, signal


#%% function definition

def get_results_lin_buckle(data, job_name='SUPERCOMPRESSIBLE_LIN_BUCKLE'):

    # get data
    results = data['post-processing'][job_name]

    # coilability and P_crit
    coilable = results['coilable'][0]
    try:
        # get load
        load = results['loads'][0]
        # get geometric parameters
        bottom_diameter = data['variables']['bottom_diameter']
        bottom_area = np.pi * bottom_diameter**2 / 4
        # get sigma crit
        sigma_crit = load / bottom_area * 1e3
    except IndexError:
        sigma_crit = None

    return coilable, sigma_crit


def read_and_clean_results_riks(data, job_name='SUPERCOMPRESSIBLE_RIKS',
                                additional_strain_thresh=.05,
                                get_energy=False, n_interpolation=10000):
    '''
    Reads and cleans results of a supercompressible simulation.

    Parameters
    ----------
    # TODO: complete
    '''

    # get data
    results = data['post-processing'][job_name]

    # get geometric parameters
    pitch = data['variables']['pitch']
    bottom_diameter = data['variables']['bottom_diameter']
    bottom_area = np.pi * bottom_diameter**2 / 4

    # get load-displacement
    u_3 = np.abs(np.array(results['U'][-1]))
    rf_3 = np.abs(np.array(results['RF'][-1]))

    # get stress-strain
    strain = u_3 / pitch
    stress = rf_3 / bottom_area * 1e3  # kPa

    # clean data
    # to make sure strain always increase (pchip requires strickly increasing)
    diff = np.diff(strain)
    indices = np.where(np.logical_and(diff >= -0.01, diff <= 0.))[0] + 1
    strain = np.delete(strain, indices)
    stress = np.delete(stress, indices)

    # recover load and displacement
    u_3 = strain * pitch
    rf_3 = stress * bottom_area / 1e3

    # if last value is max, it is not considered
    diff = np.diff(strain)
    acceptable_curve = np.size(np.where(diff < 0)) == 0
    if acceptable_curve:
        i_stress_local_maxs = signal.argrelextrema(stress, np.greater)
        # delete increasing in load at the end
        if np.size(i_stress_local_maxs) > 0:
            while stress[-2] < stress[-1]:
                strain = strain[:-1]
                stress = stress[:-1]

    if not get_energy or not acceptable_curve:
        return (u_3, rf_3), (strain, stress), (None, (None, None))

    # is it possible to compute energy?
    success = False
    if np.size(i_stress_local_maxs) > 0:
        strain_thresh = strain[i_stress_local_maxs[0][0]] + additional_strain_thresh
        if strain[-1] > strain_thresh and strain[-1] < 1.1:
            success = True

    if not success:
        return (u_3, rf_3), (strain, stress), (None, (None, None))

    # compute energy
    # append point
    if strain[-1] < 1.:
        strain = np.append(strain, 1.)
        stress = np.append(stress, 0.)

    # interpolate
    interp = interpolate.pchip(strain, stress)
    x = np.linspace(0, 1, n_interpolation)
    y = interp(x)

    # compute energy
    energy = integrate.simps(y, x=x)

    return (u_3, rf_3), (strain, stress), (energy, (x, y))
