'''
Created on 2020-05-05 16:14:14
Last modified on 2020-09-30 07:51:43

@author: L. F. Pereira (lfpereira@fe.up.pt)
'''


# imports

# standard library
from collections import OrderedDict

# third-party
import numpy as np
from scipy import interpolate, integrate, signal


# function definition

def get_results(data, max_strain=.02, additional_strain_thresh=.05,
                n_interpolation=10000):

    # get data (linear buckling)
    coilable, sigma_crit = get_results_lin_buckle(data)

    # get data (Riks)
    if coilable:
        _, _, (energy, _), E_max = read_and_clean_results_riks(
            data, additional_strain_thresh=additional_strain_thresh,
            n_interpolation=n_interpolation, get_energy=True)
    else:
        energy = None

    # update coilability
    if coilable and E_max is not np.nan and E_max > max_strain:
        coilable = 2

    return OrderedDict([('coilable', coilable), ('sigma_crit', sigma_crit), ('energy', energy)])


def get_results_lin_buckle(data, job_name='SUPERCOMPRESSIBLE_LIN_BUCKLE'):

    # get data
    results = data['post-processing'][job_name]

    # coilability and P_crit
    coilable = results['coilable']
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

    # verify if has info about max strain
    E_max = np.max(np.abs(np.array(results.get('E', [np.nan]))))

    # get geometric parameters
    bottom_diameter = data['variables']['bottom_diameter']
    bottom_area = np.pi * bottom_diameter**2 / 4
    pitch = data['variables']['ratio_pitch'] * bottom_diameter

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
    # TODO: verify if this is required
    acceptable_curve = np.size(np.where(diff < 0)) == 0
    if acceptable_curve:
        i_stress_local_maxs = signal.argrelextrema(stress, np.greater)
        # delete increasing in load at the end
        if np.size(i_stress_local_maxs) > 0:
            while stress[-2] < stress[-1]:
                strain = strain[:-1]
                stress = stress[:-1]

    if not get_energy or not acceptable_curve:
        return (u_3, rf_3), (strain, stress), (None, (None, None)), E_max

    # is it possible to compute energy?
    success = False
    if np.size(i_stress_local_maxs) > 0:
        strain_thresh = strain[i_stress_local_maxs[0][0]] + additional_strain_thresh
        if strain[-1] > strain_thresh and strain[-1] < 1.1:
            success = True

    if not success:
        return (u_3, rf_3), (strain, stress), (None, (None, None)), E_max

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

    return (u_3, rf_3), (strain, stress), (energy, (x, y)), E_max
