""" Utility functions for working with molecule data. """

import csv

import numpy as np
import matplotlib.pyplot as plt

from scipy import sparse
from scipy.sparse.linalg import spsolve
from lmfit.models import GaussianModel, LorentzianModel, PseudoVoigtModel


def process_original_data(data): 
    """ Seperate the first column of the file and other columns.
        1st column: 2-theta (x-axis)
        other columns: intensity (y-axis)
        Return the resulted 2-theta and intensities after log10 processing. 

    Input: 2-D array, the first line is for x-axis, other lines for y-axis.
    """
    x = data[0]  # 2-theta
    ys = data[1:]  # intensity
    return x, ys


def plot_initial_2d(x, ys):
    """ Draw a figure for the data read from file. 

    Inputs:
        x: 1-D array.
        ys: 2-D array. 
    """
    for i in range(ys.shape[0]):
        plt.plot(x, ys[i], linewidth = 0.6)
        # plt.loglog(x, ys[i], linewidth = 0.6)
    plt.xlabel('2-theta') 
    plt.ylabel('Intensity')
    plt.savefig('Resulted_Figures/Initial_2d.png')
    plt.show() 


def plot_initial_3d(x, ys):
    """Plot original data (3d version)"""
    ax = plt.axes(projection='3d')
    for i in range(len(ys)):
        z = np.ones(len(ys[0]))*10*i  # time = dataset_index * 10 
        ax.plot3D(x,z,ys[i])
    ax.set_xlabel('q ($nm^{-1}$)')
    ax.set_ylabel('Time (min)')
    ax.set_zlabel('Intensity ($cm^{-1}$)')
    plt.savefig('Resulted_Figures/Initial_3d.png')
    plt.show()


def get_interval_indices(x, x_range):
    """ Return an array of indices, corresponding to the elements in the given 
    range.

    Inputs:
        x: 1-D array.
        x_range: 1-D list or array of 2 elements, the minimum and maximum of 
                 the range.
    """
    return np.where((x>x_range[0]) & (x<x_range[1]))[0]


def get_interval_data(x, y, x_range):
    interval_indices = get_interval_indices(x, x_range)
    return x[interval_indices], y[interval_indices]


def choose_model(model):
    if model == 'Gaussian' or model == 'gaussian' or model == 'g':
        return GaussianModel
    elif model == 'Lorentzian' or model == 'lorentzian' or model == 'l':
        return LorentzianModel
    elif model == 'Pseudo-Voigt' or model == 'pseudo-voigt' or model == 'p' or\
        model == 'pv':
        return PseudoVoigtModel
    else:
        return 1


def fit_curve(Model, x, y, initial_guess):
    mod = Model(prefix='m1_')
    pars = mod.guess(y, x=x)
    for i in range(len(initial_guess)):
        if i != 0:
            new_mod = Model(prefix='m{}_'.format(i+1))
            pars.update(new_mod.make_params())
            mod = mod + new_mod
        pars['m{}_center'.format(i+1)].set(value=initial_guess[i][0])
        pars['m{}_sigma'.format(i+1)].set(value=initial_guess[i][1])
        pars['m{}_amplitude'.format(i+1)].set(value=initial_guess[i][2])
    out = mod.fit(y, pars, x=x)
    return out.best_fit, out.result


# There is an algorithm called "Asymmetric Least Squares Smoothing" by P. 
# Eilers and H. Boelens in 2005. The paper is free and you can find it on 
# google.
def baseline_als(y, lam, p, niter=10):
    L = len(y)
    D = sparse.csc_matrix(np.diff(np.eye(L), 2))
    w = np.ones(L)
    for i in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D.dot(D.transpose())
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z


def check_float(str):
    """Check if the input string is a float or just string. 
    True if float.
    """
    try:
        f = float(str)
        return f
    except TypeError:
        return 0


def get_pars(fit_result):
    """Get results data and print them. """
    # amplitude, center, sigma, fwhm, height
    value = []
    std_err = []
    for name, par in fit_result.params.items():
        print(f"{name}: value={'%.6f'%float(par.value)}+/-{par.stderr}")
        value.append(float(par.value))
        std_err.append(check_float(par.stderr))
    return np.array(value), np.array(std_err)


def fit_curve_with_baseline(Model, x, y, x_range, guess, i=0):
    """1. Fit curve with result after subtracting baseline. 
    2. Plot results. 
    """
    xx, yy = get_interval_data(x, y, x_range)

    baseline = baseline_als(yy, 10000, 0.01)
    baseline_subtracted = yy - baseline
    best_fit, fit_result = fit_curve(Model,xx,baseline_subtracted,guess)

    plt.title(f"{i}-th dataset")
    plt.plot(xx, baseline, '-', c='tab:blue', label = 'baseline', linewidth = \
        1)
    plt.plot(xx,yy, '--', c='k', label = 'original data')
    label_bs = 'subtracted baseline'
    plt.plot(xx, baseline_subtracted, '--', c='tab:green', label = label_bs)
    plt.plot(xx, best_fit, '-', c='tab:red', label = 'fit curve')
    plt.legend()
    plt.savefig('Resulted_Figures/Dataset_{}.png'.format(i))
    plt.show()

    return fit_result


def summarize_data3D(Model, x, ys, x_range, num, guess):
    """Summerize the fitting results into an array. 
    x_range: interval limiting the range of peaks
    num: number of peaks

    The output:
    a 3-d array of shape num * 11 * len(ys)
    1st dimension: the index of peak focused on
    2nd dimension: the index of corresponding parameter
    3rd dimension: the index of current dataset

    For the 2nd dimension, first element represents time. The elements with odd
    index are different parameters, followed by an element describing the 
    corresponding error.
    """
    # data = np.zeros((11, len(ys)*num+1))  # time + pars_value + pars_stderr
    data_3d = np.zeros((num, 11, len(ys)))  # time + pars_value + pars_stderr
    for i in range(len(ys)):
        print(f"\n{i}th dataset: ")
        # data[0][i] = i*10
        fit_result = fit_curve_with_baseline(Model,x,ys[i],x_range,guess,i=i)
        pars_value, pars_stderr = get_pars(fit_result)
        for j in range(num):
            data_3d[j][0][i] = i*10
            for k in range(5):
                # store parameter values and their standard errors
                data_3d[j][2*(k+1)-1][i] = '%.6f'%float(pars_value[k+j*5])
                data_3d[j][2*(k+1)][i] = '%.6f'%float(pars_stderr[k+j*5])
    return data_3d


def plot_fwhm(data_2d, i):
    """Plot the figure showing changes in FXHM (for a peak).
    data_2d: shape of 11 * len(ys)
            11 --> number of parameter types
            len(ys) --> number of data sets
    i: the index of peak, min as 0. Imported to name picture file.
    """
    plt.title("Peak {} - Changes in FXHM".format(i+1))
    plt.plot(data_2d[0],data_2d[7],'k',linewidth=1.0)
    plt.errorbar(data_2d[0],data_2d[7],yerr=data_2d[8],fmt='o',\
        ecolor='tab:blue',color='tab:orange',elinewidth=1,capsize=1)
    plt.xlabel('Time (min)') 
    plt.ylabel('Full Width at Half Maximum')
    plt.savefig('Resulted_Figures/Peak{}_FXHM.png'.format(i+1))
    plt.show()


def plot_intensity(data_2d, i):
    """Plot the figure showing changes in Intendity (for a peak). 
    data_2d: shape of 11 * len(ys)
            11 --> number of parameter types
            len(ys) --> number of data sets
    i: the index of peak, min as 0. Imported to name picture file.
    """
    plt.title("Peak {} - Changes in Intensity".format(i+1))
    plt.plot(data_2d[0],data_2d[9],'k',linewidth=1.0)
    plt.errorbar(data_2d[0],data_2d[9],yerr=data_2d[10],fmt='o',\
        ecolor='tab:blue',color='tab:orange',elinewidth=1,capsize=1)
    plt.xlabel('Time (min)') 
    plt.ylabel('Intensity')
    plt.savefig('Resulted_Figures/Peak{}_Intensity.png'.format(i+1))
    plt.show()


def tabulate_result(data_2d, i):
    """Output data of a peak to a .csv file. 
    data_2d: shape of 11 * len(ys)
            11 --> number of parameter types
            len(ys) --> number of data sets
    i: the index of peak, min as 0. Imported to name the .csv file.
    """
    output_data = np.array(data_2d).transpose()  # same format as input file
    header = ['TIME','AMPLITUDE','STD_ERR','CENTER','STD_ERR','SIGMA',\
        'STD_ERR','FWHM','STD_ERR','HEIGHT','STD_ERR']

    with open('Output_Data/Peak_{}_Result.csv'.format(i+1), 'w', ) as fp:
        wr = csv.writer(fp, quoting=csv.QUOTE_ALL)
        wr.writerow(header)
        for line in output_data:
            wr.writerow(line)
        wr.writerow([ ])
        wr.writerow([' ','(  0.0 in STD_ERR ','means NoneType. )'])


def summarize_peaks(data_3d):
    """1. Tabulate parameters for several peaks.
    2. Plot changes in FWHM and Intensity for peaks. 
    
    data_3d:
    a 3-d array of shape num * 11 * len(ys)
    1st dimension: the index of peak focused on
    2nd dimension: the index of corresponding parameter
    3rd dimension: the index of current dataset"""
    for i in range(len(data_3d)):
        plot_fwhm(data_3d[i], i)
        plot_intensity(data_3d[i], i)
        tabulate_result(data_3d[i], i)


# def compare_models(x, y, guess1, guess2):
def compare_models(x, y, guess):
    """Compare 3 models with a figure. """
    baseline = baseline_als(y, 10000, 0.01)
    baseline_subtracted = y - baseline
    best_fit_gau,_ = fit_curve(GaussianModel, x, baseline_subtracted, guess)
    best_fit_lor,_ = fit_curve(LorentzianModel, x, baseline_subtracted, guess)
    best_fit_pse,_ = fit_curve(PseudoVoigtModel, x, baseline_subtracted, guess)

    result_gauss = best_fit_gau + baseline
    result_loren = best_fit_lor + baseline
    result_pseudo = best_fit_pse + baseline
    return result_gauss, result_loren, result_pseudo


def plot_gaussian_result(x, y, result_gauss):
    plt.title(f"Gaussian Result")
    plt.plot(x, y, '--', c='k', label='Original Data')
    plt.plot(x, result_gauss, '-', label='Gaussian')
    plt.plot(x, abs(result_gauss-y), '-', label='error')
    plt.legend()
    plt.savefig('Resulted_Figures/Comparison_Gaussian.png')
    plt.show()


def plot_lorentzian_result(x, y, result_loren):
    plt.title(f"Lorentzian Result")
    plt.plot(x, y, '--', c='k', label='Original Data')
    plt.plot(x, result_loren, '-', label='Lorentzian')
    plt.plot(x, abs(result_loren-y), '-', label='error')
    plt.legend()
    plt.savefig('Resulted_Figures/Comparison_Lorentzian.png')
    plt.show()


def plot_pseudo_voigt_result(x, y, result_voigt):
    plt.title(f"Pseudo-Voigt Result")
    plt.plot(x, y, '--', c='k', label='Original Data')
    plt.plot(x, result_voigt, '-', label='Pseudo-Voigt')
    plt.plot(x, abs(result_voigt-y), '-', label='error')
    plt.legend()
    plt.savefig('Resulted_Figures/Comparison_Pseudo_Voigt.png')
    plt.show()


def tabulate_comparison(data_2d):
    header = ['X', 'Y_Original', 'Y_Gaussian', 'Y_Lorentzian', 'Y_Pseudo_Voigt']
    with open('Output_Data/Comparison.csv', 'w', ) as fp:
        wr = csv.writer(fp, quoting=csv.QUOTE_ALL)
        wr.writerow(header)
        for line in np.array(data_2d).transpose():
            wr.writerow(np.around(line, 6))
            # wr.writerow(line)


def fit_index(data_2d):
    # Pearson's chi-square test
    # https://en.wikipedia.org/wiki/Goodness_of_fit
    fit_indices = np.zeros(3)
    for i in range(3):
        goodness = 0
        for j in range(len(data_2d[0])):
            goodness += (data_2d[i+2][j]-data_2d[1][j])**2 / data_2d[1][j]
        fit_indices[i] = goodness
    print(f"\nFit index for Gaussian: {fit_indices[0]}")
    print(f"Fit index for Lorentzian: {fit_indices[1]}")
    print(f"Fit index for Pseudo-Voigt: {fit_indices[2]}")
    return fit_indices


def summarize_comparison(x, y, x_range, guess):
    xx, yy = get_interval_data(x, y, x_range)
    result_gauss, result_loren, result_voigt = compare_models(xx, yy, guess)
    data_2d = np.zeros((5, len(xx)))  # time + pars_value + pars_stderr
    data_2d[0] = xx
    data_2d[1] = yy
    data_2d[2] = result_gauss
    data_2d[3] = result_loren
    data_2d[4] = result_voigt

    tabulate_comparison(data_2d)
    fit_index(data_2d)
    plot_gaussian_result(xx, yy, result_gauss)
    plot_lorentzian_result(xx, yy, result_loren)
    plot_pseudo_voigt_result(xx, yy, result_voigt)

