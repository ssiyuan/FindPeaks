""" Utility functions for working with data. """

import csv

import numpy as np
import matplotlib.pyplot as plt

from scipy import sparse
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse.linalg import spsolve
from lmfit.models import GaussianModel, LorentzianModel, PseudoVoigtModel

from import_files import check_output_dir
from validation import validate_x_range


def plot_initial_2d(x, ys):
    """Draw a 2D figure. Use q instead of intensity as x, and use log(intensity) instead 
    of intensity as y.

    Args:
        x (array): 1D.
        ys (array): 2D array, each row as a seperate dataset of y.
    """
    plt.title('2D Figure',weight='bold')
    plt.xlabel('q ($nm^{-1}$)',weight='bold')
    plt.ylabel('Intensity ($cm^{-1}$)',weight='bold')
    for i in range(ys.shape[0]):
        plt.plot(x, ys[i], linewidth=1.0)
    check_output_dir('output_figures')
    plt.savefig('output_figures/Initial_2d.png')
    plt.show()


def plot_initial_3d(x, ys, index_min=0, index_max=610):
    """Plot a 3D figure. Use q instead of intensity as x, and use log(intensity) instead 
    of intensity as y.

    Args:
        x (array): 1D.
        ys (array): 2D array, each row as a seperate dataset of y.
        index_min (int): lower bound for index. Default as 0.
        index_max (int): higner bound for index. Default as 620 OR the largest index of array.
    """
    if index_max > len(x):
        index_max = len(x)
    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlabel('q ($nm^{-1}$)',fontweight='bold')
    for i in range(len(ys)):
        time = np.ones(len(ys[0]))*10*(i+1)  # time = (dataset_index+1) * 10
        ax.plot3D(x[index_min:index_max], time[index_min:index_max], ys[i][index_min:index_max], 'k') 
    ax.set_ylabel('Time (min)',fontweight='bold')
    ax.set_zlabel('I ($cm^{-1}$)',fontweight='bold')
    ax.set_title('3D Figure', fontweight='bold')
    check_output_dir('output_figures')
    plt.savefig('output_figures/Initial_3d.png')
    plt.show()


def get_interval_data(x, y, x_range):
    """Find x values in the given range, and corresponding y values. 

    Args:
        x (array): 1D array
        y (array): 1D array
        x_range (array): shape of (1,2), including lower and upper bound

    Returns:
        array: elements of x in given range
        array: elements of y corresponding to x in range
    """
    interval_indices = np.where((x > x_range[0]) & (x < x_range[1]))[0]
    return x[interval_indices], y[interval_indices]


def choose_model_with_str(model):
    """Choose suitable model corresponding to the input. 

    Args:
        model (string): the model you want to select

    Returns:
        function: the function corresponding to model input
    """
    if model == 'Gaussian' or model == 'gaussian' or model == 'g':
        return GaussianModel
    elif model == 'Lorentzian' or model == 'lorentzian' or model == 'l':
        return LorentzianModel
    elif model == 'Pseudo-Voigt' or model == 'pseudo-voigt' or model == 'p' or model == 'pv':
        return PseudoVoigtModel
    else:
        return 1


def fit_curve(Model, x, y, initial_guess, center_min=0.0):
    """Fit peaks of original data to similar curve of given model.
    Args:
        Model (function): the model used to fit peaks
        x (array): 1D
        y (array): 1D
        initial_guess (array): 1D array, the initial guess of center, sigma, and amplitude
        center_min (float, optional): set the lower bound of the center. Defaults to 0.

    Returns:
        array: the model results on the y-axis
        class 'lmfit.minimizer.MinimizerResult': including the parameters and errors of the\
             model which fits the peak best
    """
    mod = Model(prefix='m1_')
    pars = mod.guess(y, x=x)
    for i in range(len(initial_guess)):
        if i != 0:
            new_mod = Model(prefix='m{}_'.format(i+1))
            pars.update(new_mod.make_params())
            mod = mod + new_mod
        pars['m{}_center'.format(
            i+1)].set(value=initial_guess[i][0], min=center_min)
        pars['m{}_sigma'.format(i+1)].set(value=initial_guess[i][1])
        pars['m{}_amplitude'.format(i+1)].set(value=initial_guess[i][2], min=0)
    out = mod.fit(y, pars, x=x)
    return out.best_fit, out.result


# Algorithm of "Asymmetric Least Squares Smoothing" by P. Eilers and H. Boelens in 2005.
def get_baseline(y, lam, p, iter_num=10):
    """Generate a baseline for an input y set.

    Args:
        y (array): 1D, y values
        lam (int): smoothness, 10^2 <= λ <= 10^9
        p (float): wheighting deviations
        iter_num (int, optional): times for iterations. Defaults to 10.

    Returns:
        array: 1D, the baseline
    """
    length = len(y)
    diag = np.eye(length)
    w = np.ones(length)
    D = sparse.csc_matrix(np.diff(diag, 2))
    for i in range(iter_num):
        W = sparse.spdiags(w, 0, length, length)
        Z = W + lam * D.dot(D.transpose())
        baseline = spsolve(Z, w*y)
        w = p * (y > baseline) + (1-p) * (y < baseline)
    return baseline


def check_stderr(stderr):
    """Check if the standard error is a float or NoneType. 
    Replace it with 0 when it is NoneType.

    Args:
        stderr (string): the standard error, a float or string 'NoneType'

    Returns:
        float: the float format of stderr; or 0 when stderr is 'NoneType'
    """
    try:
        err = float(stderr)
        return err
    except TypeError:
        return 0


def get_pars(fit_result):
    """Summerize the fitting results and print them. 

    Args:
        fit_result (class 'lmfit.minimizer.MinimizerResult'): the fitting result about model parameters

    Returns:
        array: parameters of fitting results, in order are amplitude, center, sigma, fwhm, height
        array: the corresponding standard error of parameters
    """
    value = []
    std_err = []
    for name, par in fit_result.params.items():
        print(f"{name}: value={'%.6f'%float(par.value)}+/-{par.stderr}")
        value.append(float(par.value))
        std_err.append(check_stderr(par.stderr))
    print("\n")
    return np.array(value), np.array(std_err)


def plot_fitting_results(i, x, y, baseline, best_fit):
    """Plot the fitting results with baseline and original curve.

    Args:
        i (int): index of the current dataset, to name the figure
        x (array): 1D
        y (array): 1D
        baseline (array): 1D
        best_fit (array): 1D, the fitting result of the original data subtracting baseline
    """
    plt.title(f"{i}-th Dataset",weight='bold')
    plt.rcParams.update({'font.weight': 'bold'})
    plt.plot(x, baseline, '-', c='tab:blue', label='Baseline', linewidth=1)
    plt.plot(x, y, '--', c='k', label='Original Data', linewidth=1)
    label_bs = 'Subtracted Baseline'
    plt.plot(x, y - baseline, '--', c='tab:green', label=label_bs, linewidth=1)
    plt.plot(x, best_fit, '-', c='tab:red', label='Fit Curve', linewidth=1)
    plt.legend()
    plt.xlabel('q ($nm^{-1}$)',weight='bold')
    plt.ylabel('Intensity ($cm^{-1}$)',weight='bold')
    check_output_dir('output_figures')
    plt.savefig('output_figures/Dataset_{}.png'.format(i))
    plt.close()


def summarize_data3D(Model, x, ys, x_range, guess, center_min=0.0):
    """Fit peaks with baseline and summerize the fitting results into an array. 

    Args:
        Model (function): model used to fit peaks
        x (array): x values
        ys (array): y datasets
        x_range (array): shape of (1,2), lower and upper bound
        guess (array): 2D array. Each row with 3 elements, center, sigma and amplitude.
        center_min (float, optional): the lower bound of center. Defaults to 0.

    Returns:
        array: shape of (len(guess), 11, len(ys))
            1st dimension: the peak focused on
            2nd dimension: time+amplitude+error+center+error+sigma+error+fwhm+error+height+error
            3rd dimension: current dataset
    """
    validate_x_range(x, x_range)

    data_3d = np.zeros((len(guess), 11, len(ys)))
    for i in range(len(ys)):
        # print(f"\n{i}th dataset: ")
        interval_x, interval_y = get_interval_data(x, ys[i], x_range)
        baseline = get_baseline(interval_y, 10000, 0.01)
        baseline_subtracted = interval_y - baseline
        best_fit, fit_result = fit_curve(
            Model, interval_x, baseline_subtracted, guess, center_min)
        plot_fitting_results(i, interval_x, interval_y, baseline, best_fit)
        pars_value, pars_stderr = get_pars(fit_result)

        for j in range(len(guess)):
            data_3d[j][0][i] = i
            for k in range(5):
                # store parameter values and their standard errors
                data_3d[j][2*(k+1)-1][i] = '%.6f' % float(pars_value[k+j*5])
                data_3d[j][2*(k+1)][i] = '%.6f' % float(pars_stderr[k+j*5])
    return data_3d


def plot_fwhm(data_2d, i):
    """Plot the figure showing changes in FWHM (for a peak)

    Args:
        data_2d (array): shape of (11, len(ys)). Each row is for a parameter type, 
                        each column is for a dataset.
        i (int): index of peak, min as 0. Imported to name picture file.
    """
    plt.title("Peak {} - Changes in FWHM".format(i+1),weight='bold')
    plt.plot(data_2d[0]*10, data_2d[7], 'k', linewidth=1.0)
    plt.errorbar(data_2d[0]*10, data_2d[7], yerr=data_2d[8], fmt='o',
                 ecolor='tab:blue', color='tab:orange', elinewidth=1, capsize=1)
    plt.xlabel('Time (min)',weight='bold')
    plt.ylabel('Full Width at Half Maximum',weight='bold')
    check_output_dir('output_figures')
    plt.savefig('output_figures/Peak{}_FWHM.png'.format(i+1))
    plt.show()


def plot_amplitude(data_2d, i):
    """Plot the figure showing changes in amplitude (for a peak), representing 
    area of a peak. 

    Args:
        data_2d (array): shape of (11, len(ys)). Each row is for a parameter type, 
                        each column is for a dataset.
        i (int): index of peak, min as 0. Imported to name picture file.
    """
    plt.title("Peak {} - Changes in Peak Area".format(i+1),weight='bold')
    plt.plot(data_2d[0]*10, data_2d[1], 'k', linewidth=1.0)
    plt.errorbar(data_2d[0]*10, data_2d[1], yerr=data_2d[2], fmt='o',
                 ecolor='tab:blue', color='tab:orange', elinewidth=1, capsize=1)
    plt.xlabel('Time (min)',weight='bold')
    plt.ylabel('Peak Area',weight='bold')
    check_output_dir('output_figures')
    plt.savefig('output_figures/Peak{}_PeakArea.png'.format(i+1))
    plt.show()


def plot_intensity(data_2d, i):
    """Plot the figure showing changes in peak height.

    Args:
        data_2d (array): shape of (11, len(ys)). Each row is for a parameter type, 
                        each column is for a dataset.
        i (int): index of peak, min as 0. Imported to name picture file.
    """
    plt.title("Peak {} - Changes in Intensity".format(i+1),weight='bold')
    plt.plot(data_2d[0]*10, data_2d[9], 'k', linewidth=1.0)
    plt.errorbar(data_2d[0]*10, data_2d[9], yerr=data_2d[10], fmt='o',
                 ecolor='tab:blue', color='tab:orange', elinewidth=1, capsize=1)
    plt.xlabel('Time (min)',weight='bold')
    plt.ylabel('Intensity ($cm^{-1}$)',weight='bold')
    check_output_dir('output_figures')
    plt.savefig('output_figures/Peak{}_Intensity.png'.format(i+1))
    plt.show()


def tabulate_result(data_2d, i, dir_path='output_files'):
    """Output data of a peak to a .csv file. 

    Args:
        data_2d (array): shape of (11, len(ys)). Each row is for a parameter type, 
                        each column is for a dataset.
        i (int): index of peak, min as 0. Imported to name the .csv file.
        dir_path (str, optional): the directory to store the file. Defaults to 'output_files'.
    """
    check_output_dir(dir_path)
    # convert to same format as input file
    output_data = np.array(data_2d).transpose()
    header = ['Dataset Index', 'AMPLITUDE', 'STD_ERR', 'CENTER', 'STD_ERR', 'SIGMA', 'STD_ERR', 'FWHM',
              'STD_ERR', 'HEIGHT', 'STD_ERR']

    with open(f"{dir_path}/Peak_{i+1}_Result.csv", "w", encoding="utf-8") as fp:
        # with open('output_files/Peak_{}_Result.csv'.format(i+1), "w", ) as fp:
        wr = csv.writer(fp, quoting=csv.QUOTE_ALL)
        wr.writerow(header)
        for line in output_data:
            wr.writerow(line)
        wr.writerow([])
        wr.writerow([' ', '(  0.0 in STD_ERR ', 'means NoneType. )'])


def summarize_peaks(data_3d, dir_path='output_files'):
    """Tabulate the results for each peak, and plot changes along FWHM and Intensity.

    Args:
        data_3d (array): shape of (num, 11, len(ys))
                        dimension 1: peak
                        dimension 2: time, parameters and errors
                        dimension 3: datasets
        dir_path (str, optional): the directory to store the file. Defaults to 'output_files'.
    """
    for i in range(len(data_3d)):
        plot_fwhm(data_3d[i], i)
        plot_amplitude(data_3d[i], i)
        plot_intensity(data_3d[i], i)
        tabulate_result(data_3d[i], i, dir_path)


def compare_models(x, y, guess, center_min=0.0):
    """Compare 3 models with a figure.

    Args:
        x (array): 1D
        y (array): 1D
        guess (array): 2D array. Columns correspond to: center, sigma and amplitude.
        center_min (float, optional): lower bound of center. Defaults to 0.0.

    Returns:
        array: fitting results of y for Gaussian model
        array: fitting results of y for Lorentzian Model
        array: fitting results of y for Pseudo Voigt Model
    """
    baseline = get_baseline(y, 10000, 0.01)
    baseline_subtracted = y - baseline
    best_fit_gau, _ = fit_curve(
        GaussianModel, x, baseline_subtracted, guess, center_min)
    best_fit_lor, _ = fit_curve(
        LorentzianModel, x, baseline_subtracted, guess, center_min)
    best_fit_pse, _ = fit_curve(
        PseudoVoigtModel, x, baseline_subtracted, guess, center_min)

    result_gauss = best_fit_gau + baseline
    result_loren = best_fit_lor + baseline
    result_pseudo = best_fit_pse + baseline
    return result_gauss, result_loren, result_pseudo


def plot_single_model(x, y, model_result, model_str):
    """Plot a single model with the error.

    Args:
        x (array): 1D
        y (array): 1D, original data
        model_result (array): 1D, best fitting result
        model_str (string): name of the model
    """
    plt.title(f"{model_str} Result",weight='bold')
    plt.rcParams.update({'font.weight': 'bold'})
    plt.plot(x, y, '--', c='k', label='Original Data', linewidth=1.0)
    plt.plot(x, model_result, '-', label='{}'.format(model_str), linewidth=1.0)
    plt.plot(x, abs(model_result-y), '-', label='Error', linewidth=1.0)
    plt.legend()
    plt.xlabel('2-theta',weight='bold')
    plt.ylabel('Intensity',weight='bold')
    check_output_dir('output_figures')
    plt.savefig('output_figures/Comparison_{}.png'.format(model_str))
    # plt.show()
    plt.close()


def tabulate_comparison(data_2d, dir_path='output_files'):
    """Store the comparison between 3 models into a .csv file

    Args:
        data_2d (array): 2D. Each cols correspond to x, original y, y for Gaussian model,
                        y for Lorentzian model, y for Pseudo-Voigt model. 
        dir_path (str, optional): the directory to store the file. Defaults to 'output_files'.
    """
    check_output_dir(dir_path)
    header = ['X', 'Y_Original', 'Y_Gaussian',
              'Y_Lorentzian', 'Y_Pseudo_Voigt']
    with open(f"{dir_path}/Comparison.csv", "w", encoding="utf-8") as fp:
        wr = csv.writer(fp, quoting=csv.QUOTE_ALL)
        wr.writerow(header)
        for line in np.array(data_2d).transpose():
            wr.writerow(np.around(line, 6))


def fit_index(data_2d):
    """Calculate the fit index for each model (Chi-Square).

    Args:
        data_2d (array): 2D. Each cols correspond to x, original y, y for Gaussian model,
                        y for Lorentzian model, y for Pseudo-Voigt model. 

    Returns:
        array: 1D, fit indices for each model
    """
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


def summarize_comparison(x, y, x_range, guess, center_min=0.0, dir_path='output_files'):
    """A summary of the comparison between 3 models, including tabulating data and plotting figure. 

    Args:
        x (array): 1D
        y (array): 1D
        x_range (array): shape of (1,2), including lower and upper bound
        guess (array): 2D array. Columns correspond to: center, sigma and amplitude. 
        center_min (float, optional): lower bound of center. Defaults to 0.0.
        dir_path (str, optional): the directory to store the file. Defaults to 'output_files'.
    """
    validate_x_range(x, x_range)
    interval_x, interval_y = get_interval_data(x, y, x_range)
    result_gauss, result_loren, result_voigt = compare_models(
        interval_x, interval_y, guess, center_min)
    data_2d = np.zeros((5, len(interval_x)))  # time + pars_value + pars_stderr
    data_2d[0] = interval_x
    data_2d[1] = interval_y
    data_2d[2] = result_gauss
    data_2d[3] = result_loren
    data_2d[4] = result_voigt

    tabulate_comparison(data_2d, dir_path)
    fit_index(data_2d)
    plot_single_model(interval_x, interval_y, result_gauss, 'Gaussian')
    plot_single_model(interval_x, interval_y,
                             result_loren, 'Lorentzian')
    plot_single_model(interval_x, interval_y,
                             result_voigt, 'PseudoVoigt')


def form_mesopore(data_3d): 
    extent = data_3d[1][1]/data_3d[0][1]
    print(f"\nThe extent of mesopore formation is: \n{extent}. \n")
    return extent