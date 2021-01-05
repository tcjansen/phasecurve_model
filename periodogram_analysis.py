from scipy.signal import argrelmax
import numpy as np
import matplotlib.pyplot as plt
import os
import csv
import multiprocessing
from functools import partial

def get_tics():
    rv_ticsdir = os.getcwd() + '/tics'
    toi_ticsdir = rv_ticsdir + '/all-shortP-tois'

    toi_tics = [each for each in os.listdir(toi_ticsdir)
                if os.path.isfile(toi_ticsdir + '/' + each + '/ls_periodogram.txt') and
                not each.startswith('.') and
                not each.startswith('n')]
    rv_tics = [each for each in os.listdir(rv_ticsdir)
               if os.path.isfile(rv_ticsdir + '/' + each + '/ls_periodogram.txt') and
               not each.startswith('.') and
               not each.startswith('n') and
               not each.startswith('a')]
    return toi_ticsdir, toi_tics, rv_ticsdir, rv_tics


def remove_lessthan2orbits_tois(toi_info_file):
    tics, norbits = np.genfromtxt(toi_info_file, delimiter=',', unpack=True, usecols=(1, 41), skip_header=1)

    with open('remove_tois.sh', 'w') as w:
        for i in range(len(tics)):
            if norbits[i] <= 2:
                w.write('echo `rm -r ./tics/all-shortP-tois/' + str(int(tics[i])) + '`\n')

    return


def get_mindist(powers, periods, period, center0=False):

    maxima_i = argrelmax(powers)[0]
    maxima = powers[maxima_i]
    maxima_dist = (periods[maxima_i] - period) / period

    # pick the closest maxima to the transit period, store this distance
    if center0:
        absdist = abs(maxima_dist)
        return maxima_dist[absdist == np.amin(absdist)]
    else:
        return np.amin(abs(maxima_dist))


def dists_to_maxima(tics, ticsdir, tag='periodogram', center0=False):
    hist = []
    for tic in tics:
        ticdir = ticsdir + '/' + str(tic)
        periods, powers = np.genfromtxt(ticdir + '/ls_' + tag + '.txt', unpack=True)
        period = np.genfromtxt(ticdir + '/period.txt')

        # find the maxima and their period "distance" to the transit period in units of their orbital period
        hist += [get_mindist(powers, periods, period, center0=center0)]

    return hist


def get_dists_to_maxima(tag='periodogram', center0=False, yscale='log'):

    toi_ticsdir, toi_tics, rv_ticsdir, rv_tics = get_tics()

    tois = dists_to_maxima(toi_tics, toi_ticsdir, tag=tag, center0=center0)
    rvs = dists_to_maxima(rv_tics, rv_ticsdir, tag='periodogram', center0=center0)
    # plot a histogram of these values for each group
    rangemin = np.minimum(min(tois), min(rvs))
    rangemax = np.maximum(max(tois), max(rvs))
    rangemax = 0.5
    bins = np.linspace(rangemin, rangemax, 10)

    if center0:
        title = '(highest power period - measured period) / measured period'
        legendloc = 'upper left'
    else:
        title = '|highest power period - measured period| / measured period'
        legendloc = 'upper right'

    plt.figure()
    plt.hist([rvs, tois], label=['RV objects', 'TOIs'], density=True, align='mid', edgecolor='white')
    # plt.hist(rvs, label='rvs', density=False, align='mid', edgecolor='white')
    plt.ylabel('probability density')
    plt.xlabel(title)
    plt.legend(fontsize='large', loc=legendloc)
    plt.yscale(yscale)
    plt.show()


def id_harmonics(tic, ticsdir, tag='periodogram', plot_harmonics=False,
                 plot_harmonic_range=False):
    """ Compare LS-periodogram power of the orbital period harmonics to the power
    of the given orbital period from transits. Compared powers are equal to the average
    of the powers within 10% of the measured orbital period (this is just to compensate
    for the frequency binning, which may not line up exactly with the given
    orbital period.)
    """

    harmonics = np.array([1/3, 1/2, 1, 2, 3])
    tolerance = 0.05

    ticdir = ticsdir + '/' + str(tic)
    periods, powers = np.genfromtxt(ticdir + '/ls_' + tag + '.txt', unpack=True)
    period = np.genfromtxt(ticdir + '/period.txt')
    toi_harmonics = harmonics * period

    if plot_harmonics:
        fig, ax = plt.subplots(figsize=(10,5))
        plt.title('TIC ' + str(tic) + ', period = ' + str(np.round(period, 2)) + ' days')
        plt.plot(periods, powers, color='black')
        plt.axvline(period, color='red', ls='--')
        plt.axvline(toi_harmonics[0], color='red', ls=':')
        for line in toi_harmonics[1:]:
            plt.axvline(line, color='red', ls=':')
        plt.xlim(np.min(periods) - 0.1 * np.min(periods), np.max(periods))
        plt.xscale('log')
        plt.xlabel('period [days]')
        plt.ylabel(r'$\chi^2_{null} - \chi^2$')
        plt.savefig(ticdir + '/' + tag + '_harmonics.png')
        plt.close(fig)

    peak_powers = np.array([])  # power of harmonic / power of "fundamental"
    for h in toi_harmonics:
        grab = (periods >= h * (1 - tolerance)) & (periods < h * (1 + tolerance))
        mean_bic = np.mean(powers[grab])
        if h == period:
            fundamental_bic = np.mean(powers[grab])
        peak_powers = np.append(peak_powers, np.mean(powers[grab]))

    result = np.array(peak_powers)
    relative_powers = result / fundamental_bic  # the index of the fundamental

    if plot_harmonic_range:
        fig, ax = plt.subplots(figsize=(10,5))
        plt.plot(periods, powers, color='black')
        yfill = np.arange(0, max(powers))
        for h in toi_harmonics:
            if h == period:
                ls='--'
            else:
                ls=':'
            plt.axvline(h, ls=ls, color='red')
            plt.fill_betweenx(yfill, h - tolerance * h, h + tolerance * h,
                              facecolor='#B552EE', alpha=0.5)
        plt.xscale('log')
        plt.title('TIC ' + str(tic))
        plt.xlabel('period [days]')
        plt.ylabel(r'$\chi^2_{null} - \chi^2$')
        plt.savefig(ticdir + '/' + tag + '_harmonic-ranges.png')
        plt.close(fig)

    return result


def compare_dist(toi_ticsdir, toi_tics, rv_ticsdir, rv_tics,
              tag='periodogram', center0=False):
    toi_ticperiods = []
    toi_mindists = []
    for tic in toi_tics:
        ticdir = toi_ticsdir + '/' + str(tic)
        periods, powers = np.genfromtxt(ticdir + '/ls_' + tag + '.txt', unpack=True)
        period = np.genfromtxt(ticdir + '/period.txt')
        toi_ticperiods += [period]
        toi_mindists += [get_mindist(powers, periods, period, center0=center0)]

    rv_ticperiods = []
    rv_mindists = []
    for tic in rv_tics:
        ticdir = rv_ticsdir + '/' + str(tic)
        periods, powers = np.genfromtxt(ticdir + '/ls_' + tag + '.txt', unpack=True)
        period = np.genfromtxt(ticdir + '/period.txt')
        rv_ticperiods += [period]
        rv_mindists += [get_mindist(powers, periods, period, center0=center0)]

    fig, ax = plt.subplots()
    plt.scatter(toi_ticperiods, toi_mindists, color='#02338f', alpha=0.8, label='TOIs')
    plt.scatter(rv_ticperiods, rv_mindists, color='#13b9e3', alpha=0.8, label='RV objects')
    plt.ylabel('|highest power period - measured period| / measured period')
    plt.xlabel('measured period [days]')
    plt.xscale('log')
    plt.yscale('log')
    plt.legend(loc='lower right')
    plt.show()
    plt.close(fig)


def check_for_harmonic_signals(tics, ticsdir, bicfile, tolerance=0.025,
                               tag='periodogram'):
    # write results to a file with columns:
    # tic, 1st harmonic (fundamental), 2nd harmonic, 3rd harmonic
    # then if there's a peak within 5% of where the harmonics should be, AND
    # if BIC >= 10, return 1
    # then plot in bar chart
    harmonics = np.array([1/3, 1/2, 1, 2, 3])
    results = []
    for tic in tics:
        ticdir = ticsdir + '/' + str(tic)
        periods, powers = np.genfromtxt(ticdir + '/ls_' +
                                        tag + '.txt', unpack=True)
        period = np.genfromtxt(ticdir + '/period.txt')
        tic_harmonics = harmonics * period
        maxima_i = argrelmax(powers)[0]

        filerow = [float(tic)]
        for h in tic_harmonics:
            grab = ((periods >= h * (1 - tolerance)) &
                   (periods < h * (1 + tolerance)))
            period_range = periods[grab]

            if len(period_range) == 0:  # harmonic is beyond sample range
                filerow += [np.nan]
                continue

            lower_P_bound = min(period_range)
            upper_P_bound = max(period_range)
            maxima_in_range = ((periods[maxima_i] >= lower_P_bound) &
                               (periods[maxima_i] <= upper_P_bound))
            mean_bic = np.mean(powers[grab])

            if maxima_i[maxima_in_range].any() and mean_bic >= 10:
                filerow += [round(mean_bic, 9)]
            else:
                filerow += [0.000000000]

        results += [np.array(filerow)]


    with open(bicfile, 'w') as w:
        w.write('tic, 1/3, 1/2, 1, 2, 3\n')
        np.savetxt(w, np.array(results))
        w.close()


def get_harmonic_percentages(tics, bicfile):
    harmonics = np.array([1/3, 1/2, 1, 2, 3])
    total_tics = len(tics)
    harmonic_percentages = []
    results_by_harmonic = np.genfromtxt(bicfile)
    results_by_harmonic = results_by_harmonic.T[1:]
    for i in range(len(harmonics)):
        nonzero = results_by_harmonic[i] > 0
        harmonic_percentages += [len(results_by_harmonic[i][nonzero]) /
                                 total_tics]
    return np.array(harmonic_percentages) * 100


def plot_harmonic_percentages(toi_tics, rv_tics, toi_bicfile, rv_bicfile):
    toi_percents = get_harmonic_percentages(toi_tics, toi_bicfile)
    rv_percents = get_harmonic_percentages(rv_tics, rv_bicfile)

    fig, ax = plt.subplots()
    labels = ['1/3', '1/2', '1', '2', '3']
    x = np.arange(len(labels))
    width = 0.35
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    plt.bar(x - width/2,
            rv_percents, width, color='#13b9e3',
            label='RV objects')
    plt.bar(x + width/2,
            toi_percents, width,
            color='#02338f', label='TOIs')
    plt.xlabel('period of detected signal [in units of measured P]')
    plt.ylabel(r'$\%$ of TICs')
    plt.legend()
    plt.show()
    plt.close(fig)




if __name__ == '__main__':

    toi_ticsdir, toi_tics, rv_ticsdir, rv_tics = get_tics()

    # remove_lessthan2orbits_tois('toi-catalog_single-pl-sys-nsec_sample-spoc.csv')

    # dists_to_maxima(rv_tics, rv_ticsdir, tag='periodogram', center0=False)

    # get_dists_to_maxima(tag='periodogram', center0=True, yscale='log')

    # compare_dist(toi_ticsdir, toi_tics, rv_ticsdir, rv_tics)

    # id_harmonics(rv_tics, rv_ticsdir, tag='periodogram', plot_harmonics=True, plot_harmonic_range=False)

    # check_for_harmonic_signals(rv_tics, rv_ticsdir, 'RV_harmonics_result_BICs.txt')

    plot_harmonic_percentages(toi_tics, rv_tics, 'TOI_harmonics_result_BICs.txt',
                              'RV_harmonics_result_BICs.txt')

    # pool = multiprocessing.Pool(processes=3)
    # wrapperfunc = partial(id_harmonics, ticsdir=toi_ticsdir, tag='periodogram',
    #                       plot_harmonics=True, plot_harmonic_range=True)
    # pool.map(wrapperfunc, toi_tics)
    # pool.close()
    # pool.join()

