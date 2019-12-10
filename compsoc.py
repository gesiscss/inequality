def ba_model_nx(n, c, G=None, Seed=None):
    from random import seed
    if Seed is not None:
        seed(Seed)
    from networkx import Graph
    if G == None:
        G = Graph()
        G.add_edge(0, 1)
        source = 2
    else:
        source = len(G)
    from compsoc import sample_target_vertex
    while source < n:
        target = sample_target_vertex(c, G)
        G.add_edge(target, source)
        source += 1
    return G

def bin_pdf(a, bins=24):
    import numpy as np
    decades = np.ceil(np.log10(max(a[:, 0]))-np.log10(min(a[:, 0])))
    bin_min = np.log10(min(a[:, 0]))
    a_bin = np.full((bins, 3), np.nan)
    for i in range(0, bins):
        b = a[(a[:, 0] >= round(10**(bin_min+i*decades/bins))) & (a[:, 0] < round(10**(bin_min+(i+1)*decades/bins)))]
        if len(b) > 0:
            a_bin[i, 0] = np.mean(b[:, 0])
            a_bin[i, 1] = sum(b[:, 1])
            a_bin[i, 2] = round(10**(bin_min+(i+1)*decades/bins))-round(10**(bin_min+i*decades/bins))
    a_bin = a_bin[~np.isnan(a_bin[:, 0])]
    a_bin = np.column_stack((a_bin, (a_bin[:, 1]/sum(a[:, 1]))/a_bin[:, 2]))
    return a_bin

def compare_functions(f):
    function = ['exponential', 'stretched_exponential', 'lognormal', 'lognormal_positive', 'power_law', 'truncated_power_law']
    from numpy import zeros
    f_compare_R = zeros((6, 6), dtype=float)
    f_compare_p_R = zeros((6, 6), dtype=float)
    for i in range(0, 6):
        for j in range(0, 6):
            R, p_R = f.distribution_compare(function[i], function[j])
            f_compare_R[i, j] = R
            f_compare_p_R[i, j] = p_R
    from pandas import DataFrame
    return DataFrame(f_compare_R, index=function, columns=function), DataFrame(f_compare_p_R, index=function, columns=function)

def fit_power_law(l, discrete, xmin=None, fit=None, sims=2500, bootstrap=1000, data_original=False, markersize=12, linewidth=2, fontsize=18, marker=0, color=0, xlabel='x', title='', legend=True, letter='', Pdf=None, png=None):
    import matplotlib.pyplot as plt
    from compsoc import pdf, bin_pdf, p_value, compare_functions
    from numpy import ceil, log10, logspace, mean, std, nan
    from pandas import DataFrame, MultiIndex
    from powerlaw import Fit
    # Fit functions to data
    f = Fit(l, discrete=discrete, xmin=xmin)
    # Identify best fit
    a = pdf(l)
    a_bin = bin_pdf(a)
    if fit == None:
        from compsoc import plot_pdf
        fit_analytical = []
        if f.exponential.D < 0.1:
            fit_analytical.append('exponential')
        if f.stretched_exponential.D < 0.1:
            fit_analytical.append('stretched_exponential')
        if f.lognormal.D < 0.1:
            fit_analytical.append('lognormal')
        if f.lognormal_positive.D < 0.1:
            fit_analytical.append('lognormal_positive')
        if f.power_law.D < 0.1:
            fit_analytical.append('power_law')
        if f.truncated_power_law.D < 0.1:
            fit_analytical.append('truncated_power_law')
        plot_pdf(a_bin=a_bin, f=f, fit=fit_analytical)
        print('Kolmogorov-Smirnov distances')
        print('----------------------------')
        print('Exponential: D =', round(f.exponential.D, 2))
        print('Stretched exponential: D =', round(f.stretched_exponential.D, 2))
        print('Lognormal: D =', round(f.lognormal.D, 2))
        print('Lognormal (positive): D =', round(f.lognormal_positive.D, 2))
        print('Power law: D =', round(f.power_law.D, 2))
        print('Truncated power law: D =', round(f.truncated_power_law.D, 2))
        print('')
        if sims != None:
            print('Plausibility of power-law fit')
            print('-----------------------------')
            p = p_value(f=f, sims=sims)
            print('Plausibility: p =', round(p, 2))
            print('')
        print('Identify best fit')
        print('-----------------')
        R, p_R = compare_functions(f)
        print('')
        print('Identify best fit: Log-likelihood ratios')
        print('----------------------------------------')
        print(R)
        print('')
        print('Identify best fit: Significance')
        print('-------------------------------')
        print(p_R)
        print('')
        print('Plotting')
        print('--------')
        Input = int(input('Which function should be plotted (0: exponential, 1: stretched_exponential, 2: lognormal, 3: lognormal_positive, 4: power_law, 5: truncated_power_law)? '))
        #print('')
        fit = ['exponential', 'stretched_exponential', 'lognormal', 'lognormal_positive', 'power_law', 'truncated_power_law'][Input]
    else:
        if sims != None:
            p = p_value(f=f, sims=sims)
        R, p_R = compare_functions(f)
    # Uncertainty of power-law fit parameters
    if bootstrap != None:
        from compsoc import uncertainty
        Uncertainty = uncertainty(l=l, discrete=discrete, xmin=xmin, bootstrap=bootstrap)
    # Plotting
    shape = ['o', 'v', 'h', '^', 'p', '<', 'D', '>', 's', 'd']
    color_full = ['#000000', '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999']
    color_pale = ['#7f7f7f', '#f18c8d', '#9bbedb', '#a6d7a4', '#cba6d1', '#ffbf7f', '#ffff99', '#d2aa93', '#fbc0df', '#cccccc']
    fig = plt.figure(figsize=(4, 4))
    ax1 = fig.add_subplot(111)
    if data_original == True:
        ax1.plot(a[:, 0], a[:, 2], marker=shape[marker], markersize=markersize, color=color_pale[color], ls='')
    ax1.plot(a_bin[:, 0], a_bin[:, 3], marker=shape[marker], markersize=markersize, color=color_full[color], ls='')
    space_xmin = logspace(log10(f.xmin), log10(max(f.data_original)), 100)
    scale = f.n_tail/len(f.data_original)
    if fit == 'exponential':
        ax1.plot(space_xmin, scale*f.exponential.pdf(space_xmin), color='k', ls='-', lw=linewidth, label='Exponential')
    if fit == 'stretched_exponential':
        ax1.plot(space_xmin, scale*f.stretched_exponential.pdf(space_xmin), color='k', ls='-', lw=linewidth, label='Stretched Exponential')
    if fit == 'lognormal':
        ax1.plot(space_xmin, scale*f.lognormal.pdf(space_xmin), color='k', ls='-', lw=linewidth, label='Lognormal')
    if fit == 'lognormal_positive':
        ax1.plot(space_xmin, scale*f.lognormal_positive.pdf(space_xmin), color='k', ls='-', lw=linewidth, label='Lognormal ($\mu>0$)')
    if fit == 'power_law':
        precision = int(ceil(abs(log10(f.power_law.sigma))))
        if precision == 0:
            ax1.plot(space_xmin, scale*f.power_law.pdf(space_xmin), color='k', ls='-', lw=linewidth, label='Power Law\n$\hat{\\alpha}={%.0f}$' %f.power_law.alpha)
        if precision == 1:
            ax1.plot(space_xmin, scale*f.power_law.pdf(space_xmin), color='k', ls='-', lw=linewidth, label='Power Law\n$\hat{\\alpha}={%.1f}$' %f.power_law.alpha)
        if precision == 2:
            ax1.plot(space_xmin, scale*f.power_law.pdf(space_xmin), color='k', ls='-', lw=linewidth, label='Power Law\n$\hat{\\alpha}={%.2f}$' %f.power_law.alpha)
        if precision == 3:
            ax1.plot(space_xmin, scale*f.power_law.pdf(space_xmin), color='k', ls='-', lw=linewidth, label='Power Law\n$\hat{\\alpha}={%.3f}$' %f.power_law.alpha)
        if precision >= 4:
            ax1.plot(space_xmin, scale*f.power_law.pdf(space_xmin), color='k', ls='-', lw=linewidth, label='Power Law\n$\hat{\\alpha}={%.4f}$' %f.power_law.alpha)
    if fit == 'truncated_power_law':
        ax1.plot(space_xmin, scale*f.truncated_power_law.pdf(space_xmin), color='k', ls='-', lw=linewidth, label='Truncated Power Law')
    if f.xmin > min(f.data_original):
        ax1.axvline(f.xmin, color='k', ls='--', lw=linewidth, label='$\hat{'+xlabel+'}_{\mathrm{min}}={%.0f}$' %f.xmin if xmin == None else '$'+xlabel+'_{\mathrm{min}}={%.0f}$' %f.xmin)
    ax1.xaxis.set_ticks_position('both')
    ax1.yaxis.set_ticks_position('both')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('$'+xlabel+'$', fontsize=fontsize)
    ax1.set_ylabel('$p('+xlabel+')$', fontsize=fontsize)
    ax1.set_title(title, fontsize=fontsize)
    ax1.tick_params(axis="x", which='major', direction="in", width=linewidth, size=4*linewidth, labelsize=fontsize, pad=7)
    ax1.tick_params(axis="x", which='minor', direction="in", width=linewidth, size=2*linewidth, labelsize=fontsize, pad=7)
    ax1.tick_params(axis="y", which='major', direction="in", width=linewidth, size=4*linewidth, labelsize=fontsize)
    ax1.tick_params(axis="y", which='minor', direction="in", width=linewidth, size=2*linewidth, labelsize=fontsize)
    ax1.spines['left'].set_linewidth(linewidth)
    ax1.spines['right'].set_linewidth(linewidth)
    ax1.spines['bottom'].set_linewidth(linewidth)
    ax1.spines['top'].set_linewidth(linewidth)
    if legend == True:
        ax1.legend(fontsize=fontsize*3/4)
    plt.gcf().text(0., 0.9, letter, fontsize=fontsize*2)
    plt.subplots_adjust(left=0.2, right=0.95, bottom=0.15, top=0.9)
    plt.show()
    if Pdf != None:
        fig.savefig(Pdf)
    if png != None:
        fig.savefig(png)
    # Parameters and test statistics
    parameters = DataFrame([[len(l), 
                             round(mean(l), 2), 
                             round(std(l), 2), 
                             max(l), 
                             int(round(f.xmin, 0)) if discrete == True else f.xmin, 
                             (int(round(Uncertainty['xmin_std'], 0)) if discrete == True else round(Uncertainty['xmin_std'], int(ceil(abs(log10(Uncertainty['xmin_std'])))))) if bootstrap != None else float('nan'), 
                             round(f.power_law.alpha, int(ceil(abs(log10(Uncertainty['alpha_std']))))) if bootstrap != None else round(f.power_law.alpha, int(ceil(abs(log10(f.power_law.sigma))))), 
                             round(Uncertainty['alpha_std'], int(ceil(abs(log10(Uncertainty['alpha_std']))))) if bootstrap != None else float('nan'), 
                             int(f.n_tail), 
                             int(round(Uncertainty['n_tail_std'], 0)) if bootstrap != None else float('nan'), 
                             round(p, 2) if sims != None else nan]], 
                           columns=['$n$', '$\\langle x\\rangle$', '$\sigma_x$', '$x_{\mathrm{max}}$', '$\hat{x}_{\mathrm{min}}$', '$\sigma_{\hat{x}_{\mathrm{min}}}$', '$\hat{\\alpha}$', '$\sigma_{\hat{\\alpha}}$', '$n_{\mathrm{tail}}$', '$\sigma_{n_{\mathrm{tail}}}$', '$p$'])
    def three(x): return round(x, 3-int(ceil(log10(abs(x)))))
    test_statistics = DataFrame([[three(R.iloc[4]['exponential']), 
                                  round(p_R.iloc[4]['exponential'], 2), 
                                  three(R.iloc[4]['stretched_exponential']), 
                                  round(p_R.iloc[4]['stretched_exponential'], 2), 
                                  three(R.iloc[4]['lognormal']), 
                                  round(p_R.iloc[4]['lognormal'], 2), 
                                  three(R.iloc[4]['lognormal_positive']), 
                                  round(p_R.iloc[4]['lognormal_positive'], 2), 
                                  three(R.iloc[4]['truncated_power_law']), 
                                  round(p_R.iloc[4]['truncated_power_law'], 2)]], 
                      columns=MultiIndex.from_product([['Exponential', 'Stretched  Exponential', 'Lognormal', 'Lognormal ($\mu>0$)', 'Truncated Power Law'], ['$\mathcal{R}$', '$p$']]))
    return parameters, test_statistics

def p_value(f, sims=2500):
    prob = f.n_tail/len(f.data_original)
    body = [x for x in f.data_original if x < f.xmin]
    l = []
    from random import random, sample
    from powerlaw import Fit, Power_Law
    for i in range(0, sims):
        x = []
        for j in range(0, len(f.data_original)):
            if random() <= prob:
                x.append(int(Power_Law(discrete=True, xmin=f.xmin, parameters=[f.power_law.alpha]).generate_random(1)))
            else:
                x.append(sample(body, 1)[0])
        x_fit = Fit(x, discrete=True).power_law
        l.append(x_fit.KS() > f.power_law.KS())
    p = sum(l)/sims
    return p

def pdf(l):
    import collections
    counter = collections.Counter(l)
    import numpy as np
    a = np.column_stack((list(counter.keys()), list(counter.values())))
    a = np.column_stack((a, a[:, 1]/sum(a[:, 1])))
    a = a[a[: ,0].argsort()]
    return a

def plot_pdf(a_bin, label=None, f=None, fit=[], xscale_log=True):
    import matplotlib.pyplot as plt
    plt.plot(a_bin[:, 0], a_bin[:, 3], marker='o', ls='', label=label)
    if f != None:
        from numpy import ceil, logspace, log10
        space_xmin = logspace(log10(f.xmin), log10(max(f.data_original)), 100)
        scale = f.n_tail/len(f.data_original)
        if 'exponential' in fit:
            plt.plot(space_xmin, scale*f.exponential.pdf(space_xmin), label='Exponential')
        if 'stretched_exponential' in fit:
            plt.plot(space_xmin, scale*f.stretched_exponential.pdf(space_xmin), label='Stretched Exponential')
        if 'lognormal' in fit:
            plt.plot(space_xmin, scale*f.lognormal.pdf(space_xmin), label='Lognormal')
        if 'lognormal_positive' in fit:
            plt.plot(space_xmin, scale*f.lognormal_positive.pdf(space_xmin), label='Lognormal ($\mu>0$)')
        if 'power_law' in fit:
            precision = int(ceil(abs(log10(f.power_law.sigma))))
            if precision == 0:
                plt.plot(space_xmin, scale*f.power_law.pdf(space_xmin), label='Power Law\n$\hat{\\alpha}={%.0f}$' %f.power_law.alpha)
            if precision == 1:
                plt.plot(space_xmin, scale*f.power_law.pdf(space_xmin), label='Power Law\n$\hat{\\alpha}={%.1f}$' %f.power_law.alpha)
            if precision == 2:
                plt.plot(space_xmin, scale*f.power_law.pdf(space_xmin), label='Power Law\n$\hat{\\alpha}={%.2f}$' %f.power_law.alpha)
            if precision == 3:
                plt.plot(space_xmin, scale*f.power_law.pdf(space_xmin), label='Power Law\n$\hat{\\alpha}={%.3f}$' %f.power_law.alpha)
            if precision >= 4:
                plt.plot(space_xmin, scale*f.power_law.pdf(space_xmin), label='Power Law\n$\hat{\\alpha}={%.4f}$' %f.power_law.alpha)
        if 'truncated_power_law' in fit:
            plt.plot(space_xmin, scale*f.truncated_power_law.pdf(space_xmin), label='Truncated Power Law')
        plt.axvline(f.xmin, ls='--', label='$\hat{x}_{\mathrm{min}}={%.0f}$' %f.xmin)
    plt.xlabel('$x$')
    plt.ylabel('$p(x)$')
    if xscale_log == True:
        plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.show()

def sample(d):
    from random import random
    r = random()*d[len(d)-1]
    i = 0
    while d[i] < r:
        i += 1
    return i

def sample_target_vertex(c, G):
    d = dict(G.degree())
    d[0] = d[0]**c
    for i in range(1, len(d)):
        d[i] = d[i-1]+d[i]**c
    from compsoc import sample
    target = sample(d)
    return target

def uncertainty(l, discrete=True, xmin=None, bootstrap=1000):
    from numpy import random, std
    from pandas import DataFrame, concat
    from powerlaw import Fit
    outputs = DataFrame(columns=['xmin_std', 'alpha_std', 'n_tail_std'])
    for i in range(0, bootstrap):
        l_bootstrap = random.choice(l, size=len(l), replace=True)
        f_bootstrap = Fit(l_bootstrap, discrete=discrete, xmin=xmin)
        output = DataFrame([[f_bootstrap.xmin, f_bootstrap.power_law.alpha, f_bootstrap.n_tail]], index=[i], columns=['xmin_std', 'alpha_std', 'n_tail_std'])
        outputs = concat([outputs, output])
    return std(outputs)
