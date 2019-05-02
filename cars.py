# colors
#color_full = ['#000000', '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999']
#color_pale = ['#7f7f7f', '#f18c8d', '#9bbedb', '#a6d7a4', '#cba6d1', '#ffbf7f', '#ffff99', '#d2aa93', '#fbc0df', '#cccccc']

#def graphics():
#    # dependencies
#    import matplotlib as mpl
#    # function
#    linewidth = 2
#    # plot parameters
#    mpl.rcParams['xtick.major.size'] = 4*linewidth
#    mpl.rcParams['xtick.major.width'] = 1*linewidth
#    mpl.rcParams['xtick.minor.size'] = 2*linewidth
#    mpl.rcParams['xtick.minor.width'] = 1*linewidth
#    mpl.rcParams['ytick.major.size'] = 4*linewidth
#    mpl.rcParams['ytick.major.width'] = 1*linewidth
#    mpl.rcParams['ytick.minor.size'] = 2*linewidth
#    mpl.rcParams['ytick.minor.width'] = 1*linewidth
#    mpl.rcParams['xtick.direction'] = 'in'
#    mpl.rcParams['ytick.direction'] = 'in'
#    mpl.rcParams['axes.linewidth'] = linewidth
#    mpl.rcParams.update({'font.size': 24})
#    mpl.rcParams["font.sans-serif"] = "Arial"

def fact_matrix(df_selections, norm=True, sym=False):
    # dependencies
    import itertools
    import numpy as np
    import pandas as pd
    from scipy.sparse import csr_matrix, coo_matrix, triu
    from sklearn.preprocessing import normalize
    # function
    if {'transaction', 'fact', 'weight'}.issubset(df_selections.columns):
        def extract_vertices(df, *columns):
            l = [df[column].unique().tolist() for column in columns]
            return list(set(itertools.chain.from_iterable(l)))
        transactions = extract_vertices(df_selections, 'transaction')
        transactions_id = {value: i for i, value in enumerate(transactions)}
        facts = extract_vertices(df_selections, 'fact')
        facts_id = {value: i for i, value in enumerate(facts)}
        rows = [transactions_id[x] for x in df_selections['transaction'].values]
        columns = [facts_id[y] for y in df_selections['fact'].values]
        cells = df_selections['weight'].tolist()
        G = coo_matrix((cells, (rows, columns))).tocsr()
        GT = csr_matrix.transpose(G)
        if norm == False:
            H = GT*G
        else:
            GN = normalize(G, norm='l1', axis=1)
            H = GT*GN
        w = pd.Series(np.squeeze(np.array(H.sum(axis=1))))
        d = pd.Series(np.array(H.diagonal()))
        e = d/w
        H_nodiag = H.tolil()
        H_nodiag.setdiag(values=0)
        k = pd.Series(np.array([len(i) for i in H_nodiag.data.tolist()]))
        s = k/w
        facts = pd.Series(facts)
        facts = pd.concat([facts, k, w, d, e, s], axis=1)
        facts.columns = ['fact', 'degree', 'weight', 'selfselection', 'embeddedness', 'sociability']
        if sym == False:
            return H.tocsr(), facts
        else:
            return triu(HN.tocoo()).tocsr(), facts
    else:
        print('Dataframe is not a proper selection table.')

### fit_univariate

def fit_univariate(x, discrete, xmin, xlabel, title, bins=24, bootstrap=100, col=0, marker='o', markersize=12, linewidth=2, fontsize=24, unbinned_data=True, pdf=None, png=None):
    # dependencies
    import collections
    import igraph as ig
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import powerlaw as pl
    import warnings
    warnings.filterwarnings('ignore')
    # colors
    color_full = ['#000000', '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999']
    color_pale = ['#7f7f7f', '#f18c8d', '#9bbedb', '#a6d7a4', '#cba6d1', '#ffbf7f', '#ffff99', '#d2aa93', '#fbc0df', '#cccccc']
    # identify best fit
    function = ['exponential', 'stretched_exponential', 'lognormal_positive', 'power_law', 'truncated_power_law']
    xfit = pl.Fit(x, discrete=discrete, xmin=xmin)
    print('')
    xcurve = np.linspace(xfit.xmin, max(xfit.data_original), 1000)
    yscale = xfit.n_tail/len(xfit.data_original)
    p = ig.power_law_fit(x, xmin=xfit.xmin).p
    xfit.plot_pdf(linear_bins=False, original_data=True, marker=marker, ls='')
    if xfit.exponential.D < 0.1:
        plt.plot(xcurve, yscale*xfit.exponential.pdf(xcurve), label=function[0])
    if xfit.stretched_exponential.D < 0.1:
        plt.plot(xcurve, yscale*xfit.stretched_exponential.pdf(xcurve), label=function[1])
    if xfit.lognormal_positive.D < 0.1:
        plt.plot(xcurve, yscale*xfit.lognormal_positive.pdf(xcurve), label=function[2])
    if xfit.power_law.D < 0.1:
        plt.plot(xcurve, yscale*xfit.power_law.pdf(xcurve), label=function[3])
    if xfit.truncated_power_law.D < 0.1:
        plt.plot(xcurve, yscale*xfit.truncated_power_law.pdf(xcurve), label=function[4])
    plt.xlabel('$'+xlabel+'$')
    plt.ylabel('$PDF('+xlabel+')$')
    plt.legend()
    plt.show()
    print('')
    print('Kolmogorov-Smirnov goodness of fit')
    print('----------------------------------')
    print('exponential: D =', round(xfit.exponential.D, 2))
    print('stretched_exponential: D =', round(xfit.stretched_exponential.D, 2))
    print('lognormal_positive: D =', round(xfit.lognormal_positive.D, 2))
    print('power_law: D =', round(xfit.power_law.D, 2))
    print('truncated_power_law: D =', round(xfit.truncated_power_law.D, 2))
    print('')
    xfit_compare_R = np.zeros((5, 5), dtype=float)
    xfit_compare_p = np.zeros((5, 5), dtype=float)
    for i in range(0, 5):
        for j in range(0, 5):
            R, p = xfit.distribution_compare(function[i], function[j])
            xfit_compare_R[i, j] = R
            xfit_compare_p[i, j] = p
    print('')
    print('Loglikelihood ratios')
    print('--------------------')
    print(pd.DataFrame(xfit_compare_R, index=function, columns=function))
    print('')
    print('Significance')
    print('------------')
    print(pd.DataFrame(xfit_compare_p, index=function, columns=function))
    print('')
    print('Plausibility of power-law fit')
    print('-----------------------------')
    print('p =', p)
    print('')
    function_fit = int(input('Which function should be plotted (0: exponential, 1: stretched_exponential, 2: lognormal_positive, 3: power_law, 4: truncated_power_law)? '))
    print('')
    # get parameter uncertainty
    if bootstrap != None:
        print('Bootstrapping...')
        outputs = pd.DataFrame(columns=['x_mu', 'x_sigma', 'xmax', 'xmin', 'alpha', 'ntail'])
        for i in range(0, bootstrap):
            x_bootstrap = np.random.choice(x, size=len(x), replace=True)
            xfit_bootstrap = pl.Fit(x_bootstrap, discrete=True, xmin=None)
            output_bootstrap = pd.DataFrame([[np.mean(x_bootstrap), np.std(x_bootstrap), max(x_bootstrap), xfit_bootstrap.xmin, xfit_bootstrap.alpha, xfit_bootstrap.n_tail]], index=[i], columns=['<x>', 'sigma', 'xmax', 'xmin', 'alpha', 'ntail'])
            outputs = pd.concat([outputs, output_bootstrap])
        print('')
    # plot chosen fit
    counter = collections.Counter(xfit.data_original)
    a = np.column_stack((list(counter.keys()), list(counter.values())))
    a = np.column_stack((a, a[:, 1]/sum(a[:, 1])))
    a = a[a[: ,0].argsort()]
    decades = np.ceil(np.log10(max(a[:, 0])-min(a[:, 0])))
    bin_min = np.log10(min(a[:, 0]))
    a_binned = np.full((bins, 3), np.nan)
    for i in range(0, bins):
        b = a[(a[:, 0] >= round(10**(bin_min+i*decades/bins))) & (a[:, 0] < round(10**(bin_min+(i+1)*decades/bins)))]
        if len(b)>0:
            a_binned[i, 0] = np.mean(b[:, 0])
            a_binned[i, 1] = sum(b[:, 1])
            a_binned[i, 2] = round(10**(bin_min+(i+1)*decades/bins))-round(10**(bin_min+i*decades/bins))
    a_binned = a_binned[~np.isnan(a_binned[:, 0])]
    a_binned = np.column_stack((a_binned, (a_binned[:, 1]/sum(a[:, 1]))/a_binned[:, 2]))
    fig = plt.figure(figsize=(6, 6))
    ax1 = fig.add_subplot(111)
    if unbinned_data == True:
        ax1.plot(a[:, 0], a[:, 2], marker=marker, markersize=markersize, color=color_pale[col], ls='')
    ax1.plot(a_binned[:, 0], a_binned[:, 3], marker=marker, markersize=markersize, color=color_full[col], ls='')
    if function_fit == 0:
        ax1.plot(xcurve, yscale*xfit.exponential.pdf(xcurve), color='k', ls='-', lw=linewidth, label='Exponential')
    if function_fit == 1:
        ax1.plot(xcurve, yscale*xfit.stretched_exponential.pdf(xcurve), color='k', ls='-', lw=linewidth, label='Str. Exp.')
    if function_fit == 2:
        ax1.plot(xcurve, yscale*xfit.lognormal_positive.pdf(xcurve), color='k', ls='-', lw=linewidth, label='Lognormal')
    if function_fit == 3:
        ax1.plot(xcurve, yscale*xfit.power_law.pdf(xcurve), color='k', ls='-', lw=linewidth, label='Power Law')
    if function_fit == 4:
        ax1.plot(xcurve, yscale*xfit.truncated_power_law.pdf(xcurve), color='k', ls='-', lw=linewidth, label='Trunc. Power Law')
    ax1.axvline(xfit.power_law.xmin, color='k', ls='--', lw=linewidth, label='$\hat{'+xlabel+'}_{\mathrm{min}}={%.0f}$' %xfit.xmin)
    ax1.xaxis.set_ticks_position('both')
    ax1.yaxis.set_ticks_position('both')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('$'+xlabel+'$', fontsize=fontsize)
    ax1.set_ylabel('$PDF('+xlabel+')$', fontsize=fontsize)
    ax1.set_title(title, fontsize=fontsize)
    ax1.tick_params(axis="x", which='major', direction="in", width=linewidth, size=4*linewidth, labelsize=fontsize, pad=7)
    ax1.tick_params(axis="x", which='minor', direction="in", width=linewidth, size=2*linewidth, labelsize=fontsize, pad=7)
    ax1.tick_params(axis="y", which='major', direction="in", width=linewidth, size=4*linewidth, labelsize=fontsize)
    ax1.tick_params(axis="y", which='minor', direction="in", width=linewidth, size=2*linewidth, labelsize=fontsize)
    ax1.spines['left'].set_linewidth(linewidth)
    ax1.spines['right'].set_linewidth(linewidth)
    ax1.spines['bottom'].set_linewidth(linewidth)
    ax1.spines['top'].set_linewidth(linewidth)
    #ax1.legend(fontsize=fontsize/2)
    plt.subplots_adjust(left=0.25, right=0.95, bottom=0.15, top=0.85)
    plt.show()
    if pdf != None:
        fig.savefig(pdf)
    if png != None:
        fig.savefig(png)
    # generate output table
    if function_fit == 3:
        df = pd.DataFrame(
            [[len(x), 
              np.mean(x), 
              np.std(x), 
              max(x), 
              xfit.xmin, 
              np.std(outputs['xmin']) if bootstrap != None else float('nan'), 
              xfit.power_law.alpha, 
              np.std(outputs['alpha']) if bootstrap != None else float('nan'), 
              xfit.n_tail, 
              np.std(outputs['ntail']) if bootstrap != None else float('nan'), 
              p]], columns=['n', 'x_mu', 'x_sigma', 'xmax', 'xmin', 'xmin_std', 'alpha', 'alpha_std', 'ntail', 'ntail_std', 'p'])
    return df

### fit_bivariate

def fitted_line(a, method, D, beta):
    # dependencies
    import numpy as np
    # fitting
    a = a[:, :2] # removes previous selectors
    if method == 'vertical':
        x_min = min(a[:, 0])
        x_max = max(a[:, 0])
    if method == 'orthogonal':
        a = np.column_stack((a, np.round(a[:, 1]*a[:, 0]**(1/beta), 4))) # rounding removes artificial decimals from binary coding of integers
        a = np.unique(a, axis=0)
        a_min = a[a[:, 2] == min(a[:, 2])]
        a_max = a[a[:, 2] == max(a[:, 2])]
        x_min = (a_min[0, 1]*a_min[0, 0]**(1/beta)/D)**(1/(beta+1/beta))
        x_max = (a_max[0, 1]*a_max[0, 0]**(1/beta)/D)**(1/(beta+1/beta))
    #x_range = [x_min, x_max]
    #y_range_predict = [D*x_min**beta, D*x_max**beta]
    a_fit = np.array([[x_min, D*x_min**beta], [x_max, D*x_max**beta]])
    #return x_range, y_range_predict
    return a_fit

def fit_scaling(a, method, beta0=1.):
    #dependencies
    import numpy as np
    import sklearn.linear_model as sk_lm
    from scipy.odr import Model, Data, ODR, RealData
    from sklearn.metrics import mean_squared_error, r2_score
    # fitting
    a = a[:, :2] # removes previous selectors
    #a_log10 = np.log10(a)
    x_log10 = np.log10(a[:, 0])
    y_log10 = np.log10(a[:, 1])
    if method == 'ols':
        x_log10 = x_log10.reshape(len(x_log10), 1)
        y_log10 = y_log10.reshape(len(y_log10), 1)
        reg = sk_lm.LinearRegression()
        reg.fit(x_log10, y_log10)
        D = 10**reg.intercept_[0]
        beta = reg.coef_[0][0]
        y_log10_predict = reg.predict(x_log10)
        r2 = r2_score(y_log10, y_log10_predict)
        reducedchi = float('nan')
    if method == 'odr':
        def f(B, x):
            return B[0]*x+B[1]
        linear = Model(f)
        reg = ODR(RealData(x_log10, y_log10), linear, beta0=[beta0-0.5, beta0+0.5])
        reg_fit = reg.run()
        D = 10**reg_fit.beta[1]
        beta = reg_fit.beta[0]
        r2 = float('nan')
        reducedchi = reg_fit.res_var
    return D, beta, r2, reducedchi

def average_bivariate(a, method):
    # dependencies
    import pandas as pd
    # averaging
    if method == 'vertical':
        a = a[:, :2]
        a_average = pd.DataFrame(a).groupby(0).mean().reset_index().values
        #a_log10_mean = pd.DataFrame(a_log10).groupby(0).mean().reset_index().values
    if method == 'orthogonal':
        a = a[:, :3]
        a_average = pd.DataFrame(a).groupby(2).mean().reset_index().values[:, 1:3]
    return a_average

def bin_bivariate(a, method, bins=100):
    # dependencies
    import numpy as np
    # binning
    if method == 'vertical':
        a = a[:, :2]
        sel = 0
    if method == 'orthogonal':
        a = a[:, :3]
        sel = 2
    #logbin_min = min(np.log10(a[:, sel]))
    #logbin_max = max(np.log10(a[:, sel]))
    #decades = logbin_max-logbin_min
    l_bins = np.logspace(min(np.log10(a[:, sel])), max(np.log10(a[:, sel])), bins+1)
    l_bins = list(np.round(l_bins, 0))
    a_bin = np.full((bins, 2), np.nan)
    for i in range(0, bins-1):
        #xmin = logbin_min+i*decades/bins
        #xmax = logbin_min+(i+1)*decades/bins
        #b = a[(a[:, sel] >= 10**xmin) & (a[:, sel] < 10**xmax)]
        b = a[(a[:, sel] >= l_bins[i]) & (a[:, sel] < l_bins[i+1])]
        if len(b) > 0:
            a_bin[i, 0] = np.mean(b[:, 0])
            a_bin[i, 1] = np.mean(b[:, 1])
    i = bins-1
    #xmin = logbin_min+i*decades/bins
    #xmax = logbin_min+(i+1)*decades/bins
    #b = a[a[:, sel] >= 10**xmin]
    b = a[a[:, sel] >= l_bins[i]]
    if len(b) > 0:
        a_bin[i, 0] = np.mean(b[:, 0])
        a_bin[i, 1] = np.mean(b[:, 1])
    a_bin = a_bin[~np.isnan(a_bin[:, 0])]
    return a_bin

def reduce_orthogonal_fit_scaling(a, method=['average', 'bin'], beta0=1.):
    # dependencies
    import numpy as np
    # reduction and fitting
    beta_orthogonal_odr = beta0
    conv = 0.
    i = 0
    while beta_orthogonal_odr != conv:
        conv = beta_orthogonal_odr
        a = a[:, :2]
        a = np.column_stack((a, a[:, 1]*a[:, 0]**(1/beta_orthogonal_odr)))
        if method == 'average':
            a_orthogonal = average_bivariate(a, method='orthogonal')
        if method == 'bin':
            a_orthogonal = bin_bivariate(a, method='orthogonal')
        D_orthogonal_odr, beta_orthogonal_odr, r2_orthogonal_odr, reducedchi_orthogonal_odr = fit_scaling(a_orthogonal, method='odr')
        i += 1
        if i == 10:
            break
    if beta_orthogonal_odr != conv:
        beta10 = np.mean([beta_orthogonal_odr, conv])
        a = a[:, :2]
        a = np.column_stack((a, a[:, 1]*a[:, 0]**(1/beta10)))
        if method == 'average':
            a_orthogonal = average_bivariate(a, method='orthogonal')
        if method == 'bin':
            a_orthogonal = bin_bivariate(a, method='orthogonal')
        D_orthogonal_odr, beta_orthogonal_odr, r2_orthogonal_odr, reducedchi_orthogonal_odr = fit_scaling(a_orthogonal, method='odr')
    return D_orthogonal_odr, beta_orthogonal_odr, r2_orthogonal_odr, reducedchi_orthogonal_odr

def bootstrap_scaling(a, fit, reduction=None, beta0=1.):
    # dependencies
    import numpy as np
    import pandas as pd
    # bootstrapping
    df_bootstrap = pd.DataFrame(columns=['D', 'beta', 'r2', 'reducedchi'])
    for i in range(0, 100):
        a_sample = a[np.random.choice(range(0, len(a)), size=len(a), replace=True), :]
        if fit == 'ols':
            if reduction == None:
                D_sample, beta_sample, r2_sample, reducedchi_sample = fit_scaling(a_sample, method='ols')
            if reduction == 'average':
                D_sample, beta_sample, r2_sample, reducedchi_sample = fit_scaling(average_bivariate(a_sample, method='vertical'), method='ols')
            if reduction == 'bin':
                D_sample, beta_sample, r2_sample, reducedchi_sample = fit_scaling(bin_bivariate(a_sample, method='vertical'), method='ols')
        if fit == 'odr':
            if reduction == None:
                D_sample, beta_sample, r2_sample, reducedchi_sample = fit_scaling(a_sample, method='odr', beta0=beta0)
            if reduction == 'average':
                D_sample, beta_sample, r2_sample, reducedchi_sample = reduce_orthogonal_fit_scaling(a_sample, method='average', beta0=beta0)
            if reduction == 'bin':
                D_sample, beta_sample, r2_sample, reducedchi_sample = reduce_orthogonal_fit_scaling(a_sample, method='bin', beta0=beta0)
        parameters = pd.DataFrame([[D_sample, beta_sample, r2_sample, reducedchi_sample]], index=[i], columns=['D', 'beta', 'r2', 'reducedchi'])
        df_bootstrap = pd.concat([df_bootstrap, parameters])
    return df_bootstrap

def fit_bivariate(x, y, fit, xlabel, ylabel, title, reduction=None, beta0=1., bootstrap=True, color=0, marker=0, markersize=12, linewidth=2, fontsize=24, raw_data=True, pdf=None, png=None):
    # dependencies
    from cars import fitted_line, fit_scaling, average_bivariate, bin_bivariate, reduce_orthogonal_fit_scaling, bootstrap_scaling
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    # colors
    color_full = ['#000000', '#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#ffff33', '#a65628', '#f781bf', '#999999']
    color_pale = ['#7f7f7f', '#f18c8d', '#9bbedb', '#a6d7a4', '#cba6d1', '#ffbf7f', '#ffff99', '#d2aa93', '#fbc0df', '#cccccc']
    
    # shapes
    shape = ['o', 'v', 'h', '^', 'p', '<', 'D', '>', 's', 'd']
    
    # preprocess data
    a = np.column_stack((x, y))
    a = a[~np.any(a == 0, axis=1)]
    a = a[a[:, 0].argsort()]
    a_unique = np.unique(a, axis=0)
    
    # fit data
    if fit == 'ols':
        if reduction == None:
            D, beta, r2, reducedchi = fit_scaling(a, method='ols')
        if reduction == 'average':
            D, beta, r2, reducedchi = fit_scaling(average_bivariate(a, method='vertical'), method='ols')
        if reduction == 'bin':
            D, beta, r2, reducedchi = fit_scaling(bin_bivariate(a, method='vertical'), method='ols')
        a_fit = fitted_line(a, method='vertical', D=D, beta=beta)
    if fit == 'odr':
        if reduction == None:
            D, beta, r2, reducedchi = fit_scaling(a, method='odr', beta0=beta0)
        if reduction == 'average':
            D, beta, r2, reducedchi = reduce_orthogonal_fit_scaling(a, method='average', beta0=beta0)
        if reduction == 'bin':
            D, beta, r2, reducedchi = reduce_orthogonal_fit_scaling(a, method='bin', beta0=beta0)
        a_fit = fitted_line(a, method='orthogonal', D=D, beta=beta)
    
    # bootstrap data
    if bootstrap == True:
        df_bootstrap = bootstrap_scaling(a, fit=fit, reduction=reduction)
    
    # bin data for plotting
    if fit == 'ols':
        a_bin_plot = bin_bivariate(a, method='vertical', bins=24)
    if fit == 'odr':
        a = np.column_stack((a, a[:, 1]*a[:, 0]**(1/beta)))
        a_bin_plot = bin_bivariate(a, method='orthogonal', bins=24)
    
    # plot data and fit
    fig = plt.figure(figsize=(6, 6))
    ax1 = fig.add_subplot(111)
    if raw_data == True:
        plt.plot(a_unique[:, 0], a_unique[:, 1], marker=shape[marker], markersize=markersize, color=color_pale[color], ls='')
    ax1.plot(a_bin_plot[:, 0], a_bin_plot[:, 1], marker=shape[marker], markersize=markersize, color=color_full[color], ls='')
    if bootstrap == True:
        certainty = int(np.ceil(abs(np.log10(np.std(beta-df_bootstrap['beta'])))))
        if certainty == 0:
            a_fit_label = '$'+ylabel+'\propto '+xlabel+'^{%.0f}$' %beta
        if certainty == 1:
            a_fit_label = '$'+ylabel+'\propto '+xlabel+'^{%.1f}$' %beta
        if certainty == 2:
            a_fit_label = '$'+ylabel+'\propto '+xlabel+'^{%.2f}$' %beta
        if certainty == 3:
            a_fit_label = '$'+ylabel+'\propto '+xlabel+'^{%.3f}$' %beta
        if certainty >= 4:
            a_fit_label = '$'+ylabel+'\propto '+xlabel+'^{%.4f}$' %beta
    else:
        a_fit_label = '$'+ylabel+'\propto '+xlabel+'^{%.2f}$' %beta
    ax1.plot(a_fit[:, 0], a_fit[:, 1], color='k', ls='-', lw=linewidth, label=a_fit_label)
    ax1.xaxis.set_ticks_position('both')
    ax1.yaxis.set_ticks_position('both')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('$'+xlabel+'$', fontsize=fontsize)
    ax1.set_ylabel('$'+ylabel+'$', fontsize=fontsize)
    ax1.set_title(title, fontsize=fontsize)
    ax1.tick_params(axis="x", which='major', direction="in", width=linewidth, size=4*linewidth, labelsize=fontsize, pad=7)
    ax1.tick_params(axis="x", which='minor', direction="in", width=linewidth, size=2*linewidth, labelsize=fontsize, pad=7)
    ax1.tick_params(axis="y", which='major', direction="in", width=linewidth, size=4*linewidth, labelsize=fontsize)
    ax1.tick_params(axis="y", which='minor', direction="in", width=linewidth, size=2*linewidth, labelsize=fontsize)
    ax1.spines['left'].set_linewidth(linewidth)
    ax1.spines['right'].set_linewidth(linewidth)
    ax1.spines['bottom'].set_linewidth(linewidth)
    ax1.spines['top'].set_linewidth(linewidth)
    ax1.legend(fontsize=fontsize*2/3)
    plt.subplots_adjust(left=0.25, right=0.95, bottom=0.15, top=0.85)
    if pdf != None:
        fig.savefig(pdf)
    if png != None:
        fig.savefig(png)
    
    # create output
    df_stats = pd.DataFrame([[len(a), D, beta, np.std(beta-df_bootstrap['beta']) if bootstrap == True else float('nan'), r2, reducedchi]], columns=['n', 'D', 'beta', 'beta_std', 'r2', 'reduced_chi'])
    return df_stats, df_bootstrap

### avalanches

def avalanches(df_selections, fact, time):
    # dependencies
    import pandas as pd
    # function
    df_selections.sort_values(by=[fact, time], inplace=True)
    df_selections.reset_index(drop=True, inplace=True)
    l_avalanches = []
    avalanche = None
    time_to = None
    time_from = 0
    size = 0
    for _, row in df_selections.iterrows():
        # if symbol is new or sequence is interrupted
        if (row[fact] != avalanche or row[time] - time_to > 1):
            # then write avalanche to list ...
            l_avalanches.append((avalanche, time_from, time_to, size))
            # ... and initiate new avalanche
            time_from = row[time]
            avalanche = row[fact]
            size = 0
        avalanche = row[fact]
        time_to = row[time]
        size += row['weight']
    l_avalanches.append((avalanche, time_from, time_to, size))
    l_avalanches.pop(0)
    df_avalanches = pd.DataFrame(l_avalanches, columns=['avalanche', 'time_from', 'time_to', 'size'])
    df_avalanches['size'] = pd.Series(df_avalanches['size'])
    df_avalanches['duration'] = df_avalanches['time_to'] - df_avalanches['time_from'] + 1
    df_avalanches['frequency'] = 1/df_avalanches['duration']
    return df_avalanches

### generative models

# dependencies

import networkx as nx
import random as rd

# functions

def compute_zeta(b, n):
    zeta = 1
    for i in range(2, n):
        zeta += 1/i**b
    return zeta

def sample(L):
    r = abs(1-rd.random())*L[len(L)-1]
    i = 0
    while L[i] < r:
        i += 1
    return i

def sample_starting_node(a, G):
    L = dict(G.degree())
    L[0] = L[0]**a
    for i in range(1, len(L)):
        L[i] = L[i-1] + L[i]**a
    i = sample(L)
    return i

def sample_distance(b, zeta, n):
    L = [0]*n
    L[0] = 2**(-b)
    for i in range(1, n-1):
        L[i] = L[i-1] + (i+1)**(-b)
    L[n-1] = zeta
    i = sample(L)
    return i + 2

def unused_degree(G, unvisited, visited):
    unused_deg = []
    for k in unvisited:
        j = 0
        for i in G.neighbors(k):
            if not(i in visited): j += 1
        unused_deg.append(j)
    return unused_deg

def del_zeros_from_lists(L1, L2):
    temp1 = []
    temp2 = []
    for i in range(0, len(L1)):
        if L2[i] != 0:
            temp1.append(L1[i])
            temp2.append(L2[i])
    return temp1, temp2

def unvisited_neighbors(neighbors, visited):
    unvisited = []
    for i in neighbors:
        if not(i in visited):
            unvisited.append(i)
    return unvisited

def traversal(G, s, distance, g, source):
    visited = [s]
    i = 1
    stamp = True
    while (i <= distance) and (stamp == True):
        neighbors = G.neighbors(s)
        unvisited = unvisited_neighbors(neighbors, visited)
        if i == distance:
            # !!! special condition !!!
            # unvisited nodes are not neighbors of starting node (visited[0])
            unvisited = unvisited_neighbors(unvisited, G.neighbors(visited[0]))
        if unvisited != []:
            # calculate unused degrees of unvisited neighbors
            unused_deg = unused_degree(G, unvisited, visited)
            (unvisited, unused_deg) = del_zeros_from_lists(unvisited, unused_deg)
            if unused_deg != []:
                # sample from unused neighbors
                # according to unused degree and 'g'
                unused_deg[0] = unused_deg[0]**g
                for j in range(1, len(unvisited)):
                    unused_deg[j] = unused_deg[j-1] + unused_deg[j]**g
                k = sample(unused_deg)
                s = unvisited[k]
                # append the chosen node
                visited.append(s)
                i += 1
            else: stamp = False
        else: stamp = False
    if i > distance:
        # add new edge
        print('search successful:')
        print('adding new edge =', (visited[0], s))
        G.add_edge(visited[0], s, weight=1)
        # increase weight along search path
        print('increasing weights along search path =', visited)
        for l in range(0, len(visited)-2):
            G[visited[l]][visited[l+1]]['weight'] += 1
        print('.......................................................................')
        return True
    else:
        # add new node and edge
        print('search unsuccessful: adding new node and edge =', (visited[0], source))
        G.add_edge(visited[0], source, weight=1)
        return False

# generators

def simulate_feedback(a, b, g, n, seed=None):
    if n < 1:
        print('error: graph must have n>1, n=%d'%n)
    else:
        zeta = compute_zeta(b, 100000)
        if seed is not None:
            rd.seed(seed)
        G = nx.Graph(name='feedback')
        G.add_edge(0, 1, weight=1) # add initial 2 nodes with an edge
        source = 2 # next node is 2
        while source < n: # now add the other n-2 nodes
            print('source =', source)
            traverse = False
            s = sample_starting_node(a, G)
            print('starting node =', s)
            distance = sample_distance(b, zeta, len(G)-1)
            print('distance =', distance)
            if distance > len(G)-1:
                # create a new node and connect it to starting node
                print('distance >= graph: connecting new node to starting node =', (s, source))
                G.add_edge(s, source, weight=1)
            else:
                # run traversal algorithm (to find path)
                # add new edge or new node and edge
                print('distance < graph: traversing...')
                traverse = traversal(G, s, distance, g, source)
            if not traverse:
                source += 1
                print('.......................................................................')
    return G

def simulate_ba(a, n, seed=None):
    if n < 1:
        print('error: graph must have n>1, n=%d'%n)
    else:
        if seed is not None:
            rd.seed(seed)
        G = nx.Graph(name='ba')
        G.add_edge(0, 1) # add initial 2 nodes with an edge
        source = 2 # next node is 2
        while source < n: # now add the other n-2 nodes
            s = sample_starting_node(a, G)
            G.add_edge(s, source)
            source += 1
        return G

###

