def h_index(citations):
    if len(citations) == 0: return 0
    if len(citations) == 1: return 1
    citations = sorted(citations, reverse=True)
    h_ind = 0
    for i, elem in enumerate(citations):
        if i + 1 > elem:
            return i
        h_ind = i + 1
    return h_ind

def author_h_index_in_year_X(publications_citations_no_uncited, authors, year_x):
    combined_h = publications_citations_no_uncited[
        (publications_citations_no_uncited.year_cit < year_x) &
        (publications_citations_no_uncited.author.isin(authors))]
    combined_h = combined_h.groupby(['author', 'pub_id']).agg({'id1': 'count'}).reset_index()
    author_hind_at_year = combined_h.groupby('author').agg({'id1': h_index}).reset_index()
    author_hind_at_year['year_pub'] = year_x
    author_hind_at_year = author_hind_at_year.rename({'id1': 'h-index'}, axis='columns')
    return author_hind_at_year

def list_append(lst, item):
    lst.append(item)
    return lst

def get_last_consec(arr_diff):
    import numpy as np
    if arr_diff.size == 0: return 1
    last_ind = np.where(arr_diff > 1)[0]
    if last_ind.size == 0: return 15
    last_ind = last_ind[0]
    return sum(arr_diff[:last_ind])+1

def quantile_binary(quant):
    return quant == 1

def add_absolute_counts(counts, citations_year_auth_df, author_year_numPub_df):
    counts = counts.merge(author_year_numPub_df, on=['author', 'year'], how='left')
    counts['num_pub'] = counts['num_pub'].fillna(0)
    counts = counts.merge(citations_year_auth_df, on=['author', 'year'], how='left')
    counts['num_cit'] = counts['num_cit'].fillna(0)
    return counts

def add_cumulative_counts(counts, feature):
    counts = calculate_cumulative_for_authors(counts, feature)
    return counts

def calculate_cumulative_for_authors(data, criterion):
    # data - the dataframe containing author publications or citations data
    # criterion - 'num_pub' (or) 'num_cit'
    # Group years and associative data and calculates the cumulative value
    data = data.set_index('year').sort_index()
    import pandas as pd
    data['cum_'+criterion] = data.groupby(['author'])[criterion].transform(pd.Series.cumsum)
    data = data.reset_index()

    return data

def create_counts(features, citations_year_auth_df, author_year_numPub_df, start_years, CAREER_AGES):
    counts0 = features[['author', 'cohort']].copy()
    counts0 = counts0[counts0['cohort'].isin(start_years)]
    counts0['year'] = counts0['cohort'].apply(lambda x: [x + i for i in range(0, CAREER_AGES)])
    import pandas as pd
    counts = pd.DataFrame(counts0['year'].tolist(), index=counts0['author']).stack().reset_index(
        level=1, drop=True).reset_index(name='year')[['year', 'author']]
    counts = counts.merge(features[['author', 'cohort', 'gender']], on='author', how='inner')
    counts['career_age'] = counts['year'] - counts['cohort'] + 1
    counts['year'] = counts['year'].astype('int32')
    counts = add_absolute_counts(counts, citations_year_auth_df, author_year_numPub_df)
    counts = add_cumulative_counts(counts, 'num_cit')
    counts = add_cumulative_counts(counts, 'num_pub')
    return counts

def create_counts_win(base_df, publications_citations_no_uncited, WINDOW_SIZE, start_years, file_ext=''):
    counts = base_df.copy(deep=True)
    counts = add_citation_window_counts(base_df, publications_citations_no_uncited, WINDOW_SIZE, start_years)
    return counts

def get_start_years(START_YEAR, LAST_START_YEAR, features):
    all_years = features['cohort'].unique()
    start_years = [year for year in all_years if START_YEAR <= year <= LAST_START_YEAR]
    start_years = sorted(start_years)
    return start_years

def add_citation_window_counts(counts, combined_df, WINDOW_SIZE, start_years):
    shift = -(WINDOW_SIZE - 1)
    df_list = []
    for year in start_years:
        df_year = combined_df[combined_df.cohort == year]
        for y in range(year, year + 15 - WINDOW_SIZE +1):
            df_window = df_year[(df_year.year_pub >= y) & (df_year.year_pub < y + WINDOW_SIZE) &
                                (df_year.year_cit >= y) & 
                                (df_year.year_cit < df_year.year_pub + WINDOW_SIZE)]
            df_window = df_window.groupby('author').agg({'id1': 'count'}).reset_index()
            df_window['year'] = y
            df_window = df_window.rename({'id1': 'win_num_cit'}, axis=1)
            df_list.append(df_window)
    import pandas as pd
    df_cit_window = pd.concat(df_list).sort_values(by=['author', 'year'])
    counts = counts.merge(df_cit_window, on=['author', 'year'], how='left')
    counts['win_num_cit'] = counts['win_num_cit'].fillna(0)
    counts['win_num_pub'] = counts.groupby('author')['num_pub'].transform(
        lambda x: x.rolling(WINDOW_SIZE, min_periods=WINDOW_SIZE).sum().shift(shift))
    return counts

def pdf(l):
    import collections
    counter = collections.Counter(l)
    import numpy as np
    a = np.column_stack((list(counter.keys()), list(counter.values())))
    a = np.column_stack((a, a[:, 1]/sum(a[:, 1])))
    a = a[a[: ,0].argsort()]
    return a

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

def gini(array):
    import numpy as np
    array_copy = np.copy(array)
    #"""Calculate the Gini coefficient of a numpy array."""
    # based on bottom eq:
    # http://www.statsdirect.com/help/generatedimages/equations/equation154.svg
    # from:
    # http://www.statsdirect.com/help/default.htm#nonparametric_methods/gini.htm
    # All values are treated equally, arrays must be 1d:
    try:
        if np.min(array_copy) < 0:
            array_copy -= np.min(array_copy)
    except ValueError:
        return 0
    array_copy += 0.0000001
    array_copy = np.sort(array_copy)
    index = np.arange(1,array_copy.shape[0]+1)
    n = array_copy.shape[0]
    return ((np.sum((2 * index - n  - 1) * array_copy)) / (n * np.sum(array_copy)))

def cliffsD(ser1, ser2):
    import numpy as np
    np_1 = np.array(ser1, dtype=np.int8)
    np_2 = np.array(ser2, dtype=np.int8)
    return np.mean(np.sign(np_1[:, None] - np_2).mean(axis=1))

def mann_whitney_effect_size(a, b, alternative='two-sided', effect_formula='r'):
    n1 = len(a)
    n2 = len(b)
    from scipy.stats import mannwhitneyu
    import math
    statistic, pvalue = mannwhitneyu(a, b, alternative=alternative)
    z = (statistic-(n1*n2/2))/(math.sqrt(n1*n2*(n1+n2+1)/12))
    if effect_formula == 'r':
        effect = z/math.sqrt(n1+n2)
    elif effect_formula == 'eta':
        effect = (z**2)/(n1+n2-1)
    elif effect_formula == 'common_language':
        effect = statistic/(n1*n2)
    return effect, statistic, pvalue

def bin_bivariate(a, method, bins=100, bin_min=None):
    import numpy as np
    if method == 'vertical':
        a = a[:, :2]
        sel = 0
    if method == 'orthogonal':
        a = a[:, :3]
        sel = 2
    if bin_min == None:
        bin_min = min(a[:, sel])
    bin_min_log10 = np.log10(bin_min)
    bin_max = max(a[:, sel])
    bin_max_log10 = np.log10(bin_max)
    l_bins = np.logspace(bin_min_log10, bin_max_log10, bins+1)
    l_bins = list(np.round(l_bins, 0))
    a_bin = np.full((bins, 4), np.nan)
    for i in range(0, bins-1):
        b = a[(a[:, sel] >= l_bins[i]) & (a[:, sel] < l_bins[i+1])]
        if len(b) > 0:
            a_bin[i, 0] = np.mean(b[:, 0])
            a_bin[i, 1] = np.mean(b[:, 1])
            a_bin[i, 2] = np.std(b[:, 0])
            a_bin[i, 3] = np.std(b[:, 1])            
    i = bins-1
    b = a[a[:, sel] >= l_bins[i]]
    if len(b) > 0:
        a_bin[i, 0] = np.mean(b[:, 0])
        a_bin[i, 1] = np.mean(b[:, 1])
    a_bin = a_bin[~np.isnan(a_bin[:, 0])]
    return a_bin

def fit_scaling(a, method, beta0=1.):
    import numpy as np
    import sklearn.linear_model as sk_lm
    from scipy.odr import Model, Data, ODR, RealData
    from sklearn.metrics import mean_squared_error, r2_score
    a = a[:, :2]
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

def fitted_line(a, method, D, beta):
    import numpy as np
    a = a[:, :2]
    if method == 'vertical':
        xmin = min(a[:, 0])
        xmax = max(a[:, 0])
    if method == 'orthogonal':
        a = np.column_stack((a, np.round(a[:, 1]*a[:, 0]**(1/beta), 4)))
        a = np.unique(a, axis=0)
        a_min = a[a[:, 2] == min(a[:, 2])]
        a_max = a[a[:, 2] == max(a[:, 2])]
        xmin = (a_min[0, 1]*a_min[0, 0]**(1/beta)/D)**(1/(beta+1/beta))
        xmax = (a_max[0, 1]*a_max[0, 0]**(1/beta)/D)**(1/(beta+1/beta))
    a_fit = np.array([[xmin, D*xmin**beta], [xmax, D*xmax**beta]])
    return a_fit

def bootstrap_scaling(a, fit, reduction=None, beta0=1., straps=1000):
    import numpy as np
    import pandas as pd
    df_bootstrap = pd.DataFrame(columns=['D', 'beta', 'r2', 'reducedchi'])
    for i in range(0, straps):
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

def average_bivariate(a, method):
    import pandas as pd
    if method == 'vertical':
        a = a[:, :2]
        a_average = pd.DataFrame(a).groupby(0).mean().reset_index().values
    if method == 'orthogonal':
        a = a[:, :3]
        a_average = pd.DataFrame(a).groupby(2).mean().reset_index().values[:, 1:3]
    return a_average

def first_max_of_iterable(l):
    i = 0
    while True:
        if l[i+1] < l[i]:
            break
        i += 1
    return i

def fit_bivariate_dblp(x, y, fit, xlabel, ylabel, title, letter, reduction=None, beta0=1., bootstrap=True, straps=1000, estimate_lower_cutoff=False, color=0, marker=0, markersize = 9, linewidth=2, fontsize=18, raw_data=True, pdf=None, png=None):
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.metrics import r2_score 
    color_full = ['green', 'purple']
    color_pale = ['#7fbf7f', '#bf7fbf']
    shape = ['o', 'v', 'h', '^', 'p', '<', 'D', '>', 's', 'd']
    
    a = np.column_stack((x, y))
    a = a[~np.any(a == 0, axis=1)]
    a = a[a[:, 0].argsort()]
    a_unique = np.unique(a, axis=0)
    
    if estimate_lower_cutoff == False:
        xmin = np.nan
        a_cutoff = a
    else:
        l_xmin = sorted(set(a_unique[:, 0]))[:-19]
        l_r2 = []
        sel = 0
        for xmin in l_xmin:
            a_cutoff = a[a[:, sel] >= xmin]
            a_bin_vertical_fit = bin_bivariate(a_cutoff, method='vertical', bin_min=min(a_unique[:, 0]), bins=20)
            D_bin_vertical_ols, beta_bin_vertical_ols, r2_bin_vertical_ols, reducedchi_bin_vertical_ols = fit_scaling(a_bin_vertical_fit, method='ols')
            l_r2.append(r2_bin_vertical_ols)
        xmin = l_xmin[first_max_of_iterable(l_r2)]
        a_cutoff = a[a[:, sel] >= xmin]
    if fit == 'ols':
        if reduction == None:
            D, beta, r2, reducedchi = fit_scaling(a_cutoff, method='ols')
        if reduction == 'average':
            D, beta, r2, reducedchi = fit_scaling(average_bivariate(a_cutoff, method='vertical'), method='ols')
        if reduction == 'bin':
            D, beta, r2, reducedchi = fit_scaling(bin_bivariate(a_cutoff, method='vertical'), method='ols')
        a_fit = fitted_line(a_cutoff, method='vertical', D=D, beta=beta)
    if fit == 'odr':
        if reduction == None:
            D, beta, r2, reducedchi = fit_scaling(a, method='odr', beta0=beta0)
        if reduction == 'average':
            D, beta, r2, reducedchi = reduce_orthogonal_fit_scaling(a, method='average', beta0=beta0)
        if reduction == 'bin':
            D, beta, r2, reducedchi = reduce_orthogonal_fit_scaling(a, method='bin', beta0=beta0)
        a_fit = fitted_line(a, method='orthogonal', D=D, beta=beta)
    
    if bootstrap == True:
        df_bootstrap = bootstrap_scaling(a_cutoff, fit=fit, reduction=reduction, straps=straps)
    r2 = r2_score(a_cutoff[:, 1], D*a_cutoff[:, 0]**beta)
    
    if fit == 'ols':
        a_bin_plot = bin_bivariate(a, method='vertical', bins=20)
    if fit == 'odr':
        a = np.column_stack((a, a[:, 1]*a[:, 0]**(1/beta)))
        a_bin_plot = bin_bivariate(a, method='orthogonal', bins=20)
    
    fig = plt.figure(figsize=(4, 4))
    ax1 = fig.add_subplot(111)
    if raw_data == True:
        plt.plot(a[:, 0], a[:, 1], marker=shape[marker], markersize=markersize, color=color_pale[color], alpha=.1, ls='', markeredgewidth=0.0)
    ax1.plot(a_bin_plot[:, 0], a_bin_plot[:, 1], marker=shape[marker], markersize=markersize, color=color_full[color], ls='', markeredgewidth=0.0)
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
    
    if estimate_lower_cutoff == True:
        ax1.axvline(xmin, color='k', ls=':', lw=linewidth, label='$\hat{x}_{\mathrm{min}}=%.0f$' %xmin)
    
    ax1.xaxis.set_ticks_position('both')
    ax1.yaxis.set_ticks_position('both')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.set_xlabel('$'+xlabel+'$', fontsize=fontsize)
    ax1.set_ylabel('$'+ylabel+'$', fontsize=fontsize)
    ax1.set_title(title, fontsize=fontsize)
    ax1.tick_params(axis="x", which='major', direction="in", width=linewidth, size=4*linewidth, labelsize=fontsize)
    ax1.tick_params(axis="x", which='minor', direction="in", width=linewidth, size=2*linewidth, labelsize=fontsize)
    ax1.tick_params(axis="y", which='major', direction="in", width=linewidth, size=4*linewidth, labelsize=fontsize)
    ax1.tick_params(axis="y", which='minor', direction="in", width=linewidth, size=2*linewidth, labelsize=fontsize)
    ax1.spines['left'].set_linewidth(linewidth)
    ax1.spines['right'].set_linewidth(linewidth)
    ax1.spines['bottom'].set_linewidth(linewidth)
    ax1.spines['top'].set_linewidth(linewidth)
    ax1.legend(fontsize=fontsize-6, loc='upper left')
    plt.gcf().text(0., 0.9, letter, fontsize=fontsize*2)
    plt.subplots_adjust(left=0.25, right=0.95, bottom=0.2, top=0.9)
    if pdf != None:
        fig.savefig(pdf)
    if png != None:
        fig.savefig(png)
    plt.show()
    
    df_stats = pd.DataFrame([[len(a), xmin, D, beta, np.std(beta-df_bootstrap['beta']) if bootstrap == True else float('nan'), r2, reducedchi]], columns=['n', 'xmin', 'D', 'beta', 'beta_std', 'r2', 'reduced_chi'])
    return df_stats, df_bootstrap

def adjusted_r2(y_true, y_pred, num_feat):
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    num_sample = len(y_true)
    adj = 1 - float(num_sample-1)/(num_sample-num_feat-1)*(1 - r2)
    return adj

def scale_columns(X):
    import pandas as pd
    from sklearn.preprocessing import RobustScaler
    if len(X.columns) > 0:
        scaler = RobustScaler().fit(X)
        standardized_cols = scaler.transform(X)
    else: 
        standardized_cols = []
    return pd.DataFrame(standardized_cols, index=X.index, columns=X.columns)
