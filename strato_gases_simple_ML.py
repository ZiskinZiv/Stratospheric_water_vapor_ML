#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 08:54:32 2021
OK so far:
    swoosh h2o: 1994-2019 30S to 30N mean, 82 hpa
    regressors:
    QBO_CDAS = +5 months lag correlated with h2o: 0.508
    Anom_nino3p4 = no lags corr with h2o: -0.167
    LR:
    no CV does R2 of 0.2857
    Cross validate 5 kfolds: mean R2: 0.1786 std R2: 0.245
    SVM:
    CV 5 kfolds: mean R2: 0.418, mean adj_R2: 0.408,
    std R2: 0.047, std adj_R2: 0.0485
    need to plot residuals with best model.
@author: shlomi
"""
from strat_paths import work_chaim
ml_path = work_chaim / 'ML'


def plot_repeated_kfold_dist(df, model_dict, X, y):
    import seaborn as sns
    sns.set_theme(style='ticks', font_scale=1.5)
    in_sample_r2 = {}
    for model_name, model in model_dict.items():
        model.fit(X, y)
        in_sample_r2[model_name] = model.score(X, y)
    print(in_sample_r2)
    df_melted = df.T.melt(var_name='model', value_name=r'R$^2$')
    fg = sns.displot(data=df_melted, x=r'R$^2$', col="model",
                     kind="hist", col_wrap=2, hue='model', stat='density',
                     kde=True)
    for ax in fg.axes:
        label = ax.title.get_text()
        model = label.split('=')[-1].strip()
        mean = df.T.mean().loc[model]
        std = df.T.std().loc[model]
        median = df.T.median().loc[model]
        in_sample = in_sample_r2[model]
        textstr = '\n'.join((
            r'$\mathrm{mean}=%.2f$' % (mean, ),
            r'$\mathrm{median}=%.2f$' % (median, ),
            r'$\mathrm{std}=%.2f$' % (std, ),
            r'in sample result$=%.2f$' % (in_sample, )))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=14,
                verticalalignment='top', bbox=props)
    fg.fig.suptitle('Out of sample testing models comparison')
    fg.fig.subplots_adjust(top=0.916)
    return fg


def assemble_cvr_dataframe(path=ml_path, score='test_r2', n_splits=5):
    import pandas as pd
    lr, lr_model = cross_validate_using_optimized_HP(
        path, model='LR', n_splits=n_splits)
    svm, svm_model = cross_validate_using_optimized_HP(
        path, model='SVM', n_splits=n_splits)
    rf, rf_model = cross_validate_using_optimized_HP(
        path, model='RF', n_splits=n_splits)
    mlp, mlp_model = cross_validate_using_optimized_HP(
        path, model='MLP', n_splits=n_splits)
    df = pd.DataFrame([lr[score], svm[score], rf[score], mlp[score]])
    df.index = ['MLR', 'SVM', 'RF', 'MLP']
    len_cols = len(df.columns)
    df.columns = ['kfold_{}'.format(x+1) for x in range(len_cols)]
    model_dict = {'MLR': lr_model, 'RF': rf_model,
                  'SVM': svm_model, 'MLP': mlp_model}
    return df, model_dict


def cross_validate_using_optimized_HP(path=ml_path, model='SVM', n_splits=5,
                                      n_repeats=20,
                                      scorers=['r2', 'r2_adj',
                                               'neg_mean_squared_error',
                                               'explained_variance']):
    from sklearn.model_selection import cross_validate
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.model_selection import KFold
    from sklearn.model_selection import RepeatedKFold
    from sklearn.metrics import make_scorer
    scores_dict = {s: s for s in scorers}
    if 'r2_adj' in scorers:
        scores_dict['r2_adj'] = make_scorer(r2_adj_score)
    if model != 'LR':
        hp_params = get_HP_params_from_optimized_model(path, model)
    ml = ML_Classifier_Switcher()
    ml_model = ml.pick_model(model_name=model)
    if model != 'LR':
        ml_model.set_params(**hp_params)
    print(ml_model)
    X = produce_X()
    y = produce_y()
    X = X.sel(time=slice('1994', '2019'))
    y = y.sel(time=slice('1994', '2019'))
    # cv = TimeSeriesSplit(5)
    # cv = KFold(10, shuffle=True, random_state=1)
    cv = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats,
                       random_state=1)
    cvr = cross_validate(ml_model, X, y, scoring=scores_dict, cv=cv)
    return cvr, ml_model


def get_HP_params_from_optimized_model(path=ml_path, model='SVM'):
    import joblib
    from aux_functions_strat import path_glob
    files = path_glob(path, 'GRSRCHCV_*.pkl')
    file = [x for x in files if model in x.as_posix()][0]
    gr = joblib.load(file)
    df = read_one_gridsearchcv_object(gr)
    return df.iloc[0][:-2].to_dict()


def produce_X(regressors=['qbo_cdas', 'anom_nino3p4'],
              lag={'qbo_cdas': 5}):
    from make_regressors import load_all_regressors
    ds = load_all_regressors()
    ds = ds[regressors].dropna('time')
    if lag is not None:
        for key, value in lag.items():
            print(key, value)
            ds[key] = ds[key].shift(time=value)
    X = ds.dropna('time').to_array('regressor')
    X = X.transpose('time', 'regressor')
    return X


def produce_y(path=work_chaim, detrend='lowess',
              sw_var='combinedeqfillanomfillh2oq', filename='swoosh_latpress-2.5deg.nc',
              lat_mean=[-30, 30], plevel=82, deseason='std'):
    import xarray as xr
    from aux_functions_strat import lat_mean
    from aux_functions_strat import detrend_ts
    from aux_functions_strat import anomalize_xr
    file = path / filename
    da = xr.open_dataset(file)[sw_var]
    if plevel is not None:
        da = da.sel(level=plevel, method='nearest')
    if lat_mean is not None:
        da = lat_mean(da)
    if detrend is not None:
        if detrend == 'lowess':
            da = detrend_ts(da)
    if deseason is not None:
        da = anomalize_xr(da, freq='MS', units=deseason)
    y = da
    return y


def r2_adj_score(y_true, y_pred, **kwargs):
    from sklearn.metrics import r2_score
    r2 = r2_score(y_true, y_pred)
    n = len(y_true)
    if 'p' in kwargs:
        p = kwargs['p']
    else:
        p = 2
    r2_adj = 1.0 - (1.0 - r2) * (n - 1.0) / (n - p)
    # r2_adj = 1-(1-r2)*(n-1)/(n-p-1)
    return r2_adj


def single_cross_validation(X_val, y_val, model_name='SVM',
                            n_splits=5, scorers=['r2', 'r2_adj',
                                                 'neg_mean_squared_error',
                                                 'explained_variance'],
                            seed=42, savepath=None, verbose=0,
                            param_grid='dense', n_jobs=-1):
    # from sklearn.model_selection import cross_validate
    from sklearn.model_selection import StratifiedKFold
    from sklearn.model_selection import KFold
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.model_selection import GridSearchCV
    # from sklearn.model_selection import train_test_split
    from sklearn.metrics import make_scorer
    # from string import digits
    # import numpy as np
    # import xarray as xr
    scores_dict = {s: s for s in scorers}
    if 'r2_adj' in scorers:
        scores_dict['r2_adj'] = make_scorer(r2_adj_score)

    X = X_val.dropna('time').sel(time=y_val['time'])
    y = y_val

    # if param_grid == 'light':
    #     print(np.unique(X.feature.values))

    # configure the cross-validation procedure
    # cv = TimeSeriesSplit(n_splits=n_splits)
    cv = KFold(n_splits=n_splits, random_state=seed, shuffle=True)
    # print('CV TimeSeriesKfolds of {}.'.format(n_splits))
    print('CV KFold of {}.'.format(n_splits))
    # define the model and search space:

    ml = ML_Classifier_Switcher()
    print('param grid group is set to {}.'.format(param_grid))
    # if outer_split == '1-1':
    #     cv_type = 'holdout'
    #     print('holdout cv is selected.')
    # else:
    #     cv_type = 'nested'
    #     print('nested cv {} out of {}.'.format(
    #         outer_split.split('-')[0], outer_split.split('-')[1]))
    sk_model = ml.pick_model(model_name, pgrid=param_grid)
    search_space = ml.param_grid
    # define search
    gr_search = GridSearchCV(estimator=sk_model, param_grid=search_space,
                             cv=cv, n_jobs=n_jobs,
                             scoring=scores_dict,
                             verbose=verbose,
                             refit=False, return_train_score=True)

    gr_search.fit(X, y)
    features = [x for x in X['regressor'].values]
    if savepath is not None:
        filename = 'GRSRCHCV_{}_{}_{}_{}_{}_{}.pkl'.format(model_name, '+'.join(features), '+'.join(
            scorers), n_splits,
            param_grid, seed)
        save_gridsearchcv_object(gr_search, savepath, filename)
    return gr_search


def save_gridsearchcv_object(GridSearchCV, savepath, filename):
    import joblib
    print('{} was saved to {}'.format(filename, savepath))
    joblib.dump(GridSearchCV, savepath / filename)
    return


def load_one_gridsearchcv_object(path=ml_path, model_name='SVM', verbose=True):
    """load one gridsearchcv obj with model_name and features and run read_one_gridsearchcv_object"""
    from aux_functions_strat import path_glob
    import joblib
    # first filter for model name:
    if verbose:
        print('loading GridsearchCVs results for {} model'.format(model_name))
    model_files = path_glob(path, 'GRSRCHCV_*.pkl')
    model_files = [x for x in model_files if model_name in x.as_posix()]
    # now select features:
    # if verbose:
    #     print('loading GridsearchCVs results with {} features'.format(features))
    # model_features = [x.as_posix().split('/')[-1].split('_')[3] for x in model_files]
    # feat_ind = get_feature_set_from_list(model_features, features)
    # also get the test ratio and seed number:
    # if len(feat_ind) > 1:
    #     if verbose:
    #         print('found {} GR objects.'.format(len(feat_ind)))
    #     files = sorted([model_files[x] for x in feat_ind])
    #     outer_splits = [x.as_posix().split('/')[-1].split('.')[0].split('_')[-3] for x in files]
    #     grs = [joblib.load(x) for x in files]
    #     best_dfs = [read_one_gridsearchcv_object(x) for x in grs]
    #     di = dict(zip(outer_splits, best_dfs))
    #     return di
    # else:
        # file = model_files[feat_ind]
        # seed = file.as_posix().split('/')[-1].split('.')[0].split('_')[-1]
        # outer_splits = file.as_posix().split('/')[-1].split('.')[0].split('_')[-3]
    # load and produce best_df:
    gr = joblib.load(model_files[0])
    best_df = read_one_gridsearchcv_object(gr)
    return best_df


def read_one_gridsearchcv_object(gr):
    """read one gridsearchcv multimetric object and
    get the best params, best mean/std scores"""
    import pandas as pd
    # first get all the scorers used:
    scorers = [x for x in gr.scorer_.keys()]
    # now loop over the scorers:
    best_params = []
    best_mean_scores = []
    best_std_scores = []
    for scorer in scorers:
        df_mean = pd.concat([pd.DataFrame(gr.cv_results_["params"]), pd.DataFrame(
            gr.cv_results_["mean_test_{}".format(scorer)], columns=[scorer])], axis=1)
        df_std = pd.concat([pd.DataFrame(gr.cv_results_["params"]), pd.DataFrame(
            gr.cv_results_["std_test_{}".format(scorer)], columns=[scorer])], axis=1)
        # best index = highest score:
        best_ind = df_mean[scorer].idxmax()
        best_mean_scores.append(df_mean.iloc[best_ind][scorer])
        best_std_scores.append(df_std.iloc[best_ind][scorer])
        best_params.append(df_mean.iloc[best_ind].to_frame().T.iloc[:, :-1])
    best_df = pd.concat(best_params)
    best_df['mean_score'] = best_mean_scores
    best_df['std_score'] = best_std_scores
    best_df.index = scorers
    return best_df


def order_of_mag(minimal=-5, maximal=1):
    import numpy as np
    return [10**float(x) for x in np.arange(minimal, maximal + 1)]


class ML_Classifier_Switcher(object):

    def pick_model(self, model_name, pgrid='normal'):
        """Dispatch method"""
        # from sklearn.model_selection import GridSearchCV
        self.param_grid = None
        method_name = str(model_name)
        # Get the method from 'self'. Default to a lambda.
        method = getattr(self, method_name, lambda: "Invalid ML Model")
#        if gridsearch:
#            return(GridSearchCV(method(), self.param_grid, n_jobs=-1,
#                                return_train_score=True))
#        else:
        # Call the method as we return it
        # whether to select lighter param grid, e.g., for testing purposes.
        self.pgrid = pgrid
        return method()

    def SVM(self):
        from sklearn.svm import SVR
        import numpy as np
        if self.pgrid == 'light':
            self.param_grid = {'kernel': ['poly'],
                               'C': [0.1],
                               'gamma': [0.0001],
                               'degree': [1, 2],
                               'coef0': [1, 4]}
        # elif self.pgrid == 'normal':
        #     self.param_grid = {'kernel': ['rbf', 'sigmoid', 'linear', 'poly'],
        #                        'C': order_of_mag(-1, 2),
        #                        'gamma': order_of_mag(-5, 0),
        #                        'degree': [1, 2, 3, 4, 5],
        #                        'coef0': [0, 1, 2, 3, 4]}
        elif self.pgrid == 'dense':
            # self.param_grid = {'kernel': ['rbf', 'sigmoid', 'linear', 'poly'],
            #                    'C': np.logspace(-2, 2, 10), # order_of_mag(-2, 2),
            #                    'gamma': np.logspace(-5, 1, 14), # order_of_mag(-5, 0),
            #                    'degree': [1, 2, 3, 4, 5],
            #                    'coef0': [0, 1, 2, 3, 4]}
            self.param_grid = {'kernel': ['rbf', 'sigmoid', 'linear'],
                               'C': np.logspace(-2, 2, 10), # order_of_mag(-2, 2),
                                'gamma': np.logspace(-5, 1, 14)}#, # order_of_mag(-5, 0),
                               # 'degree': [1, 2, 3, 4, 5],
                               # 'coef0': [0, 1, 2, 3, 4]}
        return SVR()

    def MLP(self):
        import numpy as np
        from sklearn.neural_network import MLPRegressor
        if self.pgrid == 'light':
            self.param_grid = {
                'activation': [
                    'identity',
                    'relu'],
                'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50)]}
        # elif self.pgrid == 'normal':
        #     self.param_grid = {'alpha': order_of_mag(-5, 1),
        #                        'activation': ['identity', 'logistic', 'tanh', 'relu'],
        #                        'hidden_layer_sizes': [(50, 50, 50), (50, 100, 50), (100,)],
        #                        'learning_rate': ['constant', 'adaptive'],
        #                        'solver': ['adam', 'lbfgs', 'sgd']}
        elif self.pgrid == 'dense':
            self.param_grid = {'alpha': np.logspace(-5, 1, 7),
                               'activation': ['identity', 'logistic', 'tanh', 'relu'],
                               'hidden_layer_sizes': [(10, 10, 10), (10, 20, 10), (10,), (5,), (1,)],
                               'learning_rate': ['constant', 'adaptive'],
                               'solver': ['adam', 'lbfgs', 'sgd']}
            #(1,),(2,),(3,),(4,),(5,),(6,),(7,),(8,),(9,),(10,),(11,), (12,),(13,),(14,),(15,),(16,),(17,),(18,),(19,),(20,),(21,)
        return MLPRegressor(random_state=42, max_iter=500, learning_rate_init=0.1)

    def RF(self):
        from sklearn.ensemble import RandomForestRegressor
        # import numpy as np
        if self.pgrid == 'light':
            self.param_grid = {'max_features': ['auto', 'sqrt']}
        elif self.pgrid == 'normal':
            self.param_grid = {'max_depth': [5, 10, 25, 50, 100],
                               'max_features': ['auto', 'sqrt'],
                               'min_samples_leaf': [1, 2, 5, 10],
                               'min_samples_split': [2, 5, 15, 50],
                               'n_estimators': [100, 300, 700, 1200]
                               }
        elif self.pgrid == 'dense':
            self.param_grid = {'max_depth': [5, 10, 25, 50, 100, 150, 250],
                               'max_features': ['auto', 'sqrt'],
                               'min_samples_leaf': [1, 2, 5, 10, 15, 25],
                               'min_samples_split': [2, 5, 15, 30, 50, 70, 100],
                               'n_estimators': [100, 200, 300, 500, 700, 1000, 1300, 1500]
                               }
        return RandomForestRegressor(random_state=42, n_jobs=-1)

    def LR(self):
        from sklearn.linear_model import LinearRegression
        return LinearRegression(n_jobs=-1)
