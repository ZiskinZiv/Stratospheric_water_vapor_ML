#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 13:20:30 2019
1) run LR with qbo_1 and qbo_2 and ch4, do validation test hold one out
2) multioutput to gett cross_val_scores on cv=10 or cv=400
2) send the figs to Shawn davis, ask about 1984-1987
@author: shlomi
"""
from sklearn_xarray import RegressorWrapper


class parameters:

    def __init__(self,
                 model_name='LR',
                 season='all',
                 regressors_file='Regressors.nc',
                 swoosh_field='combinedanomfillanom',
                 regressors=None,   # default None means all regressors
                 reg_except=None,
                 poly_features=None,
                 time_shift=None,
                 data_name='swoosh',
                 species='h2o',
                 time_period=None,
                 area_mean=False,
                 original_data_file='swoosh_latpress-2.5deg.nc'):
        import sys
        self.filing_order = ['data_name', 'field', 'model_name', 'season',
                             'reg_selection', 'special_run']
        self.delimeter = '_'
        self.model_name = model_name
        self.season = season
        self.time_shift = time_shift
        self.poly_features = poly_features
        self.reg_except = reg_except
        self.regressors = regressors
        self.reg_selection = 'R1'   # All
        self.regressors_file = regressors_file
#        self.sw_field_list = ['combinedanomfillanom', 'combinedanomfill',
#                              'combinedanom', 'combinedeqfillanom',
#                              'combinedeqfill', 'combinedeqfillseas',
#                              'combinedseas', 'combined']
        self.swoosh_field = swoosh_field
        self.run_on_cluster = False  # False to run locally mainly to test stuff
        self.special_run = 'normal'
        self.data_name = data_name  # merra, era5
        self.species = species  # can be T or phi for era5
        self.shift = None
        self.time_period = time_period
        self.area_mean = area_mean
        self.original_data_file = original_data_file  # original data filename (in work_path)
        if sys.platform == 'linux':
            self.work_path = '/home/shlomi/Desktop/DATA/Work_Files/Chaim_Stratosphere_Data/'
            self.cluster_path = '/mnt/cluster/'
        elif sys.platform == 'darwin':  # mac os
            self.work_path = '/Users/shlomi/Documents/Chaim_Stratosphere_Data/'
            self.cluster_path = '/Users/shlomi/Documents/cluster/'

    # update attrs from dict containing keys as attrs and vals as attrs vals
    # to be updated

    def from_dict(self, d):
        self.__dict__.update(d)
        return self

    def show(self, name='all'):
        from aux_functions_strat import text_blue
        if name == 'all':
            for attr, value in vars(self).items():
                text_blue(attr, end="  "), print('=', value, end="  ")
                print('')
        elif hasattr(self, name):
            text_blue(name, end="  "), print('=', getattr(self, name),
                                             end="  ")
            print('')

    def load(self, name='regressors'):
        import xarray as xr
        if name == 'regressors':
            data = xr.open_dataset(self.regressors_file)
        elif name == 'original':
            data = xr.open_dataset(self.work_path + self.original_data_file)
        return data

    def select_model(self, model_name=None, ml_params=None):
        # pick ml model from ML_models class dict:
        ml = ML_Switcher()
        if model_name is not None:
            ml_model = ml.pick_model(model_name)
            self.model_name = model_name
        else:
            ml_model = ml.pick_model(self.model_name)
        # set external parameters if i want:
        if ml_params is not None:
            ml_model.set_params(**ml_params)
        return ml_model


class ML_Switcher(object):
    def pick_model(self, model_name):
        """Dispatch method"""
        method_name = str(model_name)
        # Get the method from 'self'. Default to a lambda.
        method = getattr(self, method_name, lambda: "Invalid ML Model")
        # Call the method as we return it
        return method()

    def LR(self):
        from sklearn.linear_model import LinearRegression
        return LinearRegression(n_jobs=-1, copy_X=True)

    def GPSR(self):
        from gplearn.genetic import SymbolicRegressor
        return SymbolicRegressor(random_state=42, n_jobs=1, metric='mse')

    def LASSOCV(self):
        from sklearn.linear_model import LassoCV
        import numpy as np
        return LassoCV(random_state=42, cv=5, n_jobs=-1,
                       alphas=np.logspace(-5, 1, 60))

    def MTLASSOCV(self):
        from sklearn.linear_model import MultiTaskLassoCV
        import numpy as np
        return MultiTaskLassoCV(random_state=42, cv=10, n_jobs=-1,
                                alphas=np.logspace(-5, 2, 400))

    def MTLASSO(self):
        from sklearn.linear_model import MultiTaskLasso
        return MultiTaskLasso()

    def KRR(self):
        from sklearn.kernel_ridge import KernelRidge
        return KernelRidge(kernel='poly', degree=2)

    def GPR(self):
        from sklearn.gaussian_process import GaussianProcessRegressor
        return GaussianProcessRegressor(random_state=42)

    def MTENETCV(self):
        import numpy as np
        from sklearn.linear_model import MultiTaskElasticNetCV
        return MultiTaskElasticNetCV(random_state=42, cv=10, n_jobs=-1,
                                alphas=np.logspace(-5, 2, 400))

#class ML_models:
#
#    def pick_model(self, model_name='LR'):
#        import numpy as np
#        from gplearn.genetic import SymbolicRegressor
#        from sklearn.linear_model import LassoCV
#        from sklearn.gaussian_process import GaussianProcessRegressor
#        from sklearn.gaussian_process.kernels import ExpSineSquared, WhiteKernel, ConstantKernel
#        from sklearn.linear_model import SGDRegressor
#        from sklearn.linear_model import RANSACRegressor
#        from sklearn.linear_model import TheilSenRegressor
#        from sklearn.linear_model import LinearRegression
#        from sklearn.linear_model import Ridge
#        from sklearn.linear_model import Lasso
#        from sklearn.linear_model import MultiTaskLassoCV
#        from sklearn.linear_model import MultiTaskLasso
#        from sklearn.linear_model import MultiTaskElasticNet
#        from sklearn.linear_model import ElasticNet
#        from sklearn.kernel_ridge import KernelRidge
#        from sklearn.ensemble import RandomForestRegressor
#        from sklearn.svm import SVR
#        from sklearn.neural_network import MLPRegressor
#        model_dict = {}
#        model_dict['GP-SR'] = SymbolicRegressor(random_state=42,
#                                                n_jobs=1, metric='mse')
#        model_dict['LASSOCV'] = LassoCV(random_state=42, cv=5, n_jobs=-1,
#                                        alphas=np.logspace(-5, 1, 60))
#        model_dict['MT-LASSOCV'] = MultiTaskLassoCV(random_state=42, cv=5,
#                                                    n_jobs=-1,
#                                                    alphas=np.logspace(-10, 1, 400))
#        model_dict['LR'] = LinearRegression(n_jobs=-1, copy_X=True)
#        model_dict['MT-LASSO'] = MultiTaskLasso()
#        model_dict['KRR'] = KernelRidge(kernel='poly', degree=2)
#        model_dict['GP-SR'] = SymbolicRegressor(random_state=42, n_jobs=1,
#                                                metric='mse')
#        model_dict['GPR'] = GaussianProcessRegressor(random_state=42)
#        self.model = model_dict.get(model_name, 'Invalid')
#        if self.model == 'Invalid':
#            raise KeyError('WRONG MODEL NAME!!')
#        return self.model


def run_ML(species='h2o', swoosh_field='combinedanomfillanom', model_name='LR',
           ml_params=None, area_mean=False, RI_proc=False,
           poly_features=None, time_period=None, cv=None,
           regressors=['qbo_1', 'qbo_2', 'ch4'], reg_except=None,
           time_shift=None):
    """Run ML model with...
    regressors = all"""
    def parse_cv(cv):
        from sklearn.model_selection import KFold
        from sklearn.model_selection import RepeatedKFold
        """input:cv number or string"""
        # check for integer:
        if isinstance(cv, int):
            return KFold(cv)
        # check for 2-tuple integer:
        if (isinstance(cv, tuple) and all(isinstance(n, int) for n in cv)
                and len(cv) == 2):
            return RepeatedKFold(n_splits=cv[0], n_repeats=cv[1],
                                 random_state=42)
        if not hasattr(cv, 'split') or isinstance(cv, str):
            raise ValueError("Expected cv as an integer, cross-validation "
                             "object (from sklearn.model_selection) "
                             ". Got %s." % cv)
        return cv
    # ints. parameters and feed run_ML args to it:
    arg_dict = locals()
    keys_to_remove = ['parse_cv', 'RI_proc', 'ml_params', 'cv']
    [arg_dict.pop(key) for key in keys_to_remove]
    p = parameters(**arg_dict)
    # p.from_dict(arg_dict)
    # select model:
    ml_model = p.select_model(model_name, ml_params)
    # pre proccess:
    X, y = pre_proccess(p)
    # unpack regressors:

    print('Running with regressors: ', ', '.join([x for x in
                                                  X.regressors.values]))
    # wrap ML_model:
    model = ImprovedRegressor(ml_model, reshapes='regressors',
                              sample_dim='time')
    # check for CV builtin model(e.g., LassoCV, GridSearchCV):
    if cv is not None and hasattr(ml_model, 'cv') and not RI_proc:
        cv = parse_cv(cv)
        print(cv)
#        if hasattr(ml_model, 'alphas'):
#            from yellowbrick.regressor import AlphaSelection
#            ml_model.set_params(cv=cv)
#            print(model.estimator)
#            visualizer = AlphaSelection(ml_model)
#            visualizer.fit(X, y)
#            g = visualizer.poof()
#            model.fit(X, y)
#        else:
        model.set_params(cv=cv)
        print(model.estimator)
        model.fit(X, y)
    # next, just do cross-val with models without CV(e.g., LinearRegression):
    elif cv is not None and not hasattr(ml_model, 'cv') and not RI_proc:
        from sklearn.multioutput import MultiOutputRegressor
        from sklearn.model_selection import cross_validate
        cv = parse_cv(cv)
        print(cv)
        # get multi-target dim:
        mt_dim = [x for x in y.dims if x != 'time'][0]
        mul = (MultiOutputRegressor(model.estimator))
        print(mul)
        mul.fit(X, y)
        cv_results = [cross_validate(mul.estimators_[i], X,
                                     y.isel({mt_dim: i}),
                                     cv=cv, scoring='r2') for i in
                      range(len(mul.estimators_))]
        cds = proccess_cv_results(cv_results, y, 'time')
        return cds
    elif RI_proc:
        model.make_RI(X, y)
    else:
        print(model.estimator)
        model.fit(X, y)
    # append parameters to model class:
    model.run_parameters_ = p
    return model


def proccess_cv_results(cvr, y, sample_dim):
    """proccess cross_validation results and build an xarray with dims of y for
    them"""
    import xarray as xr
    import numpy as np
    mt_dim = [x for x in y.dims if x != sample_dim][0]
    test = xr.DataArray([x['test_score'] for x in cvr],
                        dims=[mt_dim, 'kfold'])
    train = xr.DataArray([x['train_score'] for x in cvr],
                         dims=[mt_dim, 'kfold'])
    train.name = 'train'
    cds = test.to_dataset(name='test')
    cds['train'] = train
    cds[mt_dim] = y[mt_dim]
    cds['kfold'] = np.arange(len(cvr[0]['test_score'])) + 1
    cds['mean_train'] = cds.train.mean('kfold')
    cds['mean_test'] = cds.test.mean('kfold')
    # unstack:
    cds = cds.unstack(mt_dim)
    # put attrs back to geographical coords:
    # first pop sample dim (it is not in cds's coords)
    for coord, attr in y.attrs['coords_attrs'].items():
        if coord != sample_dim:
            cds[coord].attrs = attr
    return cds


def get_feature_multitask_dim(X, y, sample_dim):
    """return feature dim and multitask dim if exists, otherwise return empty
    lists"""
    # check if y has a multitask dim, i.e., y(sample, multitask)
    mt_dim = [x for x in y.dims if x != sample_dim]
    # check if X has a feature dim, i.e., X(sample, regressors)
    feature_dim = [x for x in X.dims if x != sample_dim]
    if mt_dim:
        mt_dim = mt_dim[0]
    if feature_dim:
        feature_dim = feature_dim[0]
    return feature_dim, mt_dim


def get_p_values(X, y, sample_dim):
    """produce p_values and return an xarray with the proper dims"""
    import numpy as np
    from sklearn.feature_selection import f_regression
    feature_dim, mt_dim = get_feature_multitask_dim(X, y, sample_dim)
    if mt_dim:
        pval = np.empty((y[mt_dim].size, X[feature_dim].size))
        for i in range(y[mt_dim].size):
            f, pval[i, :] = f_regression(X, y.isel({mt_dim: i}))
    else:
        pval = np.empty((X[feature_dim].size))
        f, pval[:] = f_regression(X, y)
    return pval


def sk_attr(est, attr):
    """check weather an attr exists in sklearn model"""
    from sklearn.utils.validation import check_is_fitted
    from sklearn.exceptions import NotFittedError
    try:
        check_is_fitted(est, attr)
        return True
    except NotFittedError:
        return False


def produce_RI(res_dict, feature_dim):
    """produce Relative Impcat from regressors and predicted fields from
    scikit learn models"""
    # input is a result_dict with keys as 'full_set' ,'reg_list[1], etc...
    # output is the RI calculated dataset from('full_set') with RI dataarrays
    import xarray as xr
    from aux_functions_strat import text_blue, xr_order
    # first all operations run on full-set dataset:
    rds = res_dict['full_set']
    names = [x for x in res_dict.keys() if x != 'full_set']
    for da_name in names:
        rds['std_' + da_name] = (rds['predict'] -
                                 res_dict[da_name]['predict']).std('time')
    std_names = [x for x in rds.data_vars.keys() if 'std' in x]
    rds['std_total'] = sum(d for d in rds[std_names].data_vars.values())
    for da_name in names:
        rds['RI_' + da_name] = rds['std_' + da_name] / rds['std_total']
        rds['RI_' + da_name].attrs['long_name'] = 'Relative Impact of '\
                                                  + da_name + ' regressor'
        rds['RI_' + da_name].attrs['units'] = 'Relative Impact'
    # get the RI names to concat them all to single dataarray:
    RI_names = [x for x in rds.data_vars.keys() if 'RI_' in x]
    rds['RI'] = xr.concat(rds[RI_names].to_array(), dim=feature_dim)
    rds['RI'].attrs = ''
    rds['RI'].attrs['long_name'] = 'Relative Impact of regressors'
    rds['RI'].attrs['units'] = 'RI'
    rds['RI'].attrs['defenition'] = 'std(predicted_full_set -\
    predicted_full_where_regressor_is_equal_to_its_median)/sum(denominator)'
    names_to_drop = [x for x in rds.data_vars.keys() if 'std' in x]
    rds = rds.drop(RI_names)
    rds = rds.drop(names_to_drop)
    rds = rds.reset_coords(drop=True)
    rds.attrs['feature_types'].append('RI')
    text_blue('Calculating RI scores for SciKit Learn Model.')
    return xr_order(rds)


def pre_proccess(params):
    """ pre proccess the data, area_mean is reducing the data to level-time,
    time_period is a slicer of the specific period. lag is inside params and
    shifting the data with minus"""
    import aux_functions_strat as aux
    # import numpy as np
    import xarray as xr
    import os
    path = os.getcwd() + '/regressors/'
    reg_file = params.regressors_file
    reg_list = params.regressors
    reg_except = params.reg_except
    poly = params.poly_features
    # load X i.e., regressors
    regressors = xr.open_dataset(path + reg_file)
    # unpack params to vars
    dname = params.data_name
    species = params.species
    season = params.season
    shift = params.time_shift
    # model = params.model
    special_run = params.special_run
    time_period = params.time_period
    area_mean = params.area_mean
#    if special_run == 'mask-pinatubo':
#        # mask vol: drop the time dates:
#        # '1991-07-01' to '1996-03-01'
#        date_range = pd.date_range('1991-07-01', '1996-03-01', freq='MS')
#        regressors = regressors.drop(date_range, 'time')
#        aux.text_yellow('Dropped ~five years from regressors due to\
#                        pinatubo...')
    path = params.work_path
    if params.run_on_cluster:
        path = '/data11/ziskin/'
    # load y i.e., swoosh or era5
    if dname == 'era5':
        nc_file = 'ERA5_' + species + '_all.nc'
        da = xr.open_dataarray(path + nc_file)
    elif dname == 'merra':
        nc_file = 'MERRA_' + species + '.nc'
        da = xr.open_dataarray(path + nc_file)
    elif dname == 'swoosh':
        ds = xr.open_dataset(path + params.original_data_file)
        field = params.swoosh_field + species + 'q'
        da = ds[field]
    # align time between y and X:
    new_time = aux.overlap_time_xr(da, regressors.to_array())
    regressors = regressors.sel(time=new_time)
    da = da.sel(time=new_time)
    if time_period is not None:
        print('selecting time period: ' + str(time_period))
        regressors = regressors.sel(time=slice(*time_period))
        da = da.sel(time=slice(*time_period))
    # slice to level and latitude:
    if dname == 'swoosh' or dname == 'era5':
        da = da.sel(level=slice(100, 1), lat=slice(-20, 20))
    elif dname == 'merra':
        da = da.sel(level=slice(100, 0.1), lat=slice(-20, 20))
    # select seasonality:
    if season != 'all':
        da = da.sel(time=da['time.season'] == season)
        regressors = regressors.sel(time=regressors['time.season'] == season)
    # area_mean:
    if area_mean:
        da = aux.xr_weighted_mean(da)
    # normalize X
    regressors = regressors.apply(aux.normalize_xr, norm=1,
                                  keep_attrs=True, verbose=False)
    # remove nans from y:
    da = aux.remove_nan_xr(da, just_geo=False)
    # regressors = regressors.sel(time=da.time)
    if 'seas' in field:
        how = 'mean'
    else:
        how = 'std'
    # deseason y
    if season != 'all':
        da = aux.deseason_xr(da, how=how, season=season, verbose=False)
    else:
        da = aux.deseason_xr(da, how=how, verbose=False)
    # shift according to scheme
#    if params.shift is not None:
#        shift_da = create_shift_da(sel=params.shift)
#        da = xr_shift(da, shift_da, 'time')
#        da = da.dropna('time')
#        regressors = regressors.sel(time=da.time)
    # saving attrs:
    attrs = [da[dim].attrs for dim in da.dims]
    da.attrs['coords_attrs'] = dict(zip(da.dims, attrs))
    # stacking reg:
    reg_names = [x for x in regressors.data_vars.keys()]
    reg_stacked = regressors[reg_names].to_array(dim='regressors').T
    # reg_select behivior:
    reg_select = [x for x in reg_stacked.regressors.values]
    if reg_list is not None:  # it is the default
        # convert str to list:
        if isinstance(regressors, str):
            regressors = regressors.split(' ')
        # select regressors:
        reg_select = [x for x in reg_list]
    if reg_except is not None and reg_list is None:
        reg_select = [x for x in reg_stacked.regressors.values if x not in
                      reg_except]
    reg_stacked = reg_stacked.sel({'regressors': reg_select})
    # poly features:
    if poly is not None:
        reg_stacked = poly_features(reg_stacked, degree=poly)
    if shift is not None:
        da = da.shift({'time': shift})
        da = da.dropna('time')
        reg_stacked = reg_stacked.sel(time=da.time)
    # da stacking:
    dims_to_stack = [x for x in da.dims if x != 'time']
    da = da.stack(samples=dims_to_stack).squeeze()
    return reg_stacked, da


def poly_features(X, feature_dim='regressors', degree=2,
                  interaction_only=False, include_bias=False,
                  normalize_poly=False, plot=False):
    from sklearn.preprocessing import PolynomialFeatures
    import xarray as xr
    from aux_functions_strat import normalize_xr
    sample_dim = [x for x in X.dims if x != feature_dim][0]
    # Vars = ['x' + str(n) for n in range(X.shape[1])]
    # dic = dict(zip(Vars, X[dim_name].values))
    poly = PolynomialFeatures(degree=degree, interaction_only=interaction_only,
                              include_bias=include_bias)
    X_new = poly.fit_transform(X)
    feature_names = [x for x in X[feature_dim].values]
    new_names = poly.get_feature_names(feature_names)
    new_names = [x.replace(' ', '*') for x in new_names]
    X_with_poly_features = xr.DataArray(X_new, dims=[sample_dim, feature_dim])
    X_with_poly_features[sample_dim] = X[sample_dim]
    X_with_poly_features[feature_dim] = new_names
    X_with_poly_features.attrs = X.attrs
    X_with_poly_features.name = 'Polynomial Features'
    if normalize_poly:
        names_to_normalize = list(set(new_names).difference(set(feature_names)))
        Xds = X_with_poly_features.to_dataset(dim=feature_dim)
        for da_name in names_to_normalize:
            Xds[da_name] = normalize_xr(Xds[da_name], norm=1, verbose=False)
        X_with_poly_features = Xds.to_array(dim=feature_dim).T
    if plot:
        import seaborn as sns
        ds = X_with_poly_features.to_dataset(dim='regressors').squeeze()
        le = len(ds.data_vars)
        df = ds.to_dataframe()
        if le <= 20:
            sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='bwr',
                        center=0.0)
        else:
            sns.heatmap(df.corr(), cmap='bwr', center=0.0)
    return X_with_poly_features


class ImprovedRegressor(RegressorWrapper):
    def __init__(self, estimator=None, reshapes=None, sample_dim=None,
                 **kwargs):
        # call parent constructor to set estimator, reshapes, sample_dim,
        # **kwargs
        super().__init__(estimator, reshapes, sample_dim, **kwargs)

    def fit(self, X, y=None, **fit_params):
        """ A wrapper around the fitting function.
        Improved: adds the X_ and y_ and results_ attrs to class.
        Parameters
        ----------
        X : xarray DataArray, Dataset other other array-like
            The training input samples.

        y : xarray DataArray, Dataset other other array-like
            The target values.

        Returns
        -------
        Returns self.
        """
        self = super().fit(X, y, **fit_params)
        # set results attr
        self.results_ = self.make_results(X, y)
        setattr(self, 'results_', self.results_)
        # set X_ and y_ attrs:
        setattr(self, 'X_', X)
        setattr(self, 'y_', y)
        return self

    def make_results(self, X, y):
        """ make results for all models type into xarray"""
        from aux_functions_strat import xr_order
        import xarray as xr
        from sklearn.metrics import r2_score
        from aux_functions_strat import text_blue
        feature_dim, mt_dim = get_feature_multitask_dim(X, y, self.sample_dim)
        rds = y.to_dataset('original').copy(deep=False, data=None)
        if sk_attr(self, 'coef_') and sk_attr(self, 'intercept_'):
            rds[feature_dim] = X[feature_dim]
            if mt_dim:
                rds['params'] = xr.DataArray(self.coef_, dims=[mt_dim,
                                                               feature_dim])
                rds['intercept'] = xr.DataArray(self.intercept_, dims=[mt_dim])
                pvals = get_p_values(X, y, self.sample_dim)
                rds['pvalues'] = xr.DataArray(pvals, dims=[mt_dim,
                                                           feature_dim])
            else:
                rds['params'] = xr.DataArray(self.coef_, dims=feature_dim)
                rds['intercept'] = xr.DataArray(self.intercept_)
                pvals = get_p_values(X, y, self.sample_dim)
                rds['pvalues'] = xr.DataArray(pvals, dims=feature_dim)
        elif sk_attr(self, 'feature_importances_'):
            if mt_dim:
                rds['feature_importances'] = xr.DataArray(self.
                                                          feature_importances_,
                                                          dims=[mt_dim,
                                                                feature_dim])
            else:
                rds['feature_importances'] = xr.DataArray(self.
                                                          feature_importances_,
                                                          dims=[feature_dim])
        predict = self.predict(X)
        if mt_dim:
            predict = predict.rename({self.reshapes: mt_dim})
            rds['predict'] = predict
            r2 = r2_score(y, predict, multioutput='raw_values')
            rds['r2'] = xr.DataArray(r2, dims=mt_dim)
        else:
            rds['predict'] = predict
            r2 = r2_score(y, predict)
            rds['r2'] = xr.DataArray(r2)
        if feature_dim:
            r2_adj = 1.0 - (1.0 - rds['r2']) * (len(y) - 1.0) / \
                 (len(y) - X.shape[1])
        else:
            r2_adj = 1.0 - (1.0 - rds['r2']) * (len(y) - 1.0) / (len(y))
        rds['r2_adj'] = r2_adj
        rds['predict'].attrs = y.attrs
        rds['resid'] = y - rds['predict']
        rds['resid'].attrs = y.attrs
        rds['resid'].attrs['long_name'] = 'Residuals'
        rds['dw_score'] = (rds['resid'].diff(self.sample_dim)**2).sum(self.sample_dim,
                                                                  keep_attrs=True) / (rds['resid']**2).sum(self.sample_dim, keep_attrs=True)
        # unstack dims:
        if mt_dim:
            rds = rds.unstack(mt_dim)
        # order:
        rds = xr_order(rds)
        # put coords attrs back:
        for coord, attr in y.attrs['coords_attrs'].items():
            rds[coord].attrs = attr
        # remove coords attrs from original:
        rds.original.attrs.pop('coords_attrs')
        all_var_names = [x for x in rds.data_vars.keys()]
        sample_types = [x for x in rds.data_vars.keys()
                        if self.sample_dim in rds[x].dims]
        feature_types = [x for x in rds.data_vars.keys()
                         if feature_dim in rds[x].dims]
        error_types = list(set(all_var_names) - set(sample_types +
                                                    feature_types))
        rds.attrs['sample_types'] = sample_types
        rds.attrs['feature_types'] = feature_types
        rds.attrs['error_types'] = error_types
        rds.attrs['sample_dim'] = self.sample_dim
        rds.attrs['feature_dim'] = feature_dim
        text_blue('Producing results...')
        return rds

    def make_RI(self, X, y):
        """ make Relative Impact score for estimator into xarray"""
        import aux_functions_strat as aux
        import warnings
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        feature_dim = [x for x in X.dims if x != self.sample_dim][0]
        regressors_list = aux.get_RI_reg_combinations(X.to_dataset
                                                      (dim=feature_dim))
        res_dict = {}
        for i in range(len(regressors_list)):
            keys = ','.join([key for key in regressors_list[i].
                             data_vars.keys()])
            print('Preforming ML-Analysis with regressors: ' + keys +
                  ', median = ' + regressors_list[i].attrs['median'])
            keys = regressors_list[i].attrs['median']
            new_X = regressors_list[i].to_array(dim=feature_dim)
            new_X = aux.xr_order(new_X)
            self.fit(new_X, y)
            res_dict[keys] = self.results_
#            elif mode == 'model_all':
#                params, res_dict[keys] = run_model_for_all(new_X, y, params)
#            elif mode == 'multi_model':
#                params, res_dict[keys] = run_multi_model(new_X, y, params)
        self.results_ = produce_RI(res_dict, feature_dim)
        self.X_ = X
        return

    def plot_like(self, field, div=False, robust=False, vmax=None, vmin=None):
        from matplotlib.ticker import ScalarFormatter
        import matplotlib.pyplot as plt
        import aux_functions_strat as aux
        import pandas as pd
        if not hasattr(self, 'results_'):
            raise AttributeError('No results yet... run model.fit(X,y) first!')
        rds = self.results_
        if field not in rds.data_vars:
            raise KeyError('No ' + str(field) + ' in results_!')
        if div:
            cmap = 'bwr'
        else:
            cmap = 'viridis'
        if field in rds.attrs['sample_types']:
            orig = aux.xr_weighted_mean(rds['original'])
            try:
                times = aux.xr_weighted_mean(rds[field])
            except KeyError:
                print('Field not found..')
                return
            except AttributeError:
                times = rds[field]
            fig, axes = plt.subplots(nrows=2, sharex=True, figsize=(15, 7),
                                     num='Time_Level_Comperison')
            cmap_max = abs(max(abs(orig.min().values), abs(orig.max().values)))
            orig = orig.reindex({'time': pd.date_range(orig.time[0].values,
                                                       orig.time[-1].values,
                                                       freq='MS')})
            con = orig.T.plot.contourf(ax=axes[0], yscale='log',
                                       yincrease=False,
                                       center=0.0, levels=41, vmax=cmap_max,
                                       cmap='bwr')
            cb = con.colorbar
            cb.set_label(orig.attrs['units'], fontsize=10)
            ax = axes[0]
            ax.set_title(orig.attrs['long_name'] + ' original', loc='center')
            ax.yaxis.set_major_formatter(ScalarFormatter())
            # plot the PREDICTED :
            times = times.reindex({'time': pd.date_range(times.time[0].values,
                                                         times.time[-1].values,
                                                         freq='MS')})
            con = times.T.plot.contourf(ax=axes[1], yscale='log',
                                        yincrease=False,
                                        cmap='bwr',
                                        center=0.0,
                                        levels=41,
                                        vmax=cmap_max,
                                        extend='both',
                                        robust=robust)
            cb = con.colorbar
            try:
                cb.set_label(times.attrs['units'], fontsize=10)
            except KeyError:
                print('no units found...''')
                cb.set_label(' ', fontsize=10)
            ax = axes[1]
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.set_title(times.attrs['long_name'] + ' predicted', loc='center')
            plt.subplots_adjust(left=0.05, right=0.995)
            plt.show()
            return con
        elif field in rds.attrs['error_types']:
            # TODO: add contour lines
            suptitle = rds[field].name
            try:
                con = rds[field].plot.contourf(yscale='log', yincrease=False,
                                               add_colorbar=True,
                                               cmap=cmap, levels=41,
                                               figsize=(6, 8),
                                               robust=robust, vmax=vmax,
                                               vmin=vmin)
                ax = plt.gca()
                ax.yaxis.set_major_formatter(ScalarFormatter())
                plt.suptitle(suptitle, fontsize=12, fontweight=750)
            except KeyError:
                print('Field not found or units not found...')
                return
            except ValueError:
                con = rds[field].plot(xscale='log', xincrease=False,
                                      figsize=(6, 8))
                ax = plt.gca()
                ax.xaxis.set_major_formatter(ScalarFormatter())
                plt.suptitle(suptitle, fontsize=12, fontweight=750)
            plt.show()
            return con
        elif field in rds.attrs['feature_types']:
            # TODO: add contour lines
            con_levels = [0.001, 0.005, 0.01, 0.05]  # for pvals
            con_colors = ['blue', 'cyan', 'yellow', 'red']  # for pvals
            import xarray as xr
            fdim = rds.attrs['feature_dim']
            flist = [x for x in rds[fdim].values if
                        xr.ufuncs.fabs(rds[field].sel({fdim: x})).mean() > 0]
            if rds[fdim].sel({fdim: flist}).size > 6:
                colwrap = 6
            else:
                colwrap = None
            suptitle = rds[field].name
            try:
                if rds[field].name == 'pvalues':
                    fg = rds[field].sel({fdim: flist}).plot.contourf(col=fdim,
                                                                     yscale='log',
                                                                     yincrease=False,
                                                                     add_colorbar=False,
                                                                     colors=con_colors,
                                                                     levels=con_levels,
                                                                     figsize=(15, 4),
                                                                     robust=robust,
                                                                     extend='min',
                                                                     col_wrap=colwrap)
                else:
                    fg = rds[field].sel({fdim: flist}).plot.contourf(col=fdim,
                                                                     yscale='log',
                                                                     yincrease=False,
                                                                     add_colorbar=False,
                                                                     cmap=cmap,
                                                                     levels=41,
                                                                     figsize=(15, 4),
                                                                     robust=robust,
                                                                     vmax=rds[field].max(
                                                                     ),
                                                                     col_wrap=colwrap)
                ax = plt.gca()
                ax.yaxis.set_major_formatter(ScalarFormatter())
                fg.fig.subplots_adjust(bottom=0.3, top=0.85, left=0.05)
                cbar_ax = fg.fig.add_axes([0.1, 0.1, .8, .025])
                fg.add_colorbar(
                    cax=cbar_ax,
                    orientation="horizontal",
                    format='%0.3f')
                fg.fig.suptitle(suptitle, fontsize=12, fontweight=750)
            except KeyError:
                print('Field not found or units not found...')
                return
            except ValueError:
                fg = rds[field].plot(col=fdim, xscale='log', xincrease=False,
                                     figsize=(15, 4))
                fg.fig.subplots_adjust(bottom=0.3, top=0.85, left=0.05)
                ax = plt.gca()
                ax.xaxis.set_major_formatter(ScalarFormatter())
                plt.suptitle(suptitle, fontsize=12, fontweight=750)
            plt.show()
            return fg
