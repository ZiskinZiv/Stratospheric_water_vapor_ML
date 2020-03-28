#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 10 13:20:30 2019

@author: shlomi
"""
# TODO: build GridSearchCV support directley like ultioutput regressors
# for various models

from strat_paths import work_chaim
from strat_paths import adams_path
from sklearn_xarray import RegressorWrapper
from xarray.core.dataset import Dataset
from pathlib import Path
# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)
sound_path = work_chaim / 'sounding'


class Parameters:
    """a parameters class for stratosphere gases modelling using ML methods"""
    def __init__(self,
                 model_name='LR',
                 season=None,
#                 regressors_file='Regressors.nc',
                 swoosh_field='combinedanomfillanom',
                 regressors=None,   # default None means all regressors
#                 reg_add_sub=None,
                 poly_features=None,
                 special_run=None,
                 reg_time_shift=None,
                 add_poly_reg=None,
                 data_name='swoosh',
                 species='h2o',
                 time_period=['1994', '2019'],
                 area_mean=False,
                 lat_slice=[-20, 20],
                 plevels=None,
                 # original_data_file='swoosh_lonlatpress-20deg-5deg.nc',
                 data_file='swoosh_latpress-2.5deg.nc'
                 ,**kwagrs):
        self.filing_order = ['data_name', 'field', 'model_name', 'season',
                             'reg_selection', 'special_run']
        self.delimeter = '_'
        self.model_name = model_name
        self.season = season
        self.lat_slice = lat_slice
        self.plevels = plevels
        self.reg_time_shift = reg_time_shift
        self.poly_features = poly_features
#        self.reg_add_sub = reg_add_sub
        self.regressors = regressors
        self.special_run = special_run
        self.add_poly_reg = add_poly_reg
#        self.regressors_file = regressors_file
#        self.sw_field_list = ['combinedanomfillanom', 'combinedanomfill',
#                              'combinedanom', 'combinedeqfillanom',
#                              'combinedeqfill', 'combinedeqfillseas',
#                              'combinedseas', 'combined']
        self.swoosh_field = swoosh_field
        self.run_on_cluster = False  # False to run locally mainly to test stuff
        self.data_name = data_name  # merra, era5
        self.species = species  # can be T or phi for era5
        self.time_period = time_period
        self.area_mean = area_mean
        self.data_file = data_file  # original data filename (in work_path)
        self.work_path = work_chaim
        self.cluster_path = adams_path

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

#    def load(self, name='regressors'):
#        import xarray as xr
#        if name == 'regressors':
#            data = xr.open_dataset(self.regressors_file)
#        elif name == 'original':
#            data = xr.open_dataset(self.work_path + self.original_data_file)
#        return data
    def load_regressors(self):
        from make_regressors import load_all_regressors
        return load_all_regressors()

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
        self.param_grid = ml.param_grid
        return ml_model


class ML_Switcher(object):
    def pick_model(self, model_name):
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
        return method()

    def LR(self):
        from sklearn.linear_model import LinearRegression
        return LinearRegression(n_jobs=-1, copy_X=True)

    def RANSAC(self):
        from sklearn.linear_model import RANSACRegressor
        return RANSACRegressor(random_state=42)

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
        import numpy as np
        self.param_grid = {'gamma': np.logspace(-5, 1, 5),
                           'alpha': np.logspace(-5, 2, 5)}
        return KernelRidge(kernel='poly', degree=3)

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
#        from sklearn.linear_model impoml_modelrt Ridge
#        from sklearn.linear_model import Lasso
#        from sklearn.linear_model import MultiTaskLassoCV
#        from sklearn.linear_model import MultiTaskLasso
#        from sklearn.linear_model import MultiTaskElasticNet
#        from sklearn.linear_model import ElasticNet
#        from sklearn.kernel_ridge impoml_modelrt KernelRidge
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


def run_level_month_shift(plags=['qbo_cdas'],
                          lslice=[-20, 20],
                          time_period=['1984', '2019'], lag_period=[0, 12],
                          species=None):
    print('producing level month shift for {} regressors'.format(plags))
    print('time period: {} to {}'.format(*time_period))
    print('latitude boundries: {} to {}'.format(*lslice))
    print('month lags allowed: {} to {}'.format(*lag_period))
    if species is None:
        rds = run_ML(time_period=time_period, area_mean=True,
                     lat_slice=lslice, special_run={'optimize_reg_shift': lag_period},
                     regressors=plags)
    elif species == 't':
        rds = run_ML(time_period=time_period, species='t', area_mean=True,
                     lat_slice=lslice, special_run={'optimize_reg_shift': lag_period},
                     regressors=plags, data_file='era5_t_85hPa.nc')
    elif species == 'u':
        rds = run_ML(time_period=time_period, species='u', area_mean=True,
                     lat_slice=lslice, special_run={'optimize_reg_shift': lag_period},
                     regressors=plags, data_file='era5_u_85hPa.nc')
    return rds.level_month_shift




def produce_run_ML_for_each_season(plags=['qbo_cdas'], regressors=[
                                   'qbo_cdas', 'anom_nino3p4', 'ch4'],
                                   savepath=None, latlon=False,
                                   extra='poly_2_no_ch4_extra_terms'):
    """data for fig 7"""
    import xarray as xr
    seasons = ['JJA', 'SON', 'DJF', 'MAM']
    # first do level months reg shift for plags:
    rds = run_ML(time_period=['1984', '2019'], area_mean=True,
                 lat_slice=[-20, 20], special_run={'optimize_reg_shift': [0, 12]},
                 regressors=plags)
    lms = rds.level_month_shift
    ds_list = []
    if latlon:
        start = '2004'
        res = 'latlon'
    else:
        start = '1984'
        res = 'latpress'
    for season in seasons:
        print('selecting season {} for ML-analysis:'.format(season))
        rds = run_ML(
            time_period=[
                start,
                '2019'],
            regressors=regressors,
            season=season,
            lms=lms, swoosh_latlon=latlon, RI_proc=True,
            lat_slice=[-60, 60])
        attrs = rds.results_.attrs
        ds_list.append(rds.results_)

    to_concat_season_vars = [x for x in ds_list[0].keys(
    ) if 'time' not in ds_list[0][x].dims]
    to_concat_time_vars = [x for x in ds_list[0].keys(
    ) if 'time' in ds_list[0][x].dims]
    ds_season_list = [x[to_concat_season_vars] for x in ds_list]
    ds_time_list = [x[to_concat_time_vars] for x in ds_list]
    rds = xr.concat(ds_season_list, 'season')
    rds['season'] = seasons
    time_ds = xr.concat(ds_time_list, 'time')
    rds = xr.merge([time_ds, rds])
    rds.attrs = attrs
    if savepath is not None:
        filename = 'MLR_H2O_{}_seasons_cdas-plags_ch4_enso_{}_{}-2019.nc'.format(res, extra, start)
        comp = dict(zlib=True, complevel=9)  # best compression
        encoding = {var: comp for var in rds.data_vars}
        rds.to_netcdf(savepath / filename, 'w', encoding=encoding)
        print('saved results to {}, {}.'.format(savepath, filename))
    return rds


#def produce_regressors_response_figs(X, results, level, time, grp='time.season'):
#    time_slice = False
#    if isinstance(time, str):
#        time = int(time)  # assuming year only
#    if not isinstance(time, int) and len(time) == 2:
#        time = [str(x) for x in time]
#        time_slice = True
#        min_time = time[0]
#        max_time = time[1]
#    field = results.original.attrs['long_name']
#    units = results.original.attrs['units']
#    if time_slice:
#        da = results.params.sel(level=level, method='nearest') * X.sel(time=slice(min_time, max_time))
#    else:
#        da = results.params.sel(level=level, method='nearest') * X.sel(time=str(time))
#    plt_kwargs = {'cmap': 'bwr', 'figsize': (15, 10),
#                  'add_colorbar': False,
#                  'extend': 'both'}
#    plt_kwargs.update({'center': 0.0, 'levels': 41})
#    da_seasons = da.groupby(grp).mean('time')
#    grp_row = grp.split('.')[-1]
#    fg = da_seasons.plot.contourf(row=grp_row, col='regressors', **plt_kwargs)
#    cbar_ax = fg.fig.add_axes([0.1, 0.1, .8, .025])
#    fg.add_colorbar(
#        cax=cbar_ax, orientation="horizontal", label=units,
#        format='%0.3f')
#    # year = list(set(da.time.dt.year.values))[0]
#    level = da_seasons.level.values.item()
#    if time_slice:
#        fg.fig.suptitle('{}, level={:.2f} hPa , time={} to {}'.format(
#                field, level, min_time, max_time), fontsize=12, fontweight=750)
#    else:
#        fg.fig.suptitle('{}, level={:.2f} hPa , time={}'.format(
#                field, level, time), fontsize=12, fontweight=750)
#    fg.fig.subplots_adjust(bottom=0.2, top=0.9, left=0.05)
#    return fg


def compare_RI(rds1, rds2, names=['rds1', 'rds2']):
    """input: rds datasets"""
    import matplotlib.pyplot as plt
    import xarray as xr
    import cartopy.crs as ccrs
    fg = plot_like_results(rds1, plot_key='RI_map', level=82, cartopy=True)
    rds1 = fg.data
    plt.close()
    fg = plot_like_results(rds2, plot_key='RI_map', level=82, cartopy=True)
    rds2 = fg.data
    plt.close()
    rds = xr.concat([rds1, rds2], 'model')
    rds['model'] = names
    proj = ccrs.PlateCarree(central_longitude=0)
    plt_kwargs = {'cmap': 'viridis', 'figsize': (15, 10),
                  'add_colorbar': False, 'levels': 41,
                  'extend': None, 'vmin': 0.0}
    plt_kwargs.update({'subplot_kws': dict(projection=proj),
                           'transform': ccrs.PlateCarree()})
    label_add = 'Relative impact'
    # plt_kwargs.update(kwargs)
    label_add += ' at level= {:.2f} hPa'.format(82.54)
    fg = rds.plot.contourf(col='regressors', row='model', **plt_kwargs)
    cbar_ax = fg.fig.add_axes([0.1, 0.1, .8, .025])
    fg.add_colorbar(
                cax=cbar_ax, orientation="horizontal", label='',
                format='%0.3f')
    fg.fig.suptitle(label_add, fontsize=12, fontweight=750)
    [ax.coastlines() for ax in fg.axes.flatten()]
    [ax.gridlines(
        crs=ccrs.PlateCarree(),
        linewidth=1,
        color='black',
        alpha=0.5,
        linestyle='--',
        draw_labels=False) for ax in fg.axes.flatten()]
    fg.fig.subplots_adjust(bottom=0.2, top=0.9, left=0.05)
    plt.show()
    return fg


def run_grid_multi(reg_stacked, da_stacked, params):
    """run one grid_search_cv on a multioutputregressors(model)"""
    import xarray as xr
    params.grid_search.fit(reg_stacked, da_stacked)
    rds = xr.Dataset()
    rds['cv_results'] = process_gridsearch_results(params.grid_search)
    rds['best_score'] = xr.DataArray(params.grid_search.best_score_)
    rds['best_params'] = xr.DataArray(
        list(
            params.grid_search.best_params_.values()),
        dims=['cv_params'])
    rds['cv_params'] = list(params.grid_search.param_grid.keys())
    rds.attrs['scoring_for_best'] = params.grid_search.refit
    # predict = xr.DataArray([est.predict(reg_stacked) for est in
    #                         params.multi.estimators_], dims='samples')
    # rds['predict'] = predict
    return params, rds


def save_cv_results(cvr, ml_param_add=None, savepath=work_chaim):
    import pandas as pd
    # run linear kernel with alpha opt
    # run poly kernel with degrees: 1 to 3 and optimize gamma, alpha, with fixed coef0
    # run rbf, sigmoid, laplacian with gamma, alpha opt.
    # save for each kernel, then run_ML with these parameters, latlon H2O
    if 'lon' in cvr.attrs and len(cvr.attrs['lon']) > 1:
        geo = 'latlon'
    else:
        geo = 'latpress'
    name = cvr.attrs['model_name']
    params = cvr.attrs['param_names']
    species = cvr.attrs['species']
    time = cvr.attrs['time']
    max_year = pd.to_datetime(max(time)).year
    min_year = pd.to_datetime(min(time)).year
    regs = cvr.attrs['regressors']
    cvr_to_save = cvr[[x for x in cvr.data_vars if 'best_model' not in x]]
    cvr_to_save.attrs.pop('time')
    # shifted = cvr.attrs['reg_shifted']
    if ml_param_add is not None:
        filename = 'CVR_{}_{}_{}_{}_{}_{}_{}-{}.nc'.format(
            name,
            ml_param_add,
            '_'.join(params),
            species, geo,
            '_'.join(regs),
            min_year,
            max_year)
    else:
        filename = 'CVR_{}_{}_{}_{}_{}_{}-{}.nc'.format(
            name,
            '_'.join(params),
            species, geo,
            '_'.join(regs),
            min_year,
            max_year)
    comp = dict(zlib=True, complevel=9)  # best compression
    encoding = {var: comp for var in cvr_to_save.data_vars}
    cvr_to_save.to_netcdf(savepath / filename, 'w', encoding=encoding)
    print('saved results to {}.'.format(savepath/filename))
    return


def plot_cv_results(cvr, level=82, col_param=None, row_param=None):
    # TODO: need to fix plot
    log_scale_params = ['gamma', 'alpha']
    splits = cvr.split.size
    param_names = cvr.attrs['param_names']
    # if not isinstance(level, list):
    #     level = int(level)
    cvr_level = cvr.sel(level=level, method='nearest')
        
    # elif isinstance(level, list) and len(level) == 2:
        # cvr_level = cvr.sel(level=slice(level[0], level[1]))
    # else:
    #     raise('level should be either a len(2) list or int')
    if len(param_names) == 1:
        tt = cvr_level[['mean_train_score', 'mean_test_score']].to_array(dim='task')
        fg = tt.plot(col='task', xscale='log')
        fg.fig.suptitle('mean train/test score for the {} hPa level and {} splits'.format(level, splits))
    elif len(param_names) == 2:
        tt = cvr_level[['mean_train_score', 'mean_test_score']].to_array(dim='task')
        fg = tt.plot.contourf(vmin=0.0, levels=21, col='task', xscale='log', yscale='log')
        fg.fig.suptitle('mean train/test score for the {} hPa level and {} splits'.format(level, splits))
    elif len(param_names) == 3:
        tt = cvr_level[['mean_train_score', 'mean_test_score']].to_array(dim='task')
        fg = tt.plot.contourf(vmin=0.0, levels=21, row='coef0',col='task', xscale='log', yscale='log')
        fg.fig.suptitle('mean train/test score for the {} hPa level and {} splits'.format(level, splits))
    return cvr_level

    
def process_gridsearch_results(GridSearchCV):
    import xarray as xr
    import pandas as pd
    """takes GridSreachCV object with cv_results and xarray it into dataarray"""
    params = GridSearchCV.param_grid
    scoring = GridSearchCV.scoring
    names = [x for x in params.keys()]

    # unpack param_grid vals to list of lists:
    pro = [[y for y in x] for x in params.values()]
    ind = pd.MultiIndex.from_product((pro), names=names)
#        result_names = [x for x in GridSearchCV.cv_results_.keys() if 'split'
#                        not in x and 'time' not in x and 'param' not in x and
#                        'rank' not in x]
    result_names = [
        x for x in GridSearchCV.cv_results_.keys() if 'param' not in x]
    ds = xr.Dataset()
    for da_name in result_names:
        da = xr.DataArray(GridSearchCV.cv_results_[da_name])
        ds[da_name] = da
    ds = ds.assign(dim_0=ind).unstack('dim_0')
    # get all splits data and concat them along number of splits:
    all_splits = [x for x in ds.data_vars if 'split' in x]
    train_splits = [x for x in all_splits if 'train' in x]
    test_splits = [x for x in all_splits if 'test' in x]
    splits = [x for x in range(len(train_splits))]
    train_splits = xr.concat([ds[x] for x in train_splits], 'split')
    test_splits = xr.concat([ds[x] for x in test_splits], 'split')
    # replace splits data vars with newly dataarrays:
    ds = ds[[x for x in ds.data_vars if x not in all_splits]]
    ds['split_train_score'] = train_splits
    ds['split_test_score'] = test_splits
    ds['split'] = splits

#    name = [x for x in ds.data_vars.keys() if 'mean_test' in x]
#    # if scoring:
#    mean_test = xr.concat(ds[name].data_vars.values(), dim='scoring')
#    mean_test.name = 'mean_test'
#    name = [x for x in ds.data_vars.keys() if 'mean_train' in x]
#    mean_train = xr.concat(ds[name].data_vars.values(), dim='scoring')
#    mean_train.name = 'mean_train'
#    name = [x for x in ds.data_vars.keys() if 'std_test' in x]
#    std_test = xr.concat(ds[name].data_vars.values(), dim='scoring')
#    std_test.name = 'std_test'
#    name = [x for x in ds.data_vars.keys() if 'std_train' in x]
#    std_train = xr.concat(ds[name].data_vars.values(), dim='scoring')
#    std_train.name = 'std_train'
#    # ds = ds.drop(ds.data_vars.keys())
#    ds['mean_test'] = mean_test
#    ds['mean_train'] = mean_train
#    ds['std_test'] = std_test
#    ds['std_train'] = std_train
#    mean_test_train = xr.concat(ds[['mean_train', 'mean_test']].data_vars.
#                                values(), dim='train_test')
#    std_test_train = xr.concat(ds[['std_train', 'std_test']].data_vars.
#                               values(), dim='train_test')
#    ds['train_test'] = ['train', 'test']
#    # ds = ds.drop(ds.data_vars.keys())
#    ds['MEAN'] = mean_test_train
#    ds['STD'] = std_test_train
#    # CV = xr.Dataset(coords=GridSearchCV.param_grid)
#    ds = xr.concat(ds[['MEAN', 'STD']].data_vars.values(), dim='MEAN_STD')
#    ds['MEAN_STD'] = ['MEAN', 'STD']
    ds.attrs['name'] = 'CV_results'
    ds.attrs['param_names'] = names
    if isinstance(scoring, str):
        ds.attrs['scoring'] = scoring
        # ds = ds.reset_coords(drop=True)
#    else:
#        ds['scoring'] = scoring
    if GridSearchCV.refit:
        ds['best_score'] = GridSearchCV.best_score_
        ds['best_model'] = GridSearchCV.best_estimator_
        for name in names:
            ds['best_{}'.format(name)] = GridSearchCV.best_params_[name]
    return ds


def run_model_with_shifted_plevels(model, X, y, Target, plevel=None, lms=None):
    import xarray as xr
    import numpy as np
    from aux_functions_strat import text_blue
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.model_selection import GridSearchCV
    if lms is None:
        print('NO level month shift dataarray found: running model.fit(X, y) without shifted plevels')
        return model.fit(X, y)
    print(model.estimator)
    level_month_shift = lms
    regs = [x for x in level_month_shift.reg_shifted.values]
    print('Running with shifted plevels,' +
          ' with regressors shifts per level: {}'.format(regs))
#        for reg in regs:
    level_results = []
    lms = level_month_shift
    if plevel is None:
        levels = lms.level.values
    elif isinstance(plevel, list) and len(plevel) == 2:
        levels = lms.sel(level=slice(plevel[0], plevel[1])).level.values
    else:
        levels = np.expand_dims(lms.sel(level=plevel, method='nearest').level.values, axis=0)
    for lev in levels:
        level_shift = lms.sel(level=lev, method='nearest').values
        shift_dict = dict(zip(regs, level_shift))
        Xcopy = reg_shift(X, shift_dict)
        # p.plevels = lev
        # _, y = pre_proccess(p, verbose=False)
        Target = Target.from_dict(dict(plevels=lev, verbose=False))
        y = Target.pre_process()
        y_shifted = y.sel(time=Xcopy.time)
#            print('shifting target data {} months'.format(str(shift)))
#            print('X months: {}, y_months: {}'.format(X_shifted.time.size,
#                  y_shifted.time.size))
        if isinstance(model, MultiOutputRegressor):
            model.fit(Xcopy, y_shifted)
            level_results.append([x.results_ for x in model.estimators_])
        elif isinstance(model, GridSearchCV):
            model.fit(Xcopy, y_shifted)
            cv_ds = process_gridsearch_results(model)
            level_results.append(cv_ds)
        else:
            model.fit(Xcopy, y_shifted, verbose=False)
            level_results.append(model.results_)
    rds = xr.concat(level_results, dim='level')
    print(text_blue('Done!'))
    if isinstance(model, GridSearchCV):
        rds['level'] = levels
        rds.attrs['model_name'] = p.model_name
        if 'H2O' in y.attrs['long_name']:
            rds.attrs['species'] = 'H2O'
        elif 'temperature' or 'Temperature' in y.attrs['long_name']:
            rds.attrs['species'] = 't'
        elif 'wind' or 'Wind' in y.attrs['long_name']:
            rds.attrs['species'] = 'u'
        y_coords = y.unstack('samples').coords
        for coord, val in y_coords.items():
            rds.attrs[coord] = val.values
        rds.attrs['regressors'] = X.regressors.values
        rds.attrs['reg_shifted'] = regs
        return rds
    # rds = xr.concat(reg_results, dim='reg_shifted')
    rds['reg_shifted'] = regs
    rds['level_month_shift'] = level_month_shift
    if isinstance(model, MultiOutputRegressor):
        for est in model.estimators_:
            est.results_ = rds
    else:
        model.results_ = rds
    return model


def run_ML(species='h2o', swoosh_field='combinedanomfillanom', model_name='LR',
           ml_params=None, area_mean=False, RI_proc=False,
           poly_features=None, time_period=['1994', '2019'], cv=None,
           regressors=['era5_qbo_1', 'era5_qbo_2', 'ch4', 'radio_cold_no_qbo'],
           reg_time_shift=None, season=None, add_poly_reg=None, lms=None,
           special_run=None, gridsearch=False, plevels=None,
           lat_slice=[-60, 60], swoosh_latlon=False, param_grid=None,
           data_file='swoosh_latpress-2.5deg.nc'):
    """Run ML model with...
    regressors = None
    special_run is a dict with key as type of run, value is values passed to
    the special run
    reg_time_shift = {'radio_cold_no_qbo':[1,36]} - get the 36 lags of the regressor
    example special_run={'optimize_time_shift':(-12,12)}
    use optimize_time_shift with area_mean=True,
    add_poly_reg = {'anom_nino3p4': 2} : adds enso^2 to regressors"""
    from aux_functions_strat import overlap_time_xr
    def parse_cv(cv):
        from sklearn.model_selection import KFold
        from sklearn.model_selection import RepeatedKFold
        """input:cv number or string"""
        # check for integer:
        if 'kfold' in cv.keys():
            n_splits = cv['kfold']
            print('CV is KFold with n_splits={}'.format(n_splits))
            return KFold(n_splits=n_splits)
        if 'rkfold' in cv.keys():
            n_splits = cv['rkfold'][0]
            n_repeats = cv['rkfold'][1]
            print('CV is ReapetedKFold with n_splits={},'.format(n_splits) +
                  ' n_repeates={}'.format(n_repeats))
            return RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats,
                                 random_state=42)
        # check for 2-tuple integer:
#        if (isinstance(cv, tuple) and all(isinstance(n, int) for n in cv) and
#                len(cv) == 2):
#            return RepeatedKFold(n_splits=cv[0], n_repeats=cv[1],
#                                 random_state=42)
#        if not hasattr(cv, 'split') or isinstance(cv, str):
#            raise ValueError("Expected cv as an integer, cross-validation "
#                             "object (from sklearn.model_selection) "
#                             ". Got %s." % cv)
#        return cv
    # ints. parameters and feed run_ML args to it:
    arg_dict = locals()
    if swoosh_latlon:
        arg_dict['data_file'] = 'swoosh_lonlatpress-20deg-5deg.nc'
        arg_dict['swoosh_field'] = 'combinedanom'
        print('lat/lon run selected, (no fill product for lat/lon)')
    keys_to_remove = ['parse_cv', 'RI_proc', 'ml_params', 'cv', 'gridsearch',
                      'swoosh_latlon', 'lms', 'param_grid']
    [arg_dict.pop(key) for key in keys_to_remove]
    p = Parameters(**arg_dict)
    # p.from_dict(arg_dict)
    # select model:
    ml_model = p.select_model(model_name, ml_params)
    # pre proccess:
    Pset = PredictorSet(**arg_dict, loadpath=Path().cwd()/'regressors')
    Target = TargetArray(**arg_dict, loadpath=work_chaim)
    X = Pset.pre_process()
    y = Target.pre_process()
    # align time X, y:
    new_time = overlap_time_xr(y, X)
    X = X.sel(time=new_time)
    y = y.sel(time=new_time)
    # X, y = pre_proccess(p)
    # unpack regressors:

    print('Running with regressors: ', ', '.join([x for x in
                                                  X.regressors.values]))
    if plevels is None:
        print('Running with all pressure levels.')
    else:
        print('Running with {} pressure levels.'.format(plevels))
    # wrap ML_model:
    model = ImprovedRegressor(ml_model, reshapes='regressors',
                              sample_dim='time')
    # run special mode: optimize_time_shift !only with area_mean=true:
    if (p.special_run is not None
            and 'optimize_time_shift' in p.special_run.keys()):
        print(model.estimator)
        import numpy as np
        import xarray as xr
        from matplotlib.ticker import ScalarFormatter
        import matplotlib.pyplot as plt
        min_shift, max_shift = p.special_run['optimize_time_shift']
        print('Running with special mode: optimize_time_shift,' +
              ' with months shifts: {}, {}'.format(str(min_shift),
                                                   str(max_shift)))
        plt.figure()
        # a full year + and -:
        shifts = np.arange(min_shift, max_shift + 1)
        opt_results = []
        for shift in shifts:
            y_shifted = y.shift({'time': shift})
            y_shifted = y_shifted.dropna('time')
            X_shifted = X.sel(time=y_shifted.time)
#            print('shifting target data {} months'.format(str(shift)))
#            print('X months: {}, y_months: {}'.format(X_shifted.time.size,
#                  y_shifted.time.size))
            model.fit(X_shifted, y_shifted, verbose=False)
            opt_results.append(model.results_)
        rds = xr.concat(opt_results, dim='months_shift')
        rds['months_shift'] = shifts
        rds['level_month_shift'] = rds.months_shift.isel(
                months_shift=rds.r2_adj.argmax(dim='months_shift'))
        rds.level_month_shift.plot.line('r.-', y='level', yincrease=False)
        rds.r2_adj.T.plot.pcolormesh(yscale='log', yincrease=False, levels=21)
        ax = plt.gca()
        ax.set_title(', '.join(X.regressors.values.tolist()))
        ax.yaxis.set_major_formatter(ScalarFormatter())
        print('Done!')
        return rds
    # run special mode: optimize_reg_shift !only with area_mean=true:
    if (p.special_run is not None
            and 'optimize_reg_shift' in p.special_run.keys()):
        print(model.estimator)
        import numpy as np
        import xarray as xr
        from matplotlib.ticker import ScalarFormatter
        import matplotlib.pyplot as plt
        min_shift, max_shift = p.special_run['optimize_reg_shift']
        print('Running with special mode: optimize_reg_shift,' +
              ' with months shifts: {}, {}'.format(str(min_shift),
                                                   str(max_shift)))
        # plt.figure()
        # a full year + and -:
        shifts = np.arange(min_shift, max_shift + 1)
        # X = X.sel(regressors=['qbo_cdas', 'radio_cold_no_qbo'])
        # X = X.sel(regressors=['qbo_1', 'qbo_2', 'ch4', 'cold'])
        # reg_to_shift = ['qbo_1', 'qbo_2', 'cold']
        # reg_to_shift = ['qbo_cdas', 'radio_cold_no_qbo']
        reg_to_shift = [x for x in X.regressors.values]
        reg_results = []
        for reg in reg_to_shift:
            opt_results = []
            for shift in shifts:
                shift_dict = {reg: shift}
                Xcopy = reg_shift(X, shift_dict)
                y_shifted = y.sel(time=Xcopy.time)
#            print('shifting target data {} months'.format(str(shift)))
#            print('X months: {}, y_months: {}'.format(X_shifted.time.size,
#                  y_shifted.time.size))
                model.fit(Xcopy, y_shifted, verbose=False)
                opt_results.append(model.results_)
            opt_ds = xr.concat(opt_results, dim='months_shift')
            reg_results.append(opt_ds)
        rds = xr.concat(reg_results, dim='reg_shifted')
        rds['reg_shifted'] = reg_to_shift
        rds['months_shift'] = shifts
        rds['level_month_shift'] = rds.months_shift.isel(
                months_shift=rds.r2_adj.argmax(dim='months_shift'))
        fg = rds.r2_adj.T.plot.pcolormesh(yscale='log', yincrease=False,
                                          levels=21, col='reg_shifted',
                                          cmap='viridis', vmin=0.0,
                                          extend='both')
        for n_regs in range(len(fg.axes[0])):
            rds.isel(reg_shifted=n_regs).level_month_shift.plot.line('r.-',
                                                                     y='level',
                                                                     yincrease=False,
                                                                     ax=fg.axes[0][n_regs])
        ax = plt.gca()
        # ax.set_title(', '.join(X.regressors.values.tolist()))
        plt.suptitle(', '.join(X.regressors.values.tolist()),
                     fontweight='bold')
        plt.subplots_adjust(top=0.85, right=0.82)
        ax.yaxis.set_major_formatter(ScalarFormatter())
        print('Done!')
        return rds
#    if (p.special_run is not None
#            and 'run_with_shifted_plevels' in p.special_run.keys()):
#        import xarray as xr
#        from aux_functions_strat import text_blue
#        print(model.estimator)
#        level_month_shift = p.special_run['run_with_shifted_plevels']
#        regs = [x for x in level_month_shift.reg_shifted.values]
#        print('Running with special mode: run_with_shifted_plevels,' +
#              ' with regressors shifts per level: {}'.format(regs))
##        for reg in regs:
#        level_results = []
#        lms = level_month_shift
#        for lev in lms.level.values:
#            level_shift = lms.sel(level=lev, method='nearest').values
#            shift_dict = dict(zip(regs, level_shift))
#            Xcopy = reg_shift(X, shift_dict)
#            p.plevels = lev
#            _, y = pre_proccess(p, verbose=False)
#            y_shifted = y.sel(time=Xcopy.time)
##            print('shifting target data {} months'.format(str(shift)))
##            print('X months: {}, y_months: {}'.format(X_shifted.time.size,
##                  y_shifted.time.size))
#            model.fit(Xcopy, y_shifted, verbose=False)
#            level_results.append(model.results_)
#        rds = xr.concat(level_results, dim='level')
#        print(text_blue('Done!'))
#        # rds = xr.concat(reg_results, dim='reg_shifted')
#        rds['reg_shifted'] = regs
#        rds['level_month_shift'] = level_month_shift
#        model.results_ = rds
#        return model
    # check for CV builtin model(e.g., LassoCV, GridSearchCV):
    if cv is not None:
        if hasattr(ml_model, 'cv') and not RI_proc:
            cv = parse_cv(cv)
#        if hasattr(ml_model, 'alphas'):
#            from yellowbrick.regressor import AlphaSelection
#            ml_model.set_params(cv=cv)
#            print(model.estimator)
#            visualizer = AlphaSelection(ml_model)
#            visualizer.fit(X, y)
#            g = visualizer.poof()
#            model.fit(X, y)
#        else:
#            if gridsearch:
#                ml_model.set_params(cv=cv)
#                print(ml_model.estimator)
#                ml_model.fit(X, y)
#                return ml_model
#            else:
            model.set_params(cv=cv)
            print(model.estimator_)
            model  = run_model_with_shifted_plevels(model, X, y, Target, plevel=plevels, lms=lms)
            # model.fit(X, y)
    # next, just do cross-val with models without CV(e.g., LinearRegression):
        if not hasattr(ml_model, 'cv') and not RI_proc:
            from sklearn.multioutput import MultiOutputRegressor
            from sklearn.model_selection import cross_validate
            cv = parse_cv(cv)
            # get multi-target dim:
            if gridsearch:
                from sklearn.model_selection import GridSearchCV
                if param_grid is None:
                    param_grid = p.param_grid
                gr = GridSearchCV(ml_model, param_grid, cv=cv,
                                  return_train_score=True, refit=True,
                                  scoring='r2', verbose=1)
                # mul = (MultiOutputRegressor(gr, n_jobs=1))
                print(gr)
                cvr = run_model_with_shifted_plevels(gr, X, y, Target, plevel=plevels, lms=lms)
                # mul.fit(X, y)
                return cvr
            mt_dim = [x for x in y.dims if x != model.sample_dim][0]
            mul = (MultiOutputRegressor(model.estimator))
            print(mul)
            mul = run_model_with_shifted_plevels(mul, X, y, Target, plevel=plevels, lms=lms)
            # mul.fit(X, y)
            cv_results = [cross_validate(mul.estimators_[i], X,
                                         y.isel({mt_dim: i}),
                                         cv=cv, scoring='r2',
                                         return_train_score=True) for i in
                          range(len(mul.estimators_))]
            cds = proc_cv_results(cv_results, y, model.sample_dim)
            return cds
    elif RI_proc:
        model.make_RI(X, y, Target, plevels=plevels, lms=lms)
    else:
        print(model.estimator)
        model  = run_model_with_shifted_plevels(model, X, y, Target, plevel=plevels, lms=lms)
        # model.fit(X, y)
    # append parameters to model class:
    model.run_parameters_ = p
    return model


def proc_cv_results(cvr, y, sample_dim):
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
    print(text_blue('Calculating RI scores for SciKit Learn Model.'))
    return xr_order(rds)


def pre_proccess(params, verbose=True):
    """ pre proccess the data, area_mean is reducing the data to level-time,
    time_period is a slicer of the specific period. lag is inside params and
    shifting the data with minus"""
    import aux_functions_strat as aux
    import xarray as xr
    import click
    plevels = params.plevels
    reg_list = params.regressors
    species = params.species
    season = params.season
    swoosh_field = params.swoosh_field
    lat_slice = params.lat_slice
    time_period = params.time_period
    area_mean = params.area_mean
    path = params.work_path
    poly = params.poly_features
    add_poly_reg = params.add_poly_reg
    # dict of {regressors: [1, 36]}
    reg_time_shift = params.reg_time_shift
    # load all of the regressors:
    regressors = params.load_regressors()
    # selecting specific regressors:
    if reg_list is not None:  # it is the default
        # convert str to list:
        if isinstance(reg_list, str):
            reg_list = reg_list.split(' ')
        try:
            regressors = regressors[reg_list]
        except KeyError:
            raise KeyError('The regressors selection cannot find {}'.format(reg_list))
    # selecting time period:
    if time_period is not None:
        if verbose:
            print('selecting regressors time period:{} to {}'.format(*time_period))
        regressors = regressors.sel(time=slice(*time_period))
    # anomlizing them:
    for reg in regressors.data_vars.keys():
        if reg == 'ch4':
            regressors[reg] = aux.normalize_xr(regressors[reg], norm=1, verbose=verbose)
        elif reg == 'radio_cold' or reg == 'radio_cold_no_qbo':
            regressors[reg] = aux.deseason_xr(regressors[reg], how='mean', verbose=verbose)
#        elif 'qbo_' in reg:
#            regressors[reg] = aux.normalize_xr(regressors[reg], norm=1)
#        elif 'solar' in reg:
#            regressors[reg] = aux.normalize_xr(regressors[reg], norm=5)
        elif 'bdc' in reg:
#            regressors[reg] = aux.normalize_xr(regressors[reg], norm=5)
            regressors[reg] = aux.deseason_xr(regressors[reg], how='mean', verbose=verbose)
#        elif 'vol' in reg:
#            regressors[reg] = aux.normalize_xr(regressors[reg], norm=5)
    # standartize all regressors (-avg)/std:
    regressors = aux.normalize_xr(regressors, norm=1, verbose=verbose)
    # add polynomials to chosen regs:
    if add_poly_reg is not None:
        for reg, n in add_poly_reg.items():
            if reg not in regressors.data_vars:
                raise('{} not in {}'.format(reg, regressors.data_vars))
            poly_reg = poly_regressor(regressors[reg], n=n, plot=False)
            regressors[poly_reg.name] = poly_reg
            # remove the original regressor:
            to_keep = [x for x in regressors.data_vars if x != reg]
            regressors = regressors[to_keep]
    # regressing one out of the other if neccesary:
    # just did it by specifically regressed out and saved as _index.nc'
    # making lags of some of the regressors (e.g, cold point):
    if reg_time_shift is not None:
        ds_list = []
        for reg, shifts in reg_time_shift.items():
            reg_shift_ds = regressor_shift(regressors[reg],
                                           including_lag0=False,
                                           shifts=shifts)
            ds_list.append(reg_shift_ds)
        ds = xr.merge(ds_list)
        regressors = xr.merge([regressors, ds])
        # now drop nan bc of the shifts:
        regressors = regressors.dropna('time')
    if species != 'h2o' and species != 'o3':
        swoosh_field = None
    if swoosh_field is not None:
        # load swoosh from work dir:
        ds = xr.load_dataset(path / params.original_data_file)
        if verbose:
            print('loading SWOOSH file: {}'.format(params.original_data_file))
        field = swoosh_field + species + 'q'
        if verbose:
            print('loading SWOOSH field: {}'.format(field))
    elif (swoosh_field is None and (species == 'h2o' or species == 'o3')):
        msg = 'species is {}, and original_data_file is not SWOOSH'.format(species)
        msg += ',Do you want to continue?'
        if click.confirm(msg, default=True):
            ds = xr.load_dataset(path / params.original_data_file)
            if verbose:
                print('loading file: {}'.format(params.original_data_file))
            field = species
    else:
        ds = xr.load_dataset(path / params.original_data_file)
        if verbose:
            print('loading file: {}'.format(params.original_data_file))
        field = species
    da = ds[field]
    # selecting time period:
    if time_period is not None:
        if verbose:
            print('selecting data time period:{} to {}'.format(*time_period))
        da = da.sel(time=slice(*time_period))
    # align time between y and X:
    new_time = aux.overlap_time_xr(da, regressors.to_array())
    regressors = regressors.sel(time=new_time)
    da = da.sel(time=new_time)
    # slice to level and latitude:
    if verbose:
        print('selecting latitude area: {} to {}'.format(*lat_slice))
    if da.level.size > 1:
        da = da.sel(level=slice(100, 1), lat=slice(*lat_slice))
    else:
        da = da.sel(lat=slice(*lat_slice))
    # select seasonality:
    if season is not None:
        if verbose:
            print('selecting season: {}'.format(season))
        da = da.sel(time=da['time.season'] == season)
        regressors = regressors.sel(time=regressors['time.season'] == season)
    # area_mean:
    if area_mean:
        if verbose:
            print('selecting data area mean')
        # da = aux.xr_weighted_mean(da)
        if 'lon' in da.dims:
            da = da.mean('lon', keep_attrs=True)
        da = aux.lat_mean(da)
    if plevels is not None:
        da = da.sel(level=plevels, method='nearest').expand_dims('level')
    # remove nans from y:
    da = aux.remove_nan_xr(da, just_geo=False)
    regressors = regressors.sel(time=da.time)
    # deseason y
    if season is not None:
        da = aux.deseason_xr(da, how='mean', season=season, verbose=False)
    else:
        da = aux.deseason_xr(da, how='mean', verbose=False)
    # saving attrs:
    attrs = [da[dim].attrs for dim in da.dims]
    da.attrs['coords_attrs'] = dict(zip(da.dims, attrs))
    # stacking reg:
    reg_names = [x for x in regressors.data_vars.keys()]
    reg_stacked = regressors[reg_names].to_array(dim='regressors').T
    if poly is not None:
        reg_stacked = poly_features(reg_stacked, degree=poly)
    # da stacking:
    dims_to_stack = [x for x in da.dims if x != 'time']
    da = da.stack(samples=dims_to_stack)
    # time slice:
    return reg_stacked, da


#def pre_proccess(params):
#    """ pre proccess the data, area_mean is reducing the data to level-time,
#    time_period is a slicer of the specific period. lag is inside params and
#    shifting the data with minus"""
#    import aux_functions_strat as aux
#    # import numpy as np
#    import xarray as xr
#    import os
#    path = os.getcwd() + '/regressors/'
#    reg_file = params.regressors_file
#    reg_list = params.regressors
#    reg_add_sub = params.reg_add_sub
#    poly = params.poly_features
#    # load X i.e., regressors
#    regressors = xr.open_dataset(path + reg_file)
#    # unpack params to vars
#    dname = params.data_name
#    species = params.species
#    season = params.season
#    shift = params.time_shift
#    lat_slice = params.lat_slice
#    # model = params.model
#    special_run = params.special_run
#    time_period = params.time_period
#    area_mean = params.area_mean
##    if special_run == 'mask-pinatubo':
##        # mask vol: drop the time dates:
##        # '1991-07-01' to '1996-03-01'
##        date_range = pd.date_range('1991-07-01', '1996-03-01', freq='MS')
##        regressors = regressors.drop(date_range, 'time')
##        aux.text_yellow('Dropped ~five years from regressors due to\
##                        pinatubo...')
#    path = params.work_path
#    if params.run_on_cluster:
#        path = '/data11/ziskin/'
#    # load y i.e., swoosh or era5
#    if dname == 'era5':
#        nc_file = 'ERA5_' + species + '_all.nc'
#        da = xr.open_dataarray(path + nc_file)
#    elif dname == 'merra':
#        nc_file = 'MERRA_' + species + '.nc'
#        da = xr.open_dataarray(path + nc_file)
#    elif dname == 'swoosh':
#        ds = xr.open_dataset(path / params.original_data_file)
#        field = params.swoosh_field + species + 'q'
#        da = ds[field]
#    # align time between y and X:
#    new_time = aux.overlap_time_xr(da, regressors.to_array())
#    regressors = regressors.sel(time=new_time)
#    da = da.sel(time=new_time)
#    if time_period is not None:
#        print('selecting time period: {} to {}'.format(time_period[0],
#              time_period[1]))
#        regressors = regressors.sel(time=slice(*time_period))
#        da = da.sel(time=slice(*time_period))
#    # slice to level and latitude:
#    if dname == 'swoosh' or dname == 'era5':
#        print('selecting latitude area: {} to {}'.format(lat_slice[0],
#              lat_slice[1]))
#        da = da.sel(level=slice(100, 1), lat=slice(lat_slice[0], lat_slice[1]))
#    elif dname == 'merra':
#        da = da.sel(level=slice(100, 0.1), lat=slice(lat_slice[0],
#                    lat_slice[1]))
#    # select seasonality:
#    if season != 'all':
#        da = da.sel(time=da['time.season'] == season)
#        regressors = regressors.sel(time=regressors['time.season'] == season)
#    # area_mean:
#    if area_mean:
#        da = aux.xr_weighted_mean(da)
#    # normalize X
#    regressors = regressors.apply(aux.normalize_xr, norm=1,
#                                  keep_attrs=True, verbose=False)
#    # remove nans from y:
#    da = aux.remove_nan_xr(da, just_geo=False)
#    # regressors = regressors.sel(time=da.time)
#    if 'seas' in field:
#        how = 'mean'
#    else:
#        how = 'std'
#    # deseason y
#    if season != 'all':
#        da = aux.deseason_xr(da, how=how, season=season, verbose=False)
#    else:
#        da = aux.deseason_xr(da, how=how, verbose=False)
#    # shift according to scheme
##    if params.shift is not None:
##        shift_da = create_shift_da(sel=params.shift)
##        da = xr_shift(da, shift_da, 'time')
##        da = da.dropna('time')
##        regressors = regressors.sel(time=da.tioptimize_time_shiftme)
#    # saving attrs:
#    attrs = [da[dim].attrs for dim in da.dims]
#    da.attrs['coords_attrs'] = dict(zip(da.dims, attrs))
#    # stacking reg:
#    reg_names = [x for x in regressors.data_vars.keys()]
#    reg_stacked = regressors[reg_names].to_array(dim='regressors').T
#    # reg_select behaviour:
#    reg_select = [x for x in reg_stacked.regressors.values]
#    if reg_list is not None:  # it is the default
#        # convert str to list:
#        if isinstance(regressors, str):
#            regressors = regressors.split(' ')
#        # select regressors:
#        reg_select = [x for x in reg_list]
#    else:
#        reg_select = [x for x in reg_stacked.regressors.values]
#    # if i want to add or substract regressor:
#    if reg_add_sub is not None:
#        if 'add' in reg_add_sub.keys():
#            to_add = reg_add_sub['add']
#            if isinstance(to_add, str):
#                to_add = [to_add]
#            reg_select = list(set(reg_select + to_add))
#        if 'sub' in reg_add_sub.keys():
#            to_sub = reg_add_sub['add']
#            if isinstance(to_sub, str):
#                to_sub = [to_sub]
#            reg_select = list(set(reg_select).difference(set(to_sub)))
##        # make sure that reg_add_sub is 2-tuple and consist or str:
##        if (isinstance(reg_add_sub, tuple) and
##                len(reg_add_sub) == 2):
##            reg_add = reg_add_sub[0]
##            reg_sub = reg_add_sub[1]
##            if reg_add is not None and isinstance(reg_add, str):
##                reg_select = [x for x in reg_select if x != reg_add]
##                reg_select.append(reg_add)
##            if reg_sub is not None and isinstance(reg_sub, str):
##                reg_select = [x for x in reg_select if x != reg_sub]
##        else:
##            raise ValueError("Expected reg_add_sub as an 2-tuple of string"
##                             ". Got %s." % reg_add_sub)
#    try:
#        reg_stacked = reg_stacked.sel({'regressors': reg_select})
#    except KeyError:
#        raise KeyError('The regressor selection cannot find %s' % reg_select)
#    # poly features:
#    if poly is not None:
#        reg_stacked = poly_features(reg_stacked, degree=poly)
#    # shift all da shift months:
#    if shift is not None:
#        da = da.shift({'time': shift})
#        da = da.dropna('time')
#        reg_stacked = reg_stacked.sel(time=da.time)
#    # special run case:
#    if special_run is not None and 'level_shift' in special_run.keys():
#        scheme = special_run['level_shift']
#        print('Running with special mode: level_shift , ' +
#              'with scheme: {}'.format(scheme))
#        shift_da = create_shift_da(path=params.work_path,
#                                   shift_scheme=scheme)
#        da = xr_shift(da, shift_da, 'time')
#        da = da.dropna('time')
#        reg_stacked = reg_stacked.sel(time=da.time)
#    # da stacking:
#    dims_to_stack = [x for x in da.dims if x != 'time']
#    da = da.stack(samples=dims_to_stack).squeeze()
#    # time slice:
#    return reg_stacked, da


def reg_shift(X, shift_dict):
    """input is X: regressors dataarray, reg_shift: dict with key:
        name to shift, value: how many months to shift"""
    Xcopy = X.copy()
    ds = Xcopy.to_dataset(dim='regressors')
    for regressor, time_shift in shift_dict.items():
        ds[regressor] = ds[regressor].shift(time=time_shift)
    Xcopy = ds.to_array(dim='regressors').dropna('time').T
    return Xcopy


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


def create_shift_da(path, shift_scheme='swoosh1'):
    """create dim_shift_da with certain scheme to be used in xr_shift"""
    import xarray as xr
    import numpy as np
    if shift_scheme == 'swoosh1':
        level = xr.open_dataset(path + 'swoosh_latpress-10deg.nc').level
        level = level.sel(level=slice(83, 10))
        shift = np.array([0,  -2,  -3,  -5,  -7,  -9, -10, -12, -14, -16,
                          -17, -19])
        shift = shift.astype(int)
        da = xr.DataArray(shift, dims='level')
        da['level'] = level
        da.name = 'shift'
    if shift_scheme == 'from_optimization':
        level = xr.open_dataset(path + 'swoosh_latpress-10deg.nc').level
        level = level.sel(level=slice(100, 1))
        shift = np.array([-4,  -6,  -7,  -9, -12, -12, -12,  10,  10,   6,
                          5,   5,   2, -10, -12,  -6,  -6,  -1,  -2, -12, -12,
                          -12, -12, -12, -12])
        shift = shift.astype(int)
        da = xr.DataArray(shift, dims='level')
        da['level'] = level
        da.name = 'shift'
    return da


def xr_shift(da, dim_shift_da, dim_to_shift='time'):
    import xarray as xr
    """shifts da with dim_shift_da dataarray containing dim and 'shift' dim.
    important: this function is called before stacking!"""
    # find the dim that not to be shifted:
    dim = [x for x in dim_shift_da.dims if x != dim_to_shift]
    assert len(dim) == 1
    coord_da = dim_shift_da[''.join(dim)]
    # get the differences between the two dataarrays's coords:
    dif = list(set(da[coord_da.name].values).
               difference(set(dim_shift_da[coord_da.name].values)))
    # shift each coords[i] with the amount spesified in dim_shift_da.values:
    da_list = []
    for i in range(dim_shift_da.size):
        da_list.append(da.sel({coord_da.name: coord_da[i].values},
                              method='nearest').shift({dim_to_shift:
                                                       dim_shift_da[i].
                                                       values.item()}))
    # last: gather all coords that were not shifted
    for d in dif:
        da_list.append(da.sel({dim.name: d}, method='nearest'))
    # combine everything into the same data array and sort:
    da_out = xr.concat(da_list, dim=coord_da.name)
    da_out = da_out.sortby(coord_da.name, ascending=False)
    return da_out


def get_corr_with_regressor(y_like, regressors_ds):
    """calculate the pearson correlation coef for y_like(time, samples) data
    and regressors_ds(time shifted regressors ds)"""
    import numpy as np
    import xarray as xr
    ar = np.empty((y_like.samples.size, len(regressors_ds)))
    for i, sample in enumerate(y_like.samples):
        df = y_like.sel(samples=sample).reset_coords(drop=True).to_dataframe()
        cdf = regressors_ds.reset_coords(drop=True).to_dataframe()
        cdf[df.columns.values.item()] = df
        corr = cdf.corr()[y_like.name].values[:-1]
        ar[i, :] = corr
    corr_da = xr.DataArray(ar, dims=['samples', 'shifts'])
    corr_da['samples'] = y_like.samples
    shifts = [x.split('_')[-1] for x in regressors_ds.data_vars.keys()]
    shifts[0] = 0
    shifts = [int(x) for x in shifts]
    corr_da['shifts'] = shifts
    corr_da = corr_da.unstack('samples')
    return corr_da


def regressor_shift(time_series_da, time_dim='time', shifts=[1, 12],
                    including_lag0=True):
    """shifts time_series_da(an xarray dataarray with time_dim and values) with
    shifts, returns a dataset of shifted time series including the oroginal"""
    import numpy as np
    import xarray as xr
    da_list = []
    if including_lag0:
        da_list.append(time_series_da)
    for shift in np.arange(shifts[0], shifts[1] + 1):
        da = time_series_da.shift({time_dim: shift})
        da.name = time_series_da.name + '_lag_' + str(shift)
        da_list.append(da)
    ds = xr.merge(da_list)
    return ds


def correlate_da_with_lag(return_max=None, return_argmax=None,
                          times=['1994', '2019'], lat_slice=[-60, 60],
                          regress_out=['qbo_1, qbo_2'], max_lag=25):
    from strato_soundings import calc_cold_point_from_sounding
    from sklearn.linear_model import LinearRegression
    radio_cold3 = calc_cold_point_from_sounding(path=sound_path,
                                                times=(times[0], times[1]),
                                                plot=False, return_mean=True)
    p = Parameters()
    p.from_dict({'lat_slice': lat_slice, 'time_period': times})
    X, y = pre_proccess(p)
    radio_cold3.name = 'radio_cpt_anoms_3_stations_randel'
    radio_cold3 = radio_cold3.sel(time=slice(times[0], times[1]))
    radio3_ds = regressor_shift(radio_cold3, shifts=[1, max_lag])
    if regress_out is not None:
        # first regress_out from wv anoms:
        rds = run_ML(species='h2o', swoosh_field='combinedanomfillanom',
                     model_name='LR', time_period=times,
                     regressors=regress_out, lat_slice=lat_slice)
        resid = rds.results_.resid
        dims_to_stack = [x for x in resid.dims if x != 'time']
        y = resid.stack(samples=dims_to_stack).squeeze()
        # now, regress out of radio3:
        X = X.sel(regressors=regress_out)
        lr = LinearRegression()
        lr.fit(X, radio_cold3)
        radio_cold3_regressed_out = radio_cold3 - lr.predict(X)
        radio3_ds = regressor_shift(radio_cold3_regressed_out, shifts=[1, 25])
    else:
        radio3_ds = regressor_shift(radio_cold3, shifts=[1, 25])
    corr_da = get_corr_with_regressor(y, radio3_ds)
    corr_da_max = corr_da.max(dim='shifts')
    corr_da_argmax = corr_da.argmax(dim='shifts')
    if return_max is not None:
        return corr_da_max
    if return_argmax is not None:
        return corr_da_argmax
    return corr_da


def poly_regressor(da, time_dim='time', n=2, plot=False):
    """takes a regressor in DataArray and calculate its polynomial at a
    degree n. if even n, restores the negative sign"""
    negative_time = da[time_dim].where(da < 0, drop=True)
    da_poly = da ** n
    if (n % 2) == 0:
        da_poly.loc[{time_dim: negative_time}] = - da_poly.loc[{time_dim: negative_time}]
    da_poly.name = '{}^{}'.format(da.name, n)
    da_poly.attrs = da.attrs
    if plot:
        da.plot()
        da_poly.plot()
    return da_poly


class Plot_type:
    def __init__(self, *rds, plot_key='predict', lat=None, lon=None,
                 level=None, time=None, regressors=None, time_mean=None,
                 **kwargs):
        self.attrs = rds[0].attrs
        self.plot_key = plot_key
        self.plot_key_list = plot_key.split('_')
        self.lat = lat
        self.lon = lon
        self.level = level
        self.time = time
        self.rds_size = len(rds)
        self.time_dim = rds[0].attrs['sample_dim']
        self.feature_dim = rds[0].attrs['feature_dim']
        self.plot_type = None
        self.regressors = regressors
        self.time_mean = time_mean
        self.plot_map = None
        self.kwargs = kwargs

    def parse_field(self, rds):
        plot_field = self.plot_key_list[0]
        # TODO: implement more fields such as pvalues, dw etc..
        if plot_field == 'predict':
            self.field = ['original', 'predict', 'resid']
            self.plot_type = 'sample'
        elif plot_field == 'r2':
            self.field = ['r2_adj']
            self.plot_type = 'error'
        elif plot_field == 'RI':
            self.field = ['RI']
            self.plot_type = 'feature'
        elif plot_field == 'params':
            self.field = ['params']
            self.plot_type = 'feature'
        elif plot_field == 'response':
            self.field = ['X', 'params']
            self.plot_type = 'response'
        else:
            return
        if len(self.field) == 1:
            data = rds[self.field[0]]
        else:
            data = rds[self.field]
        return data

    def parse_plot_key(self, rds):
        if len(self.plot_key_list) == 2:
            self.lon_mean = False
            self.lat_mean = False
            if self.plot_key_list[1] == 'map':
                self.plot_map = True
            elif self.plot_key_list[1] == 'lat-time' or self.plot_key_list[1] =='level-lat':
                self.plot_map = False
                self.lon_mean = True
            elif self.plot_key_list[1] == 'lon-time' or self.plot_key_list[1] =='level-lon':
                self.plot_map = False
                self.lat_mean = True
            elif self.plot_key_list[1] == 'level-time':
                self.plot_map = False
                self.lat_mean = True
                self.lon_mean = True
            else:
                self.plot_map = False
                self.show_options()
            if 'level' in self.plot_key_list[1]:
                self.level = None
        else:
            self.show_options()
            raise Exception('pls pick two items for plot_key e.g., predict_lat')
        self.lat_in_rds = 'lat' in rds.dims
        self.lon_in_rds = 'lon' in rds.dims
        if (not self.lat_in_rds or not self.lon_in_rds) and self.plot_map:
            raise Exception('lat or lon not in rds.dims and map requested')
#        if self.lat is None and not self.plot_map:
#            self.lat_mean = True
#        if self.lon is None and not self.plot_map:
#            self.lon_mean = True
        if not self.lon_in_rds:
            self.lon_mean = False
        if not self.lat_in_rds:
            self.lat_mean = False

    def parse_coord(self, rds, coord='time'):
        coord_val = getattr(self, coord)
        if coord_val is not None and coord in rds.dims:
            if isinstance(coord_val, list):
                if coord != 'time':
                    coord_val = [float(x) for x in coord_val if x is not None]
                else:
                    coord_val = [str(x) for x in coord_val if x is not None]
                setattr(self, coord, coord_val)
                if len(coord_val) == 1:
                    setattr(self, coord, coord_val[0])
                    data = rds.sel({coord: coord_val}, method='nearest')
                    setattr(self, coord, data[coord].sel({coord: coord_val}, method='nearest').values.item())
                elif len(coord_val) == 2:
                    if coord != 'time':
                        data = rds.sel({coord: slice(*coord_val)})
                        setattr(self, coord, [data[coord].sel({coord: x}, method='nearest').values.item() for x in coord_val])
                    else:
                        import pandas as pd
                        coord_val = [pd.to_datetime(x) for x in coord_val]
                        data = rds.sel({coord: slice(*coord_val)})
                        setattr(self, coord, [data[coord].sel({coord: x}, method='nearest').values for x in coord_val])
            else:
                if coord != 'time':
                    setattr(self, coord, str(coord_val))
                    coord_val = getattr(self, coord)
                    data = rds.sel({coord: coord_val}, method='nearest').squeeze()
                    setattr(self, coord, data[coord].values.item())
                else:
                    import pandas as pd
                    setattr(self, coord, str(coord_val))
                    coord_val = getattr(self, coord)
                    data = rds.sel({coord: coord_val}, method='nearest').squeeze()
                    if data[self.time_dim].size <= 1:
                        raise Exception('pls pick more than one time point')
                    vals = [pd.to_datetime(x) for x in data[coord].sel({coord: coord_val}).values.tolist()]
                    setattr(self, coord, vals)
            return data
        elif coord not in rds.dims:
            print('Warning: {} not in rds.dims'.format(coord))
            return rds
        elif coord_val is None:
            return rds

    def prepare_to_plot_one_rds(self, rds):
        from aux_functions_strat import lat_mean
        from aux_functions_strat import query_yes_no
        data = self.parse_field(rds)
        self.parse_plot_key(rds)
        data = self.parse_coord(data, self.time_dim)
        data = self.parse_coord(data, 'lat')
        data = self.parse_coord(data, 'lon')
        data = self.parse_coord(data, 'level')
        if 'time' in data.dims:
            data = data.dropna('time')
#        for key, val in vars(self).items():
#            print('{}: {}'.format(key, val))
        if self.plot_type == 'sample':
            data = data.to_array(dim='opr', name='name')
            self.time = self.get_coord_limits(data, 'time')
            # choose time_mean:
            data = self.parse_time_mean(data)
        elif self.plot_type == 'feature':
            # choose regressors:
            if self.regressors is not None:
                data = data.sel({self.feature_dim: self.regressors})
            if data[self.feature_dim].size > 5:
                ok = query_yes_no('regressors in data > 5, continue?')
                if not ok:
                    return
        elif self.plot_type == 'response':
            # choose regressors:
            if self.regressors is not None:
                data = data.sel({self.feature_dim: self.regressors})
            if data[self.feature_dim].size > 5:
                ok = query_yes_no('regressors in data > 5, continue?')
                if not ok:
                    return
            # create the response:
            data = data.params * data.X
            self.time = self.get_coord_limits(data, 'time')
            # choose time_mean:
            data = self.parse_time_mean(data)
        # choose lat/lon mean them:
        # self.set_latlon_mean('lat')
        # self.set_latlon_mean('lon')
        # copy attrs:
        for key, value in rds['original'].attrs.items():
            data.attrs[key] = value
        if self.lat_mean and not isinstance(self.lat, float):
            self.lat = self.get_coord_limits(data, 'lat')
            data = lat_mean(data)
        else:
            self.lat_mean = False
        if self.lon_mean and not isinstance(self.lon, float):
            self.lon = self.get_coord_limits(data, 'lon')
            data = data.mean('lon', keep_attrs=True)
        else:
            self.lon_mean = False
        return data

    def get_coord_limits(self, rds, coord):
        coord_val = getattr(self, coord)
        if coord == 'time':
            try:
                mini = rds[coord].min().dt.strftime('%Y-%m').values.item()
                maxi = rds[coord].max().dt.strftime('%Y-%m').values.item()
            except KeyError:
                return None
        else:
            try:
                mini = rds[coord].min().values.item()
                maxi = rds[coord].max().values.item()
            except KeyError:
                return None
        setattr(self, coord, [mini, maxi])
        coord_val = getattr(self, coord)
        return coord_val

    def parse_time_mean(self, rds):
        if self.time_mean is not None:
            if not self.time_mean:
                grp = self.time_dim + '.' + self.time_mean
                attrs = rds.attrs
                data = rds.groupby(grp).mean(self.time_dim)
                for key, value in attrs.items():
                    data.attrs[key] = value
                return data
            else:
                data = rds.mean(self.time_dim, keep_attrs=True)
                return data
        elif self.time_mean is None and rds[self.time_dim].size > 5 and self.plot_map:
            raise Exception('pls pick time_mean(e.g., season) for sample plots with times biggger than 3')
        else:
            return rds
#    def set_latlon_mean(self, coord='lat'):
#        coord_val = getattr(self, coord)
#        if not isinstance(coord_val, list) and coord_val is not None:
#            setattr(self, '{}_mean'.format(coord), False)

    def show_options(self):
        from aux_functions_strat import text_blue, text_green, text_yellow, text_white_underline
        print(text_white_underline('Available options for main plot_key='),
              text_blue(' main-key'), '_', text_green('geo-key'))
        print(
            text_blue('predict: '),
            'Original and reconstructed time-series and residuals with geo-keys:')
        print(text_green('  level-time: '), 'a pressure level vs. time plot.')
        print(text_green('  lat-time: '), 'a latitude vs. time plot.')
        print(text_green('  lon-time: '), 'a longitude vs. time plot.')
        print(text_green('  map: '), 'a longitude vs. latitude plot.')
        print(text_blue('RI: '), 'relative impact with geo-keys:')
        print(text_green('  level-lat: '),
              'a pressure level vs. latitude plot of each regressor.')
        print(text_green('  level-lon: '),
              'a pressure level vs. longitude plot of each regressor.')
        print(
            text_green('  map: '),
            'a longitude vs. latitude plot of each regressor.')
        print(text_blue('params: '), 'beta coeffs with geo-keys:')
        print(text_green('  level-lat: '),
              'a pressure level vs. latitude plot of each regressor.')
        print(text_green('  level-lon: '),
              'a pressure level vs. longitude plot of each regressor.')
        print(
            text_green('  map: '),
            'a longitude vs. latitude plot of each regressor.')
        print(text_blue('r2: '), 'Adjusted R^2 with geo-keys:')
        print(text_green('  level-lat: '),
              'a pressure level vs. latitude plot.')
        print(text_green('  level-lon: '),
              'a pressure level vs. longitude plot.')
        print(
            text_green('  map: '),
            'a longitude vs. latitude plot for a specific pressure level.')
        print(text_blue('response: '), 'beta coeffs * regressors with geo-keys:')
        print(text_green('  level-lat: '),
              'a pressure level vs. latitude response plot of each regressor.')
        print(text_green('  level-lon: '),
              'a pressure level vs. longitude response plot of each regressor.')
        print(
            text_green('  map: '),
            'a longitude vs. latitude respone plot of each regressor.')


def plot_like_results(*results, plot_key='predict_level', level=None,
                      res_label=None, cartopy=False, no_colorbar=False,
                      **kwargs):
    """flexible plot like function for results_ xarray attr from run_ML.
    input: plot_type - dictionary of key:plot type, value - depending on plot,
    e.g., plot_type={'predict_by_lat': 82} will plot prediction + original +
    residualas by a specific pressure level"""
    # TODO: improve predict_map_by_time_level to have single month display or
    # season
    from matplotlib.ticker import ScalarFormatter
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt
    import aux_functions_strat as aux
    import xarray as xr
    import pandas as pd
    import numpy as np
    import warnings
    import cartopy.crs as ccrs
    warnings.filterwarnings("ignore")
    arg_dict = locals()
    keys_to_remove = ['aux', 'np', 'pd', 'xr', 'mdates', 'ScalarFormatter',
                      'plt', 'res_label', 'results', 'warnings', 'ccrs']
    [arg_dict.pop(key) for key in keys_to_remove]
    arg_dict.update(**kwargs)
    p = Plot_type(*results, **arg_dict)
    proj = ccrs.PlateCarree(central_longitude=0)
    plt_kwargs = {'cmap': 'bwr', 'figsize': (15, 10),
                  'add_colorbar': False, 'levels': 41,
                  'extend': 'both'}
    if cartopy:
        plt_kwargs.update({'subplot_kws': dict(projection=proj),
                           'transform': ccrs.PlateCarree()})
    if len(results) == 1:
        rds = results[0]
        data = p.prepare_to_plot_one_rds(rds)
        key = p.plot_key_list[0]
        geo_key = p.plot_key_list[1]
        # if key == 'predict_by_level':
        if key == 'predict':
            plt_kwargs.update({'yscale': 'log', 'yincrease': False, 'cmap':
                               'bwr',
                               'figsize': (15, 10), 'add_colorbar': False,
                               'extend': 'both'})
            if geo_key == 'level-time':
                if p.lat is not None:
                    label_add = ' at lat={}'.format(p.lat)
                else:
                    if p.lat_mean:
                        label_add = ', area mean of latitudes: {} to {}'.format(
                            p.lat[0], p.lat[1])
                    if p.lon_mean:
                        label_add += ', area mean of longitudes: {} to {}'.format(
                            p.lon[0], p.lon[1])
                data = aux.xr_reindex_with_date_range(data, time_dim='time',
                                                      drop=True, freq='MS')
                # , 'vmax': cmap_max})
                plt_kwargs.update({'center': 0.0, 'levels': 41})
                plt_kwargs.update(kwargs)
                fg = data.T.plot.contourf(row='opr', **plt_kwargs)
                if not no_colorbar:
                    cbar_ax = fg.fig.add_axes([0.1, 0.1, .8, .025])
                    fg.add_colorbar(
                        cax=cbar_ax, orientation="horizontal", label=data.attrs['units'],
                        format='%0.3f')
                fg.fig.suptitle(res_label, fontsize=12, fontweight=750)
    #            cb = con.colorbar
    #            cb.set_label(da.sel(opr='original').attrs['units'], fontsize=10)
                labels = [' original', ' reconstructed', ' residuals']
                for i, ax in enumerate(fg.axes.flat):
                    ax.set_title(data.attrs['long_name'] + labels[i] + label_add,
                                 loc='center')
                # plt_kwargs.update({'extend': 'both'})
                ax = [ax for ax in fg.axes.flat][2]
                fg.fig.subplots_adjust(bottom=0.2, top=0.9, left=0.05)
                # [ax.invert_yaxis() for ax in con.ax.figure.axes]
                [ax.invert_yaxis() for ax in fg.axes.flat]
    #            [ax.yaxis.set_major_formatter(ScalarFormatter()) for ax in
    #             con.ax.figure.axes]
                [ax.yaxis.set_major_formatter(ScalarFormatter()) for ax in
                 fg.axes.flat]
                # [ax.xaxis.grid(True, which='minor') for ax in con.ax.figure.axes]
                plt.minorticks_on()
                # ax.xaxis.set_minor_locator(mdates.YearLocator())
                # ax.xaxis.set_minor_formatter(mdates.DateFormatter("\n%Y"))
                plt.setp(
                    ax.xaxis.get_majorticklabels(),
                    rotation=30,
                    ha='center')
                # plt.setp(ax.xaxis.get_minorticklabels(), rotation=30, ha='center')
                # plt.setp(ax.get_xticklabels(), rotation=30, ha="center")
                plt.show()
                return fg
            elif geo_key == 'lat-time' or geo_key == 'lon-time':
                label_add = ''
                if p.level is not None:
                    label_add = ' at level= {:.2f} hPa'.format(p.level)
                else:
                    raise Exception('pls pick a level for this plot')
                if geo_key == 'lat-time':
                    if p.lon is not None and not p.lon_mean:
                        label_add = ' at lon={}'.format(p.lon)
                    elif p.lon_mean:
                        label_add += ', area mean of longitudes: {} to {}'.format(
                            p.lon[0], p.lon[1])
                elif geo_key == 'lon-time':
                    if p.lat is not None and not p.lat_mean:
                        label_add = ' at lat={}'.format(p.lat)
                    elif p.lat_mean:
                        label_add += ', area mean of latitudes: {} to {}'.format(
                            p.lat[0], p.lat[1])
                data = aux.xr_reindex_with_date_range(data, time_dim='time',
                                                      drop=True, freq='MS')
                plt_kwargs.update({'center': 0.0, 'levels': 41})
                plt_kwargs.update({'yscale': 'linear', 'yincrease': True})
                plt_kwargs.update(kwargs)
                fg = data.T.plot.contourf(row='opr', **plt_kwargs)
                if not no_colorbar:
                    cbar_ax = fg.fig.add_axes([0.1, 0.1, .8, .025])
                    fg.add_colorbar(
                        cax=cbar_ax, orientation="horizontal", label=data.attrs['units'],
                        format='%0.3f')
                fg.fig.suptitle(res_label, fontsize=12, fontweight=750)
    #            cb = con.colorbar
    #            cb.set_label(da.sel(opr='original').attrs['units'], fontsize=10)
                labels = [' original', ' reconstructed', ' residuals']
                for i, ax in enumerate(fg.axes.flat):
                    ax.set_title(data.attrs['long_name'] + labels[i] + label_add,
                                 loc='center')
                # plt_kwargs.update({'extend': 'both'})
                ax = [ax for ax in fg.axes.flat][2]
                fg.fig.subplots_adjust(bottom=0.2, top=0.9, left=0.05)
                # [ax.xaxis.grid(True, which='minor') for ax in con.ax.figure.axes]
                plt.minorticks_on()
                # ax.xaxis.set_minor_locator(mdates.YearLocator())
                # ax.xaxis.set_minor_formatter(mdates.DateFormatter("\n%Y"))
                plt.setp(
                    ax.xaxis.get_majorticklabels(),
                    rotation=30,
                    ha='center')
                # plt.setp(ax.xaxis.get_minorticklabels(), rotation=30, ha='center')
                # plt.setp(ax.get_xticklabels(), rotation=30, ha="center")
                plt.show()
                return fg
            elif geo_key == 'map':
                label_add = data.attrs['long_name']
                if p.level is not None:
                    label_add += ' at level= {:.2f} hPa'.format(p.level)
                else:
                    raise Exception('pls pick a level for this plot')
                plt_kwargs.update({'center': 0.0, 'levels': 41})
                plt_kwargs.update({'yscale': 'linear', 'yincrease': True})
                plt_kwargs.update(kwargs)
                if p.time is not None and p.time_mean is not None:
                    if p.time_mean:
                        row = None
                    else:
                        row = p.time_mean
                    # rds = p.parse_coord(rds, 'time')
                    label_add += ', for times {} to {}'.format(p.time[0],
                                                               p.time[1])
                    fg = data.plot.contourf(row=row, col='opr',
                                            **plt_kwargs)
                elif p.time is not None and p.time_mean is None:
                    label_add += ', for times {} to {}'.format(p.time[0],
                                                               p.time[1])
                    fg = data.plot.contourf(
                        row='time', col='opr', **plt_kwargs)
                elif p.time is None:
                    raise Exception(
                        'pls pick a specific time or time range for this plot')
                if not no_colorbar:
                    cbar_ax = fg.fig.add_axes([0.1, 0.1, .8, .025])
                    fg.add_colorbar(
                        cax=cbar_ax, orientation="horizontal", label=data.attrs['units'],
                        format='%0.3f')
                # suptitle = 'time= {}'.format(date)
                fg.fig.suptitle(label_add, fontsize=12, fontweight=750)
    #            cb = con.colorbar
    #            cb.set_label(da.sel(opr='original').attrs['units'], fontsize=10)
                labels = [' original', ' reconstructed', ' residuals']
                try:
                    for i, ax in enumerate(fg.axes.flat):
                        ax.set_title(
                            data.attrs['long_name'] + labels[i], loc='center')
                except IndexError:
                    pass
                if cartopy:
                    [ax.coastlines() for ax in fg.axes.flatten()]
                    [ax.gridlines(
                        crs=ccrs.PlateCarree(),
                        linewidth=1,
                        color='black',
                        alpha=0.5,
                        linestyle='--',
                        draw_labels=False) for ax in fg.axes.flatten()]
                # plt_kwargs.update({'extend': 'both'})
                fg.fig.subplots_adjust(bottom=0.2, top=0.9, left=0.05)
                plt.show()
                return fg
        elif key == 'params' or key == 'RI':
            cmaps = {'params': 'bwr', 'RI': 'viridis'}
            plt_kwargs.update({'cmap': cmaps.get(key), 'figsize': (15, 10),
                               'add_colorbar': False,
                               'extend': 'both'})
            if key == 'params':
                plt_kwargs.update({'center': 0.0, 'levels': 41})  # , 'vmax': cmap_max})
                label_add = r'$\beta$ coefficients'
            elif key == 'RI':
                plt_kwargs.update({'vmin': 0.0, 'levels': 41})  # , 'vmax': cmap_max})
                label_add = 'Relative impact'
            if data.regressors.size > 5:
                plt_kwargs.update(col_wrap = 4)
            else:
                plt_kwargs.update(col_wrap = None)
            if geo_key == 'map':
                plt_kwargs.update(kwargs)
                if p.level is not None:
                    label_add += ' at level= {:.2f} hPa'.format(p.level)
                else:
                    raise Exception('pls pick a level for this plot')
                if 'season' in data.dims:
                    fg = data.plot.contourf(col='regressors', row='season', **plt_kwargs)
                else:
                    fg = data.plot.contourf(col='regressors', **plt_kwargs)
                if not no_colorbar:
                    cbar_ax = fg.fig.add_axes([0.1, 0.1, .8, .025])
                    fg.add_colorbar(
                        cax=cbar_ax, orientation="horizontal", label='',
                        format='%0.3f')
                fg.fig.suptitle(label_add, fontsize=12, fontweight=750)
                if cartopy:
                    [ax.coastlines() for ax in fg.axes.flatten()]
                    [ax.gridlines(
                        crs=ccrs.PlateCarree(),
                        linewidth=1,
                        color='black',
                        alpha=0.5,
                        linestyle='--',
                        draw_labels=False) for ax in fg.axes.flatten()]
    #            cb = con.colorbar
    #            cb.set_label(da.sel(opr='original').attrs['units'], fontsize=10)
                # plt_kwargs.update({'extend': 'both'})
                fg.fig.subplots_adjust(bottom=0.2, top=0.9, left=0.05)
                plt.show()
                return fg
            elif geo_key == 'level-lat' or geo_key == 'level-lon':
                if geo_key == 'level-lat':
                    if p.lon is not None and not p.lon_mean:
                        label_add += ' at lon={}'.format(p.lon)
                    elif p.lon_mean:
                        label_add += ', area mean of longitudes: {} to {}'.format(p.lon[0], p.lon[1])
                elif geo_key == 'level-lon':
                    if p.lat is not None and not p.lat_mean:
                        label_add += ' at lat={}'.format(p.lat)
                    elif p.lat_mean:
                        label_add += ', area mean of latitudes: {} to {}'.format(p.lat[0], p.lat[1])
                plt_kwargs.update({'yscale': 'log', 'yincrease': False})
                plt_kwargs.update(kwargs)
                fg = data.plot.contourf(col='regressors', **plt_kwargs)
                if not no_colorbar:
                    cbar_ax = fg.fig.add_axes([0.1, 0.1, .8, .025])
                    fg.add_colorbar(
                        cax=cbar_ax, orientation="horizontal", label='',
                        format='%0.3f')
                fg.fig.suptitle(label_add, fontsize=12, fontweight=750)
                ax = [ax for ax in fg.axes.flat][2]
                fg.fig.subplots_adjust(bottom=0.2, top=0.9, left=0.05)
                # [ax.invert_yaxis() for ax in con.ax.figure.axes]
                [ax.invert_yaxis() for ax in fg.axes.flat]
                [ax.yaxis.set_major_formatter(ScalarFormatter()) for ax in
                 fg.axes.flat]
    #            cb = con.colorbar
    #            cb.set_label(da.sel(opr='original').attrs['units'], fontsize=10)
                # plt_kwargs.update({'extend': 'both'})
                plt.show()
                return fg
        elif key == 'r2':
            cbar_kwargs = {'format': '%.2f', 'aspect': 50}  # , 'spacing': 'proportional'}
            label_add = r'Adjusted $R^2$'
            plt_kwargs.update({'cmap': 'viridis', 'figsize': (6, 8),
                               'yincrease': False, 'levels': 41, 'vmin': 0.0,
                               'yscale': 'log'})
            if geo_key == 'level-lat' or geo_key == 'level-lon':
                if geo_key == 'level-lat':
                    if p.lon is not None and not p.lon_mean:
                        label_add += ' at lon={}'.format(p.lon)
                    elif p.lon_mean:
                        label_add += ', area mean of longitudes: {} to {}'.format(p.lon[0], p.lon[1])
                elif geo_key == 'level-lon':
                    if p.lat is not None and not p.lat_mean:
                        label_add += ' at lat={}'.format(p.lat)
                    elif p.lat_mean:
                        label_add += ', area mean of latitudes: {} to {}'.format(p.lat[0], p.lat[1])
                plt_kwargs.update(kwargs)
                fg = data.plot.contourf(cbar_kwargs=cbar_kwargs, **plt_kwargs)
                # fg.colorbar.set_label(r'Adjusted $R^2$')
                fg.colorbar.set_label('')
                fg.ax.set_title(label_add, fontsize=12, fontweight=250)
                fg.ax.figure.axes[0].invert_yaxis()
                fg.ax.figure.axes[0].yaxis.set_major_formatter(ScalarFormatter())
                fg.ax.figure.subplots_adjust(bottom=0.1, top=0.90, left=0.1)
                plt.show()
                return fg
            elif geo_key == 'map':
                if p.level is not None:
                    label_add += ' at level= {:.2f} hPa'.format(p.level)
                else:
                    raise Exception('pls pick a level for this plot')
                plt_kwargs.update({'yscale': 'linear', 'yincrease': True})
                plt_kwargs.update(kwargs)
                # fg.colorbar.set_label(r'Adjusted $R^2$')
                if cartopy:
                    plt_kwargs.pop('transform')
                    plt_kwargs.pop('figsize')
                    ax = plt.axes(projection=proj)
                    fg = data.plot.contourf(ax=ax, transform=ccrs.PlateCarree(),
                                            **plt_kwargs)
                    ax.coastlines()
                    ax.gridlines(
                            crs=ccrs.PlateCarree(),
                            linewidth=1,
                            color='black',
                            alpha=0.5,
                            linestyle='--',
                            draw_labels=False)
                else:
                    fg = data.plot.contourf(**plt_kwargs)
                if not no_colorbar:
                    cbar_ax = fg.ax.figure.add_axes([0.1, 0.1, .8, .035])
                    plt.colorbar(fg, cax=cbar_ax, orientation="horizontal", **cbar_kwargs)
                    fg.colorbar.set_label('')
                fg.ax.set_title(label_add, fontsize=12, fontweight=250)
                fg.ax.figure.subplots_adjust(bottom=0.1, top=0.90, left=0.1)
                plt.show()
                return fg
        elif key == 'response':
            try:
                label_add = data.attrs['long_name'] + ' response'
            except KeyError:
                label_add = '{} response'.format(data.name)
            plt_kwargs.update({'cmap': 'bwr', 'figsize': (15, 10),
                               'add_colorbar': False, 'levels': 41,
                               'extend': 'both'})
            if geo_key == 'map':
                if p.level is not None:
                    label_add += ' at level= {:.2f} hPa'.format(p.level)
                else:
                    raise Exception('pls pick a level for this plot')
                try:
                    units = data.attrs['units']
                except KeyError:
                    units = ''
                plt_kwargs.update({'center': 0.0, 'levels': 41})
                plt_kwargs.update(kwargs)
                if p.time is not None and p.time_mean is not None:
                    if p.time_mean:
                        row = None
                    else:
                        row = p.time_mean
                    # rds = p.parse_coord(rds, 'time')
                    label_add += ', for times {} to {}'.format(p.time[0],
                                                               p.time[1])
                    fg = data.plot.contourf(row=row, col='regressors',
                                            **plt_kwargs)
                elif p.time is not None and p.time_mean is None:
                    label_add += ', for times {} to {}'.format(p.time[0],
                                                                p.time[1])
                    fg = data.plot.contourf(row='time', col='regressors',
                                            **plt_kwargs)
                elif p.time is None:
                    raise Exception('pls pick a specific time or time range for this plot')
                if not no_colorbar:
                    cbar_ax = fg.fig.add_axes([0.1, 0.1, .8, .025])
                    fg.add_colorbar(
                        cax=cbar_ax, orientation="horizontal", label=units,
                        format='%0.3f')
                fg.fig.suptitle(label_add, fontsize=12, fontweight=750)
                if cartopy:
                    [ax.coastlines() for ax in fg.axes.flatten()]
                    [ax.gridlines(
                        crs=ccrs.PlateCarree(),
                        linewidth=1,
                        color='black',
                        alpha=0.5,
                        linestyle='--',
                        draw_labels=False) for ax in fg.axes.flatten()]
                fg.fig.subplots_adjust(bottom=0.2, top=0.9, left=0.05)
                return fg
            elif geo_key == 'level-lat' or geo_key == 'level-lon':
                if geo_key == 'level-lat':
                    if p.lon is not None and not p.lon_mean:
                        label_add += ' at lon={}'.format(p.lon)
                    elif p.lon_mean:
                        label_add += ', area mean of longitudes: {} to {}'.format(p.lon[0], p.lon[1])
                elif geo_key == 'level-lon':
                    if p.lat is not None and not p.lat_mean:
                        label_add += ' at lat={}'.format(p.lat)
                    elif p.lat_mean:
                        label_add += ', area mean of latitudes: {} to {}'.format(p.lat[0], p.lat[1])
                plt_kwargs.update({'yscale': 'log', 'yincrease': False})
                plt_kwargs.update(kwargs)
                if p.time is not None and p.time_mean is not None:
                    # rds = p.parse_coord(rds, 'time')
                    label_add += ', for times {} to {}'.format(p.time[0],
                                                               p.time[1])
                    fg = data.plot.contourf(row=p.time_mean, col='regressors',
                                            **plt_kwargs)
                if not no_colorbar:
                    cbar_ax = fg.fig.add_axes([0.1, 0.1, .8, .025])
                    fg.add_colorbar(
                        cax=cbar_ax, orientation="horizontal", label='',
                        format='%0.3f')
                fg.fig.suptitle(label_add, fontsize=12, fontweight=750)
                ax = [ax for ax in fg.axes.flat][2]
                fg.fig.subplots_adjust(bottom=0.2, top=0.9, left=0.05)
                # [ax.invert_yaxis() for ax in con.ax.figure.axes]
                [ax.invert_yaxis() for ax in fg.axes.flat]
                [ax.yaxis.set_major_formatter(ScalarFormatter()) for ax in
                 fg.axes.flat]
                plt.show()
                return fg
    elif len(results) > 1:
        data = [p.prepare_to_plot_one_rds(rds) for rds in results]
        key = p.plot_key_list[0]
        geo_key = p.plot_key_list[1]
        # concat the results da's togather:
        data = xr.concat(data, 'model')
        # add labels to models:
        if res_label is None:
            data['model'] = ['result_{}'.format(x) for x in range(len(results))]
        else:
            data['model'] = res_label
        if key == 'r2':
            cbar_kwargs = {'format': '%.2f', 'aspect': 50, 'label': ''}  # , 'spacing': 'proportional'}
            label_add = r'Adjusted $R^2$'
            plt_kwargs.update({'cmap': 'viridis', 'figsize': (15, 10),
                               'yincrease': False, 'levels': 41, 'vmin': 0.0,
                               'yscale': 'log'})
            if geo_key == 'map':
                if p.level is not None:
                    label_add += ' at level= {:.2f} hPa'.format(p.level)
                else:
                    raise Exception('pls pick a level for this plot')
                plt_kwargs.update({'yscale': 'linear', 'yincrease': True})
                plt_kwargs.update(kwargs)
                # fg.colorbar.set_label(r'Adjusted $R^2$')
                fg = data.plot.contourf(col='model', **plt_kwargs)
                if cartopy:
                    [ax.coastlines() for ax in fg.axes.flatten()]
                    [ax.gridlines(
                        crs=ccrs.PlateCarree(),
                        linewidth=1,
                        color='black',
                        alpha=0.5,
                        linestyle='--',
                        draw_labels=False) for ax in fg.axes.flatten()]
                if not no_colorbar:
                    cbar_ax = fg.fig.add_axes([0.1, 0.1, .8, .035])
                    fg.add_colorbar(cax=cbar_ax, orientation="horizontal", **cbar_kwargs)
                fg.fig.suptitle(label_add, fontsize=12, fontweight=250)
                fg.fig.subplots_adjust(bottom=0.1, top=0.90, left=0.1)
                plt.show()
                return fg
        elif key == 'RI' or key == 'params':
            cmaps = {'params': 'bwr', 'RI': 'viridis'}
            plt_kwargs.update({'cmap': cmaps.get(key), 'figsize': (15, 10),
                               'add_colorbar': False,
                               'extend': 'both'})
            if key == 'params':
                plt_kwargs.update({'center': 0.0, 'levels': 41})  # , 'vmax': cmap_max})
                label_add = r'$\beta$ coefficients'
            elif key == 'RI':
                plt_kwargs.update({'vmin': 0.0, 'levels': 41})  # , 'vmax': cmap_max})
                label_add = 'Relative impact'
            if data.regressors.size > 5:
                plt_kwargs.update(col_wrap = 4)
            else:
                plt_kwargs.update(col_wrap = None)
            if geo_key == 'map':
                plt_kwargs.update(kwargs)
                if p.level is not None:
                    label_add += ' at level= {:.2f} hPa'.format(p.level)
                else:
                    raise Exception('pls pick a level for this plot')
                fg = data.plot.contourf(row='model', col='regressors', **plt_kwargs)
                if not no_colorbar:
                    cbar_ax = fg.fig.add_axes([0.1, 0.1, .8, .025])
                    fg.add_colorbar(
                        cax=cbar_ax, orientation="horizontal", label='',
                        format='%0.3f')
                fg.fig.suptitle(label_add, fontsize=12, fontweight=750)
                if cartopy:
                    [ax.coastlines() for ax in fg.axes.flatten()]
                    [ax.gridlines(
                        crs=ccrs.PlateCarree(),
                        linewidth=1,
                        color='black',
                        alpha=0.5,
                        linestyle='--',
                        draw_labels=False) for ax in fg.axes.flatten()]
    #            cb = con.colorbar
    #            cb.set_label(da.sel(opr='original').attrs['units'], fontsize=10)
                # plt_kwargs.update({'extend': 'both'})
                fg.fig.subplots_adjust(bottom=0.2, top=0.9, left=0.05)
                plt.show()
                return fg


#def plot_like_results_decrapeted(*results, plot_type={'predict_by_level': 'mean'},
#                      res_label=None, **kwargs):
#    """flexible plot like function for results_ xarray attr from run_ML.
#    input: plot_type - dictionary of key:plot type, value - depending on plot,
#    e.g., plot_type={'predict_by_lat': 82} will plot prediction + original +
#    residualas by a specific pressure level"""
#    # TODO: improve predict_map_by_time_level to have single month display or season
#    from matplotlib.ticker import ScalarFormatter
#    import matplotlib.dates as mdates
#    import matplotlib.pyplot as plt
#    import aux_functions_strat as aux
#    import xarray as xr
#    import pandas as pd
#    import numpy as np
#    key = [x for x in plot_type.keys()][0]
#    val = [x for x in plot_type.values()][0]
#    if len(results) == 1:
#        rds = results[0]
#        # only one run plots:
#        if key == 'predict_by_level':
#            # define plot kwargs:
#            plt_kwargs = {'yscale': 'log', 'yincrease': False, 'cmap': 'bwr',
#                          'figsize': (15, 10), 'add_colorbar': False,
#                          'extend': 'both'}
#            # transform into array:
#            da = rds[['original', 'predict', 'resid']].to_array(dim='opr',
#                                                                 name='name')
#            # copy attrs to new da:
#            for key, value in rds['original'].attrs.items():
#                da.attrs[key] = value
#            if val == 'mean':
#                label_add = ', area mean of latitudes: ' +\
#                             str(da.lat.min().values) + ' to ' + \
#                             str(da.lat.max().values)
#                da = aux.xr_weighted_mean(da)
#            else:
#                da = da.sel(lat=val, method='nearest')
#                label_add = ' at lat=' + str(da.lat.values.item())
#            # cmap_max = abs(max(abs(da.sel(opr='original').min().values),
#            #                    abs(da.sel(opr='original').max().values)))
#            da = da.reindex({'time': pd.date_range(da.time[0].values,
#                                                   da.time[-1].values,
#                                                   freq='MS')})
#            plt_kwargs.update({'center': 0.0, 'levels': 41})  # , 'vmax': cmap_max})
#            plt_kwargs.update(kwargs)
#            fg = da.T.plot.contourf(row='opr', **plt_kwargs)
#            cbar_ax = fg.fig.add_axes([0.1, 0.1, .8, .025])
#            fg.add_colorbar(
#                cax=cbar_ax, orientation="horizontal", label=da.attrs['units'],
#                format='%0.3f')
#            fg.fig.suptitle(res_label, fontsize=12, fontweight=750)
##            cb = con.colorbar
##            cb.set_label(da.sel(opr='original').attrs['units'], fontsize=10)
#            labels = [' original', ' reconstructed', ' residuals']
#            for i, ax in enumerate(fg.axes.flat):
#                ax.set_title(da.attrs['long_name'] + labels[i] + label_add,
#                             loc='center')
#            # plt_kwargs.update({'extend': 'both'})
#            ax = [ax for ax in fg.axes.flat][2]
#            fg.fig.subplots_adjust(bottom=0.2, top=0.9, left=0.05)
#            # [ax.invert_yaxis() for ax in con.ax.figure.axes]
#            [ax.invert_yaxis() for ax in fg.axes.flat]
##            [ax.yaxis.set_major_formatter(ScalarFormatter()) for ax in
##             con.ax.figure.axes]
#            [ax.yaxis.set_major_formatter(ScalarFormatter()) for ax in
#             fg.axes.flat]
#            # [ax.xaxis.grid(True, which='minor') for ax in con.ax.figure.axes]
#            plt.minorticks_on()
#            # ax.xaxis.set_minor_locator(mdates.YearLocator())
#            # ax.xaxis.set_minor_formatter(mdates.DateFormatter("\n%Y"))
#            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='center')
#            # plt.setp(ax.xaxis.get_minorticklabels(), rotation=30, ha='center')
#            # plt.setp(ax.get_xticklabels(), rotation=30, ha="center")
#            plt.show()
#            return fg
#        elif key == 'predict_map_by_time_level':
#            plt_kwargs = {'cmap': 'bwr', 'figsize': (15, 10),
#                          'add_colorbar': False,
#                          'extend': 'both'}
#            plt_kwargs.update({'center': 0.0, 'levels': 41})  # , 'vmax': cmap_max})
#            plt_kwargs.update(kwargs)
#            # transform into array:
#            da = rds[['original', 'predict', 'resid']].to_array(dim='opr',
#                                                                name='name')
#            if 'lon' not in da.dims:
#                raise KeyError('no lon in dims!')
#            # copy attrs to new da:
#            for key, value in rds['original'].attrs.items():
#                da.attrs[key] = value
#            if len(val) == 1:
#                date = val[0]
#                suptitle = 'time= {}'.format(date)
#                da = da.sel(time=date, method='nearest').squeeze(drop=True)
#                # fg = da.plot.contourf(col='opr',row='level', **plt_kwargs)
#                fg = da.plot.contourf(col='opr', **plt_kwargs)
#            elif len(val) == 2:
#                # seasonal mean of at least a year:
#                date = val[0]
#                plevel = val[1]
#                suptitle = 'level= {} hPa, year= {}'.format(plevel, date)
#                da = da.sel(time=date, level=plevel, method='nearest').squeeze()
#                da_seasons = da.groupby('time.season').mean('time')
#                fg = da_seasons.plot.contourf(row='season', col='opr', **plt_kwargs)
#            elif len(val) == 3:
#                min_time = val[0]
#                max_time = val[1]
#                plevel = val[2]
#                suptitle = 'level= {} hPa, time= {} to {}'.format(plevel, min_time, max_time)
#                da = da.sel(time=slice(min_time, max_time))
#                da = da.sel(level=plevel, method='nearest').squeeze()
#                # da_seasons = da.groupby('time.season').mean('time')
#                fg = da.plot.contourf(row='time', col='opr', **plt_kwargs)
#            cbar_ax = fg.fig.add_axes([0.1, 0.1, .8, .025])
#            fg.add_colorbar(
#                cax=cbar_ax, orientation="horizontal", label=da.attrs['units'],
#                format='%0.3f')
#            fg.fig.suptitle(suptitle, fontsize=12, fontweight=750)
##            cb = con.colorbar
##            cb.set_label(da.sel(opr='original').attrs['units'], fontsize=10)
#            labels = [' original', ' reconstructed', ' residuals']
#            try:
#                for i, ax in enumerate(fg.axes.flat):
#                    ax.set_title(da.attrs['long_name'] + labels[i], loc='center')
#            except IndexError:
#                pass
#            # plt_kwargs.update({'extend': 'both'})
#            fg.fig.subplots_adjust(bottom=0.2, top=0.9, left=0.05)
#            plt.show()
#            return fg
#        elif key == 'predict_map_by_level':
#            plt_kwargs = {'cmap': 'bwr', 'figsize': (15, 10),
#                          'add_colorbar': False,
#                          'extend': 'both'}
#            plt_kwargs.update({'center': 0.0, 'levels': 41})  # , 'vmax': cmap_max})
#            plt_kwargs.update(kwargs)
#            # transform into array:
#            da = rds[['original', 'predict', 'resid']].to_array(dim='opr',
#                                                                name='name')
#            if 'lon' not in da.dims:
#                raise KeyError('no lon in dims!')
#            # copy attrs to new da:
#            for key, value in rds['original'].attrs.items():
#                da.attrs[key] = value
#            if len(val) == 2:
#                plevel = val[0]
#                if val[1] == 'lat_mean':
#                    from aux_functions_strat import lat_mean
#                    label_add = ', area mean of latitudes: ' +\
#                        str(da.lat.min().values) + ' to ' + \
#                        str(da.lat.max().values)
#                    da = lat_mean(da)
#                    suptitle = 'level= {} hPa'.format(plevel)
#                    da = da.sel(level=plevel, method='nearest').squeeze()
#                elif val[1] == 'lon_mean':
#                    label_add = ', area mean of longitudes: ' +\
#                        str(da.lon.min().values) + ' to ' + \
#                        str(da.lon.max().values)
#                    da = da.mean('lon', keep_attrs=True)
#                    suptitle = 'level= {} hPa'.format(plevel)
#                    da = da.sel(level=plevel, method='nearest').squeeze()
#            else:
#                raise KeyError('pls choose level and either lon_mean or lat_mean')
#            fg = da.T.plot.contourf(row='opr', **plt_kwargs)
#            cbar_ax = fg.fig.add_axes([0.1, 0.1, .8, .025])
#            fg.add_colorbar(
#                cax=cbar_ax, orientation="horizontal", label=da.attrs['units'],
#                format='%0.3f')
#            fg.fig.suptitle(suptitle, fontsize=12, fontweight=750)
##            cb = con.colorbar
##            cb.set_label(da.sel(opr='original').attrs['units'], fontsize=10)
#            labels = [' original', ' reconstructed', ' residuals']
#            try:
#                for i, ax in enumerate(fg.axes.flat):
#                    ax.set_title(da.attrs['long_name'] + labels[i] + label_add,
#                                 loc='center')
#            except IndexError:
#                pass
#            # plt_kwargs.update({'extend': 'both'})
#            fg.fig.subplots_adjust(bottom=0.2, top=0.9, left=0.05)
#            plt.show()
#            return fg
#        elif key == 'params_map_by_level':
#            plt_kwargs = {'cmap': 'bwr', 'figsize': (15, 10),
#                          'add_colorbar': False,
#                          'extend': 'both'}
#            plt_kwargs.update({'center': 0.0, 'levels': 41})  # , 'vmax': cmap_max})
#            plt_kwargs.update(kwargs)
#            # transform into array:
#            da = rds['params']
#            if 'lon' not in da.dims:
#                raise KeyError('no lon in dims!')
#            # copy attrs to new da:
#            if len(val) == 1:
#                plevel = val[0]
#                suptitle = 'level= {} hPa'.format(plevel)
#                da = da.sel(level=plevel, method='nearest').squeeze()
#                fg = da.plot.contourf(col='regressors', **plt_kwargs)
#            else:
#                # seasonal mean of at least a year:
#                plevel = val[0]
#                flist = val[1]
#                suptitle = 'level= {} hPa'.format(plevel)
#                da = da.sel(level=plevel, method='nearest').squeeze()
#                da = da.sel(regressors=flist).squeeze()
#                fg = da.plot.contourf(col='regressors', **plt_kwargs)
#            cbar_ax = fg.fig.add_axes([0.1, 0.1, .8, .025])
#            fg.add_colorbar(
#                cax=cbar_ax, orientation="horizontal", label='coeff',
#                format='%0.3f')
#            fg.fig.suptitle(suptitle, fontsize=12, fontweight=750)
##            cb = con.colorbar
##            cb.set_label(da.sel(opr='original').attrs['units'], fontsize=10)
#            # plt_kwargs.update({'extend': 'both'})
#            fg.fig.subplots_adjust(bottom=0.2, top=0.9, left=0.05)
#            plt.show()
#            return fg
#        elif key == 'predict_by_lat':
#            # define plot kwargs:
#            plt_kwargs = {'cmap': 'bwr',
#                          'figsize': (15, 10), 'add_colorbar': False,
#                          'extend': 'both'}
#            # transform into array:
#            da = rds[['original', 'predict', 'resid']].to_array(dim='opr',
#                                                                 name='name')
#            # copy attrs to new da:
#            for key, value in rds['original'].attrs.items():
#                da.attrs[key] = value
##            if val == 'mean':
##                label_add = ', area mean of latitudes: ' +\
##                             str(da.lat.min().values) + ' to ' + \
##                             str(da.lat.max().values)
##                da = aux.xr_weighted_mean(da)
##            else:
#            da = da.sel(level=val, method='nearest')
#            label_add = ' at level= {:.2f} hPa'.format(da.level.values.item())
#            # cmap_max = abs(max(abs(da.sel(opr='original').min().values),
#            #                    abs(da.sel(opr='original').max().values)))
#            da = da.reindex({'time': pd.date_range(da.time[0].values,
#                                                   da.time[-1].values,
#                                                   freq='MS')})
#            plt_kwargs.update({'center': 0.0, 'levels': 41})  # , 'vmax': cmap_max})
#            plt_kwargs.update(kwargs)
#            fg = da.T.plot.contourf(row='opr', **plt_kwargs)
#            cbar_ax = fg.fig.add_axes([0.1, 0.1, .8, .025])
#            fg.add_colorbar(
#                cax=cbar_ax, orientation="horizontal", label=da.attrs['units'],
#                format='%0.3f')
#            fg.fig.suptitle(res_label, fontsize=12, fontweight=750)
##            cb = con.colorbar
##            cb.set_label(da.sel(opr='original').attrs['units'], fontsize=10)
#            labels = [' original', ' reconstructed', ' residuals']
#            for i, ax in enumerate(fg.axes.flat):
#                ax.set_title(da.attrs['long_name'] + labels[i] + label_add,
#                             loc='center')
#            # plt_kwargs.update({'extend': 'both'})
#            ax = [ax for ax in fg.axes.flat][2]
#            fg.fig.subplots_adjust(bottom=0.2, top=0.9, left=0.05)
#            # [ax.xaxis.grid(True, which='minor') for ax in con.ax.figure.axes]
#            plt.minorticks_on()
#            # ax.xaxis.set_minor_locator(mdates.YearLocator())
#            # ax.xaxis.set_minor_formatter(mdates.DateFormatter("\n%Y"))
#            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='center')
#            # plt.setp(ax.xaxis.get_minorticklabels(), rotation=30, ha='center')
#            # plt.setp(ax.get_xticklabels(), rotation=30, ha="center")
#            plt.show()
#            return fg
#        elif key == 'r2_by_level':
#            plt_kwargs = {'cmap': 'viridis', 'figsize': (6, 8),
#                          'levels': 41, 'vmin': 0.0}
#            plevel = val
#            da = rds['r2_adj'].sel(level=plevel, method='nearest')
#            if 'lon' not in da.dims:
#                raise KeyError('no lon in dims!')
#            plt_kwargs.update(kwargs)
#            fg = da.plot.contourf(**plt_kwargs)
#            fg.ax.figure.suptitle('R^2 adjusted', fontsize=12, fontweight=750)
#            fg.ax.set_title('level = {:.2f} hPa'.format(plevel))
#            plt.show()
#            return fg
#    elif len(results) > 1:
#        if key == 'r2':
#            plt_kwargs = {'yscale': 'log', 'yincrease': False,
#                          'cmap': 'viridis', 'figsize': (6 * len(results), 8),
#                          'levels': 41, 'vmin': 0.0}
#            # transform into array:
#            da_list = [x['r2_adj'] for x in results]
#            for i, da in enumerate(da_list):
#                da.name = res_label[i]
#            ds = xr.merge(da_list)
#            da = ds.to_array(dim='regressors', name='name')
#            if 'lon' in da.dims:
#                if val == 'lon':
#                    da = aux.xr_weighted_mean(da, mean_on_lon=False,
#                                              mean_on_lat=True)
#                else:
#                    da = aux.xr_weighted_mean(da, mean_on_lon=True,
#                                              mean_on_lat=False)
#            da['regressors'] = res_label
#            # copy attrs to new da:
#            for key, value in results[0]['r2_adj'].attrs.items():
#                da.attrs[key] = value
#            plt_kwargs.update(kwargs)
#            fg = da.plot.contourf(col='regressors', **plt_kwargs)
#            fg.fig.subplots_adjust(top=0.92, right=0.82, left=0.05)
#            # [ax.invert_yaxis() for ax in con.ax.figure.axes]
#            for i, ax in enumerate(fg.axes.flat):
#                ax.set_title(res_label[i], loc='center')
#            [ax.invert_yaxis() for ax in fg.axes.flat]
##            [ax.yaxis.set_major_formatter(ScalarFormatter()) for ax in
##             con.ax.figure.axes]
#            [ax.yaxis.set_major_formatter(ScalarFormatter()) for ax in
#             fg.axes.flat]
#            fg.fig.suptitle('R^2 adjusted', fontsize=12, fontweight=750)
#            plt.show()
#            return fg

        # two compareing plots:
    # if 'div' in keys:
    #     cmap = 'bwr'
    # else:
    #     cmap = 'viridis'
class TargetArray(Dataset):
    def __init__(self, *args, sample_dim='time', mt_dim='samples',
                 data_file=None, species='h2o', loadpath=None, verbose=False,
                 season=None, deseason_method=None, time_period=None, field=None,
                 plevels=None, area_mean=None,
                 swoosh_field='combinedanomfillanom', lat_slice=None,
                 **kwargs):
        super().__init__(*args)
        self.loadpath = loadpath
        self.verbose = verbose
        self.deseason_method = deseason_method
        self.sample_dim = sample_dim
        self.mt_dim = mt_dim
        self.time_period = time_period
        self.season = season
        self.species = species
        self.data_file = data_file
        self.swoosh_field = swoosh_field
        self.field = field
        self.plevels = plevels
        self.area_mean = area_mean
        self.lat_slice = lat_slice

    def from_dict(self, d):
        self.__dict__.update(d)
        return self

    def show(self, name='all'):
        from termcolor import colored
        if name == 'all':
            for attr, value in vars(self).items():
                print(colored('{} : '.format(attr), color='blue', attrs=['bold']), end='')
                print(colored(value, color='white', attrs=['bold']))
        elif hasattr(self, name):
            print(colored('{} : '.format(name), color='blue', attrs=['bold']), end='')
            print(colored(self.name, color='white', attrs=['bold']))

    def select_season(self, season):
        to_update = self.__dict__
        self_season = self['{}.season'.format(self.sample_dim)] == season
        self = self.sel({self.sample_dim: self_season})
        self.__dict__.update(to_update)
        self.season = season
        return self

    def select_times(self, times):
        to_update = self.__dict__
        self = self.sel({self.sample_dim: slice(*times)})
        self.__dict__.update(to_update)
        self.attrs['times'] = times
        self.time_period = times
        return self

    def lat_level_select(self, lat_slice):
        to_update = self.__dict__
        if self['level'].size > 1:
            self = self.sel(
                level=slice(
                    100, 1), lat=slice(
                    *lat_slice))
        else:
            self = self.sel(lat=slice(*lat_slice))
        self.__dict__.update(to_update)
        self.lat_slice = lat_slice
        return self

    def load(self, loadpath=None, data_file=None, species=None,
             swoosh_field=None, verbose=None):
        import xarray as xr
        import click
        # read all locals() and replace Nons vals with defualts from class:
        vars_dict = {}
        for key, val in locals().items():
            if val is None:
                vars_dict[key] = getattr(self, key)
        loadpath = vars_dict.get('loadpath')
        data_file = vars_dict.get('data_file')
        species = vars_dict.get('species')
        swoosh_field = vars_dict.get('swoosh_field')
        verbose = vars_dict.get('verbose')
        if species != 'h2o' and species != 'o3':
            swoosh_field = None
        if swoosh_field is not None:
            # load swoosh from work dir:
            field = '{}{}q'.format(swoosh_field, species)
            ds = xr.load_dataset(loadpath / data_file)[field]
            if verbose:
                print('loading SWOOSH file: {}'.format(data_file))
            if verbose:
                print('loading SWOOSH field: {}'.format(field))
        elif (swoosh_field is None and (species == 'h2o' or species == 'o3')):
            msg = 'species is {}, and original_data_file is not SWOOSH'.format(
                species)
            msg += ',Do you want to continue?'
            if click.confirm(msg, default=True):
                field = species
                ds = xr.load_dataset(
                    loadpath /
                    data_file)[
                    field]
                if verbose:
                    print('loading file: {}'.format(data_file))
        else:
            field = species
            ds = xr.load_dataset(loadpath / data_file)[field]
            if self.verbose:
                print('loading file: {}'.format(data_file))
        ds = ds.to_dataset(name=field)
        self = TargetArray(ds, **vars(self))
        assert self.sample_dim in [x for x in self.dims]
        self.field = field
        return self

    def remove_nans(self, field):
        from aux_functions_strat import remove_nan_xr
        self_da = remove_nan_xr(self[field], just_geo=False)
        return self_da

    def normalize(self, norm):
        from aux_functions_strat import normalize_xr
        ds = normalize_xr(
            self,
            norm=norm,
            verbose=self.verbose)  # 1 is mean/std
        self = TargetArray(ds, **vars(self))
        self.attrs = ds.attrs
        return self

    def deseason(self, how):
        from aux_functions_strat import deseason_xr
        if self.season is not None:
            self[self.field] = deseason_xr(self[self.field],
                                           season=self.season, how=how,
                                           verbose=self.verbose)
        else:
            self[self.field] = deseason_xr(self[self.field], how=how,
                                           verbose=self.verbose)
        self.deseason_method = how
        return self

    def do_area_mean(self):
        from aux_functions_strat import lat_mean
        to_update = self.__dict__
        if self.verbose:
            print('selecting data area mean')
        self = lat_mean(self)
        if 'lon' in self.dims:
            self = self.mean('lon', keep_attrs=True)
        self.__dict__.update(to_update)
        self.area_mean = True
        return self

    def select_plevels(self, plevels):
        to_update = self.__dict__
        self = self.sel(
                    level=plevels,
                    method='nearest').expand_dims('level')
        self.__dict__.update(to_update)
        self.plevels = plevels
        return self

    def save_dim_attrs(self):
        attrs = [self[dim].attrs for dim in self.dims]
        self.attrs['coords_attrs'] = dict(zip(self.dims, attrs))
        return self

    def mt_dim_stack(self):
        dims_to_stack = [x for x in self.dims if x != self.sample_dim]
        y = self[self.field].stack({self.mt_dim: dims_to_stack})
        return y

    def pre_process(self, stack=True):
        # flow of target pre-processing:
        # 1) load all predictors
        self = self.load()
        # 2) select times:
        self = self.select_times(self.time_period)
        # 3) lat slice and level:
        self = self.lat_level_select(self.lat_slice)
        # 4) select season:
        if self.season is not None:
            self = self.select_season(self.season)
        # 5) area mean:
        if self.area_mean:
            self = self.do_area_mean()
        # 6) select plevels:
        if self.plevels is not None:
            self = self.select_plevels(self.plevels)
        # 7) deseason
        if self.deseason_method is not None:
            self = self.deseason(how=self.deseason_method)
        # 8) save dim attrs:
        self = self.save_dim_attrs()
        attrs = self.attrs
        # 9) remove nans:
        self = TargetArray(self.remove_nans(self.field).to_dataset(name=self.field), **vars(self))
        self.attrs = attrs
        if stack:
            y = self.mt_dim_stack()
            y.attrs = self.attrs
            for key, val in self[self.field].attrs.items():
                y.attrs[key] = val
            return y
        else:
            return self


def sort_predictors_list(pred_list):
    import re
    inter_splitted = [re.split("[*]+", x) for x in pred_list]
    new_list = []
    for item in inter_splitted:
        if len(item) > 1:
            new_item = sorted(item)
            new_list.append('*'.join(new_item))
        elif len(item) == 1:
            new_list.append(item[0])
    return new_list


class PredictorSet(Dataset):
    def __init__(self, *args, sample_dim='time', feature_dim='regressors',
                 loadpath=None, verbose=False, reg_shift=None, season=None,
                 deseason_dict=None, regressors=None, time_period=None,
                 **kwargs):
        super().__init__(*args)
        self.loadpath = loadpath
        self.verbose = verbose
        self.deseason_dict = deseason_dict
        self.sample_dim = sample_dim
        self.feature_dim = feature_dim
        self.regressors = regressors
        self.time_period = time_period
        self.reg_shift = reg_shift
        self.season = season

    def from_dict(self, d):
        self.__dict__.update(d)
        return self
    
    def show(self, name='all'):
        from termcolor import colored
        if name == 'all':
            for attr, value in vars(self).items():
                print(colored('{} : '.format(attr), color='blue', attrs=['bold']), end='')
                print(colored(value, color='white', attrs=['bold']))
        elif hasattr(self, name):
            print(colored('{} : '.format(name), color='blue', attrs=['bold']), end='')
            print(colored(self.name, color='white', attrs=['bold']))

    def load(self):
        from make_regressors import load_all_regressors
        ds = load_all_regressors(self.loadpath)
        if self.verbose:
            print('loaded *index.nc files from {}'.format(self.loadpath))
        self = PredictorSet(ds, **vars(self))
        assert self.sample_dim in [x for x in self.dims]
        return self

    def normalize(self):
        from aux_functions_strat import normalize_xr
        ds = normalize_xr(self, norm=1, verbose=self.verbose)
        self = PredictorSet(ds, **vars(self))
        self.attrs = ds.attrs
        return self

    def deseason(self):
        from aux_functions_strat import deseason_xr
        # update this dict to include predictors to be deseasoned and how:
        deseason_dict = {'radio_cold': 'mean', 'radio_cold_no_qbo': 'mean',
                         'era5_bdc': 'mean'}
        if self.deseason_dict is not None:
            deseason_dict.update(self.deseason_dict)
        for pred, how in deseason_dict.items():
            if pred in self.data_vars:
                self[pred] = deseason_xr(self[pred], how=how,
                                         verbose=self.verbose)    
        return self

    def select_season(self, season):
        self_season = self['{}.season'.format(self.sample_dim)] == season
        to_update = self.__dict__
        self = self.sel({self.sample_dim: self_season})
        self.__dict__.update(to_update)
        self.season = season
        return self

    def select_times(self, times):
        to_update = self.__dict__
        self = self.sel({self.sample_dim: slice(*times)})
        self.__dict__.update(to_update)
        self.attrs['times'] = times
        self.time_period = times
        return self

    def select(self, pred_list, base_preds=True, stack=True):
        # first check for * symbol:
        import re
        import numpy as np
        pred_list = sort_predictors_list(pred_list)
        inter_splitted = [re.split("[*]+", x) for x in pred_list]
        max_degree = []
        max_inter = []
        all_base_preds = []
        for item in inter_splitted:
            max_inter.append(len(item))
            max_power = []
            for pr in item:
                all_base_preds.append(pr.split('^')[0])
                try:
                    max_power.append(int(pr.split('^')[-1]))
                except ValueError:
                    max_power.append(1)
            max_degree.append(max(max_power))
        degree = np.max(np.array(max_inter) * np.array(max_degree))
#        interactions = [x for x in pred_list if '*' in x]
#        power = [x for x in pred_list if '^' in x]
#        # select only those without power or interactions:
#        preds = list(
#            set(pred_list).difference(
#                set(power).union(
#                    set(interactions))))
#        if not preds:
        all_preds = list(set(all_base_preds))
        self_attrs = self.attrs
        to_update = self.__dict__
        self = self[all_preds]
        self.__dict__.update(to_update)
        to_update = self.__dict__
        # drop nans from sample_dim:
        self = self.dropna(self.sample_dim)
        self.__dict__.update(to_update)
        if base_preds:
            self.attrs = self_attrs
            return self
        print(pred_list)
        # transpose all dims to sample_dim first (time):
        da = self.to_array(self.feature_dim).transpose(self.sample_dim, ...)
        poly_stacked = poly_features(da, degree=degree)
        # now select just the predictors from pred_list, first sort them:
        features = poly_stacked[self.feature_dim].values
        sorted_features = sort_predictors_list(features)
        print('sorted_f:',sorted_features)
        poly_stacked[self.feature_dim] = sorted_features
        poly_selected = poly_stacked.sel({self.feature_dim: pred_list})
        poly_selected.name = 'X'
        if stack:
            return poly_selected
        else:
            ds = poly_selected.to_dataset(self.feature_dim)
            self = PredictorSet(ds, **vars(self))
            self.attrs = self_attrs
            return self

    def reg_shift(self, reg_time_shift):
        import xarray as xr
        ds_list = []
        for reg, shifts in reg_time_shift.items():
            reg_shift_ds = regressor_shift(self[reg],
                                           including_lag0=False,
                                           shifts=shifts)
            ds_list.append(reg_shift_ds)
        ds = xr.merge(ds_list)
        new_self = xr.merge([self, ds])
        self = PredictorSet(new_self.dropna(self.sample_dim), **vars(self))
        return self

    def plot(self):
        ax = self.to_dataframe().plot()
        ax.grid()
        return ax

    def level_month_shift(self, *args):
        return self

    def pre_process(self, stack=True):
        # flow of predictors pre-processing:
        # 1) load all predictors
        self = self.load()
        # 2) select base predictors (i.e., without poly)
        self = self.select(self.regressors, base_preds=True)
        # 3) select time period
        self = self.select_times(self.time_period)
        # 4) deseason
        self = self.deseason()
        # 5) normalize
        self = self.normalize()
        # 6) optional : reg_shift
        if self.reg_shift is not None:
            self = self.reg_shift(self.reg_shift)
        # 7) optional: season selection
        if self.season is not None:
            self = self.select_season(self.season)
        # 8) to_array - stacking
        if stack:
            X = self.select(self.regressors, base_preds=False, stack=True)
            return X
        else:
            self = self.select(self.regressors, base_preds=False, stack=False)
        # X = self.to_array(dim=self.feature_dim).T
        # 9) optional: poly features
        # outside of PredictorSet scope : align sample_dim (time) with da(TargetArray)
        return self


class ImprovedRegressor(RegressorWrapper):
    def __init__(self, estimator=None, reshapes=None, sample_dim=None,
                 **kwargs):
        # call parent constructor to set estimator, reshapes, sample_dim,
        # **kwargs
        super().__init__(estimator, reshapes, sample_dim, **kwargs)

    def fit(self, X, y=None, verbose=True, **fit_params):
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
        self.results_ = self.make_results(X, y, verbose)
        setattr(self, 'results_', self.results_)
        # set X_ and y_ attrs:
        setattr(self, 'X_', X)
        setattr(self, 'y_', y)
        return self

    def make_results(self, X, y, verbose=True):
        """ make results for all models type into xarray"""
        from aux_functions_strat import xr_order
        import xarray as xr
        from sklearn.metrics import r2_score
        from aux_functions_strat import text_blue
        feature_dim, mt_dim = get_feature_multitask_dim(X, y, self.sample_dim)
        rds = y.to_dataset(name='original').copy(deep=False, data=None)
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
        rds['corrcoef'] = self.corrcoef(X, y)
        # unstack dims:
        if mt_dim:
            rds = rds.unstack(mt_dim)
        # order:
        rds = xr_order(rds)
        # put coords attrs back:
        for coord, attr in y.attrs['coords_attrs'].items():
            rds[coord].attrs = attr
        # remove coords attrs from original, predict and resid:
        rds.original.attrs.pop('coords_attrs')
        rds.predict.attrs.pop('coords_attrs')
        rds.resid.attrs.pop('coords_attrs')
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
        # add X to results:
        rds['X'] = X
        if verbose:
            print(text_blue('Producing results...Done!'))
        return rds

    def save_results(self, path_like):
        ds = self.results_
        comp = dict(zlib=True, complevel=9)  # best compression
        encoding = {var: comp for var in ds.data_vars}
        ds.to_netcdf(path_like, 'w', encoding=encoding)
        print('saved results to {}.'.format(path_like))
        return

    def make_RI(self, X, y, Target, plevels=None, lms=None):
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
            self = run_model_with_shifted_plevels(self, new_X, y, Target, plevel=plevels, lms=lms)
            # self.fit(new_X, y)
            res_dict[keys] = self.results_
#            elif mode == 'model_all':
#                params, res_dict[keys] = run_model_for_all(new_X, y, params)
#            elif mode == 'multi_model':
#                params, res_dict[keys] = run_multi_model(new_X, y, params)
        self.results_ = produce_RI(res_dict, feature_dim)
        self.X_ = X
        return

    def corrcoef(self, X, y):
        import numpy as np
        import xarray as xr
        feature_dim, mt_dim = get_feature_multitask_dim(X, y, self.sample_dim)
        cor_mat = np.empty((X[feature_dim].size, y[mt_dim].size))
        for i in range(X[feature_dim].size):
            for j in range(y[mt_dim].size):
                cor_mat[i, j] = np.corrcoef(X.isel({feature_dim: i}),
                                            y.isel({mt_dim: j}))[0, 1]
        corr = xr.DataArray(cor_mat, dims=[feature_dim, mt_dim])
        corr[feature_dim] = X[feature_dim]
        corr[mt_dim] = y[mt_dim]
        corr.name = 'corrcoef'
        corr.attrs['description'] = 'np.corrcoef on each regressor and geo point'
        return corr

    def plot_like(self, field, flist=None, fmax=False, tol=0.0,
                  mean_lonlat=[True, False], title=None, **kwargs):
        # div=False, robust=False, vmax=None, vmin=None):
        """main plot for the results_ product of ImrovedRegressor
        flist: list of regressors to plot,
        fmax: wether to normalize color map on the plotted regressors,
        tol: used to control what regressors to show,
        mean_lonlat: wether to mean fields on the lat or lon dim"""
        from matplotlib.ticker import ScalarFormatter
        import matplotlib.pyplot as plt
        import aux_functions_strat as aux
        import pandas as pd
        # TODO: add area_mean support
        if not hasattr(self, 'results_'):
            raise AttributeError('No results yet... run model.fit(X,y) first!')
        rds = self.results_
        if field not in rds.data_vars:
            raise KeyError('No {} in results_!'.format(field))
        # if 'div' in keys:
        #     cmap = 'bwr'
        # else:
        #     cmap = 'viridis'
        plt_kwargs = {'yscale': 'log', 'yincrease': False, 'cmap': 'bwr'}
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
            plt_sample = {**plt_kwargs}
            plt_sample.update({'center': 0.0, 'levels': 41, 'vmax': cmap_max})
            plt_sample.update(kwargs)
            con = orig.T.plot.contourf(ax=axes[0], **plt_sample)
            cb = con.colorbar
            cb.set_label(orig.attrs['units'], fontsize=10)
            ax = axes[0]
            ax.set_title(orig.attrs['long_name'] + ' original', loc='center')
            ax.yaxis.set_major_formatter(ScalarFormatter())
            # plot the PREDICTED :
            times = times.reindex({'time': pd.date_range(times.time[0].values,
                                                         times.time[-1].values,
                                                         freq='MS')})
            plt_sample.update({'extend': 'both'})
            con = times.T.plot.contourf(ax=axes[1], **plt_sample)
            # robust=robust)
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
            [ax.invert_yaxis() for ax in con.ax.figure.axes]
            plt.show()
            return con
        elif field in rds.attrs['error_types']:
            # TODO: add contour lines
            if title is not None:
                suptitle = title
            else:
                suptitle = rds[field].name
            plt_error = {**plt_kwargs}
            plt_error.update({'cmap': 'viridis', 'add_colorbar': True,
                             'figsize': (6, 8)})
            plt_error.update(kwargs)
            if 'lon' in rds[field].dims:
                error_field = aux.xr_weighted_mean(rds[field],
                                                   mean_on_lon=mean_lonlat[0],
                                                   mean_on_lat=mean_lonlat[1])
            else:
                error_field = rds[field]
            try:
                con = error_field.plot.contourf(**plt_error)
                ax = plt.gca()
                ax.yaxis.set_major_formatter(ScalarFormatter())
                plt.suptitle(suptitle, fontsize=12, fontweight=750)
            except KeyError:
                print('Field not found or units not found...')
                return
            except ValueError:
                con = error_field.plot(xscale='log', xincrease=False,
                                      figsize=(6, 8))
                ax = plt.gca()
                ax.xaxis.set_major_formatter(ScalarFormatter())
                plt.suptitle(suptitle, fontsize=12, fontweight=750)
            plt.show()
            plt.gca().invert_yaxis()
            return con
        elif field in rds.attrs['feature_types']:
            # TODO: add contour lines
            con_levels = [0.001, 0.005, 0.01, 0.05]  # for pvals
            con_colors = ['blue', 'cyan', 'yellow', 'red']  # for pvals
            import xarray as xr
            fdim = rds.attrs['feature_dim']
            if flist is None:
                flist = [x for x in rds[fdim].values if
                         xr.ufuncs.fabs(rds[field].sel({fdim: x})).mean() > tol]
            if rds[fdim].sel({fdim: flist}).size > 6:
                colwrap = 6
            else:
                colwrap = None
            if 'lon' in rds[field].dims:
                feature_field = aux.xr_weighted_mean(rds[field],
                                                     mean_on_lon=mean_lonlat[0],
                                                     mean_on_lat=mean_lonlat[1])
            else:
                feature_field = rds[field]
            vmax = feature_field.max()
            if fmax:
                vmax = feature_field.sel({fdim: flist}).max()
            if title is not None:
                suptitle = title
            else:
                suptitle = feature_field.name
            plt_feature = {**plt_kwargs}
            plt_feature.update({'add_colorbar': False, 'levels': 41,
                                'figsize': (15, 4),
                                'extend': 'min', 'col_wrap': colwrap})
            plt_feature.update(**kwargs)
            try:
                if feature_field.name == 'pvalues':
                    plt_feature.update({'colors': con_colors,
                                        'levels': con_levels, 'extend': 'min'})
                    plt_feature.update(**kwargs)
                    plt_feature.pop('cmap', None)
                else:
                    plt_feature.update({'cmap': 'bwr',
                                        'vmax': vmax})
                    plt_feature.update(**kwargs)
                fg = feature_field.sel({fdim: flist}).plot.contourf(col=fdim,
                                                                    **plt_feature)
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
            except ValueError as valerror:
                print(valerror)
                fg = feature_field.plot(col=fdim, xscale='log', xincrease=False,
                                        figsize=(15, 4))
                fg.fig.subplots_adjust(bottom=0.3, top=0.85, left=0.05)
                ax = plt.gca()
                ax.xaxis.set_major_formatter(ScalarFormatter())
                plt.suptitle(suptitle, fontsize=12, fontweight=750)
            plt.show()
            [ax.invert_yaxis() for ax in fg.fig.axes]
            return fg
