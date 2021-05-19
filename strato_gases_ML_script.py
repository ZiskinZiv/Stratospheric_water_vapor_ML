#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 12 10:11:01 2021

@author: shlomi
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 08:31:03 2020

@author: shlomi
"""
import os
import sys
import warnings
from strat_paths import work_chaim
ml_path = work_chaim / 'ML'

if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = (
        'ignore::UserWarning,ignore::RuntimeWarning')  # Also affect subprocesses


def check_path(path):
    import os
    from pathlib import Path
    path = str(path)
    if not os.path.exists(path):
        raise argparse.ArgumentTypeError(path + ' does not exist...')
    return Path(path)


def check_regressors(regressors):
    from make_regressors import load_all_regressors
    ds = load_all_regressors()
    regs = [x for x in ds]
    for reg in regressors:
        assert reg in regs
    return regressors


def main_strato_gases_ML(args):
    from strato_gases_simple_ML import produce_X
    from strato_gases_simple_ML import produce_y
    from strato_gases_simple_ML import single_cross_validation
    if args.n_splits is not None:
        n_splits = args.n_splits
    else:
        n_splits = 5
    if args.rseed is None:
        seed = 42
    else:
        seed = args.rseed
    if args.param_grid is None:
        param_grid = 'dense'
    else:
        param_grid = args.param_grid
    if args.verbose is None:
        verbose=0
    else:
        verbose = args.verbose
    if args.n_jobs is None:
        n_jobs = -1
    else:
        n_jobs = args.n_jobs
    if args.regressors is not None:
        regressors = args.regressors
    else:
        regressors = ['qbo_cdas', 'anom_nino3p4']
    X = produce_X(regressors=regressors, lag={'qbo_cdas': 5})
    y = produce_y(path=work_chaim, detrend='lowess',
                  sw_var='combinedeqfillanomfillh2oq', filename='swoosh_latpress-2.5deg.nc',
                  lat_mean=[-30, 30], plevel=82, deseason='std')
    # scorers = ['roc_auc', 'f1', 'recall', 'precision']
    X = X.sel(time=slice('1994', '2019'))
    y = y.sel(time=slice('1994', '2019'))
    if args.scorers is None:
        scorers = ['r2', 'r2_adj', 'neg_mean_squared_error',
                   'explained_variance']
    else:
        scorers = [x for x in args.scorers]
    model_name = args.model

    if args.savepath is not None:
        savepath = args.savepath
    else:
        savepath = ml_path

    logger.info(
            'Running {} model with {} nsplits, regressors={}'.format(
                model_name, n_splits, regressors))
    model = single_cross_validation(
                X,
                y, scorers=scorers,
                model_name=model_name,
                n_splits=n_splits,
                verbose=verbose,
                param_grid=param_grid, seed=seed,
                savepath=savepath, n_jobs=n_jobs)
    print('')
    logger.info('Done!')


if __name__ == '__main__':
    import argparse
    import sys
    from pathlib import Path
    from aux_functions_strat import configure_logger
    from strat_paths import work_chaim
    ml_path = work_chaim / 'ML'
    logger = configure_logger('Strato_gases_ML')
    savepath = Path(ml_path)
    parser = argparse.ArgumentParser(
        description='a command line tool for running the ML models tuning for Starto-Gases.')
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')

    required.add_argument(
        '--savepath',
        help="a full path to download the files, e.g., /home/ziskin/Work_Files/PW_yuval/IMS_T/10mins",
        type=check_path)
    optional.add_argument(
        '--n_splits',
        help='how many splits for the CV',
        type=int)

    optional.add_argument(
        '--param_grid',
        help='param grids for gridsearchcv object',
        type=str, choices=['light', 'normal', 'dense'])

    optional.add_argument(
        '--n_jobs',
        help='number of CPU threads to do gridsearch and cross-validate',
        type=int)
    optional.add_argument(
        '--rseed',
        help='random seed interger to start psuedo-random number generator',
        type=int)
    optional.add_argument(
        '--verbose',
        help='verbosity 0, 1, 2',
        type=int)
    optional.add_argument(
        '--scorers',
        nargs='+',
        help='scorers, e.g., r2, r2_adj, etc',
        type=str)
#    optional.add_argument('--nsplits', help='select number of splits for HP tuning.', type=int)
    required.add_argument(
        '--model',
        help='select ML model.',
        choices=[
            'SVM',
            'MLP',
            'RF'])
    optional.add_argument('--regressors', help='select features for ML', type=check_regressors, nargs='+')
    parser._action_groups.append(optional)  # added this line
    args = parser.parse_args()

    if args.savepath is None:
        print('savepath is a required argument, run with -h...')
        sys.exit()

    if args.model is None:
        print('model is a required argument, run with -h...')
        sys.exit()
    logger.info('Running ML, CV with {} model'.format(args.model))
    main_strato_gases_ML(args)
