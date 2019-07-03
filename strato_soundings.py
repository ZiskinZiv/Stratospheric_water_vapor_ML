#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul  3 13:29:51 2019

@author: shlomi
"""

from strat_startup import *
import pandas as pd
import geopandas as gpd
import numpy as np
sound_path = work_chaim / 'sounding'
igra2 = pd.read_fwf(cwd / 'igra2-station-list.txt', header=None)
igra2.columns = ['station_number', 'lat', 'lon', 'alt', 'name', 'start_year',
                 'end_year', 'number']
igra2 = igra2[igra2['lat'] > -90]
world = gpd.read_file(cwd / 'gis/Countries_WGS84.shp')
geo_igra2 = gpd.GeoDataFrame(igra2, geometry=gpd.points_from_xy(igra2.lon,
                                                                igra2.lat),
                               crs=world.crs)
geo_igra2_eq = geo_igra2[np.abs(geo_igra2['lat'])<10]
geo_igra2_eq = geo_igra2_eq[geo_igra2_eq['end_year'] >= 2017]
ax = world.plot()
geo_igra2_eq.plot(ax=ax, column='alt', cmap='Reds', edgecolor='black',
                     legend=True)


def process_sounding_json(savepath=sound_path, igra_id='BRM00082332'):
    """process json files from sounding download and parse them to xarray"""
    import pandas as pd
    import json
    import xarray as xr
    import os
    # loop over lines lists in each year:
    pw_years = []
    df_years = []
    bad_line = []
    for file in sorted(savepath.glob(igra_id + '*.json')):
        year = file.as_posix().split('.')[0].split('_')[-1]
        print('Opening station {} json file year: {}'.format(igra_id, year))
        with open(file) as read_file:
            lines_list = json.load(read_file)
        # loop over the lines list:
        # pw_list = []
        dt_list = []
        df_list = []
        for lines in lines_list:
            # print('.')
            try:
                # pw = float([x for x in lines if '[mm]' in x][0].split(':')[-1])
                dt = [x for x in lines if 'Observation time' in
                      x][0].split(':')[-1].split()[0]
                # The %y (as opposed to %Y) is to read 2-digit year
                # (%Y=4-digit)
                header_line = [
                    x for x in range(
                        len(lines)) if 'Observations at'
                    in lines[x]][0] + 3
                end_line = [x for x in range(len(lines)) if
                            'Station information and sounding indices'
                            in lines[x]][0]
                header = lines[header_line].split()
                units = lines[header_line + 1].split()
                with open(savepath/'temp.txt', 'w') as f:
                    for item in lines[header_line + 3: end_line]:
                        f.write("%s\n" % item)
                df = pd.read_fwf(savepath / 'temp.txt', names=header)
                try:
                    os.remove(savepath / 'temp.txt')
                except OSError as e:  # if failed, report it back to the user
                    print("Error: %s - %s." % (e.filename, e.strerror))
#                df = pd.DataFrame(
#                    [x.split() for x in lines[header_line + 3:end_line]],
#                    columns=header)
                df = df.astype(float)
                dt_list.append(pd.to_datetime(dt, format='%y%m%d/%H%M'))
                # pw_list.append(pw)
                df_list.append(df)
                st_num = int([x for x in lines if 'Station number' in
                              x][0].split(':')[-1])
                st_lat = float([x for x in lines if 'Station latitude' in
                                x][0].split(':')[-1])
                st_lon = float([x for x in lines if 'Station longitude' in
                                x][0].split(':')[-1])
                st_alt = float([x for x in lines if 'Station elevation' in
                                x][0].split(':')[-1])
            except IndexError:
                print('no data found in lines entry...')
                bad_line.append(lines)
                continue
            except AssertionError:
                bad_line.append(lines)
                continue
        # pw_year = xr.DataArray(pw_list, dims=['time'])
        df_year = [xr.DataArray(x, dims=['mpoint', 'var']) for x in df_list]
        df_year = xr.concat(df_year, 'time')
        df_year['time'] = dt_list
        df_year['var'] = header
        # pw_year['time'] = dt_list
        # pw_years.append(pw_year)
        df_years.append(df_year)
    # pw = xr.concat(pw_years, 'time')
    da = xr.concat(df_years, 'time')
    da.attrs['description'] = 'upper air soundings full profile'
    units_dict = dict(zip(header, units))
    for k, v in units_dict.items():
        da.attrs[k] = v
#    pw.attrs['description'] = 'BET_DAGAN soundings of precipatable water'
#    pw.attrs['units'] = 'mm'  # eqv. kg/m^2
    da.attrs['station_number'] = st_num
    da.attrs['station_lat'] = st_lat
    da.attrs['station_lon'] = st_lon
    da.attrs['station_alt'] = st_alt
#    pw = pw.sortby('time')
    da = da.sortby('time')
    # drop 0 pw - not physical
    # pw = pw.where(pw > 0, drop=True)
    # pw.to_netcdf(savepath / 'PW_bet_dagan_soundings.nc', 'w')
    da.to_netcdf(savepath / igra_id + '_soundings.nc', 'w')
    return da, bad_line