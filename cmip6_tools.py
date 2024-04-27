import intake
import xarray as xr
import numpy as np
import pandas as pd
import os
import regionmask
states = regionmask.defined_regions.natural_earth_v5_0_0.us_states_50

import requests



def _table_id(var, freq):
    if var in ['ta', 'pr', 'psl', 'ua', 'va', 'zg', 'huss', 'hurs', 'rlut', 'rsut', 'rsdt', 'rsds', 'rlds', 'clt', 'clivi', 'clwvi', 'evspsbl', 'prsn', 'prw', 'ps', 'psl', 'sfcWind', 'tas', 'tasmax', 'tasmin', 'ts', 'uas', 'vas', 'wap', 'zg']:
        table_id_prefix = 'A'
    elif var in ['tos', 'zos', 'sos', 'thetao', 'so', 'uo', 'vo', 'zos']:
        table_id_prefix = 'O'

    if freq == 'day':
        if table_id_prefix == 'A':
            table_id = 'day'
        elif table_id_prefix == 'O':
            table_id = 'day'
    elif freq == 'mon':
        table_id=table_id_prefix+'mon'

    return table_id


def _standardize_coordinates(ds):
    # still building this to standardize models loaded from different sources
    try:
        ds=ds.rename({"longitude":"lon", "latitude":"lat"}) 
    except: pass

    if 'pr' in list(ds.data_vars):
        ds['pr'] = ds.pr * (24*60*60) # convert to mm/day
    if 'tas' in list(ds.data_vars):
        ds['tas'] = ds.tas - 273 #convert to degrees C

    # some datasets come out not in time order, so sort by time to make sure
    ds = ds.sortby('time')
    return ds   



def _fetch_cmip6_from_pangeo(model, experiment, member, var, table_id):
    # load the catalog
    col = intake.open_esm_datastore("https://storage.googleapis.com/cmip6/pangeo-cmip6.json")
    # search the catalog for the specified model, scenario, and variable
    cat = col.search(experiment_id=experiment, table_id=table_id, variable_id=var, source_id=model, member_id=member);
    # load the data
    dset_dict = cat.to_dataset_dict(zarr_kwargs={'consolidated': True}, cdf_kwargs={'chunks': {}}, progressbar=False);
    # pull the data out of the dictionary
    dset = dset_dict[list(dset_dict.keys())[0]]
    return dset


def _fetch_cmip6_from_esgf(model, experiment, member, var, table_id):
 # get the list of files
    files = esgf_search(variable_id=var, source_id=model, experiment_id=experiment, member_id=member,
                        table_id=table_id,
                        latest=True)
    
    data_df = _parse_files_into_df(files)

    # files are available from multiple nodes, so we need to try them one at a time
    for node in data_df.node.unique():
        # get the list of files for this node
        node_files = data_df[data_df.node==node].url.values
        # try to open each file
        for file in node_files:
            try:
                dset = xr.open_mfdataset(node_files, combine='by_coords',
                                             data_vars='minimal', coords='minimal',
                                             chunks = {'time': 500, 'lat': 45, 'lon': 45}
                                         )
                print('loaded ' + file)
                break
            except:
                print('failed to load ' + file)
                continue
        # if we successfully opened a file, stop trying
        if 'dset' in locals():
            break
    return dset


## get a list of all CMIP6 models in the pangeo catalog
def get_models(source='pangeo'):
    if source =='pangeo':
        # load the catalog
        col = intake.open_esm_datastore("https://storage.googleapis.com/cmip6/pangeo-cmip6.json")
        # get a list of all models
        models = col.df['source_id'].unique()
        # return the list

    #currently not working, such a broad search just hangs. this works ebtter with the pyesgf library
    # elif source=='esgf':
    #     res = esgf_search(variable_id='tas', latest=True)
    #     res_df = _parse_files_into_df(res)
    #     models = res_df.source_id.unique()
    return models


## for a given model, get a list of member ids that are available for a given scenario
def get_members(model, experiment,var,freq,
                source='pangeo'):
    ''' experiment should be 'historical', 'ssp370', etc.
        var should be 'tas', 'pr', etc.
        freq should be 'mon' or 'day'
        source should be 'pangeo' or 'esgf'
    '''

    table_id = _table_id(var, freq)
    if source=='pangeo':
        # load the catalog
        col = intake.open_esm_datastore("https://storage.googleapis.com/cmip6/pangeo-cmip6.json")
    
        # get a list of all members for the specified model and scenario
        #members = col.search(experiment_id=experiment, table_id=table_id, variable_id=var, grid_label='gn', source_id=model).df['member_id'].unique()
        members = col.search(experiment_id=experiment, table_id=table_id, variable_id=var, source_id=model).df['member_id'].unique()
        # return the list
    elif source=='esgf':
        res = esgf_search(variable_id=var, source_id=model, experiment_id=experiment, table_id=table_id, latest=True)
        res_df = _parse_files_into_df(res)
        members = res_df.member.unique()

    return members

def get_vars(model, experiment,freq,
                source='pangeo'):
    ''' experiment should be 'historical', 'ssp370', etc.
        var should be 'tas', 'pr', etc.
        freq should be 'mon' or 'day'
        source should be 'pangeo' or 'esgf'
    '''

    table_id = _table_id(var, freq)
    if source=='pangeo':
        # load the catalog
        col = intake.open_esm_datastore("https://storage.googleapis.com/cmip6/pangeo-cmip6.json")
    
        # get a list of all members for the specified model and scenario
        #members = col.search(experiment_id=experiment, table_id=table_id, variable_id=var, grid_label='gn', source_id=model).df['member_id'].unique()
        members = col.search(experiment_id=experiment, table_id=table_id, variable_id=var, source_id=model).df['member_id'].unique()
        # return the list
    elif source=='esgf':
        res = esgf_search(variable_id=var, source_id=model, experiment_id=experiment, table_id=table_id, latest=True)
        res_df = _parse_files_into_df(res)
        members = res_df.member.unique()

    return members

## function to load a single variable from a CMIP6 model into an xarray dataset
def get_cmip6(model, experiment, member, var, freq,
            start_date=None, end_date=None,
            area=None, timeseries=False,
            source = "pangeo"):
    '''
    model should be a string, e.g. 'CESM2'
    experiment should be 'historical', 'ssp370', etc.
    member should be a string, e.g. 'r1i1p1f1'
    var should be 'tas', 'pr', etc.
    freq should be 'mon' or 'day'
    start_date should be a string in the format 'YYYY-MM-DD' or just a year 'YYYY'
    end_date should be a string in the format 'YYYY-MM-DD' or just a year 'YYYY'
    area should be 'california' or a list of [lat_S,lat_N,lon_W,lon_E]
    timeseries should be True or False
    source should be 'pangeo' or 'esgf'
    '''
    
    table_id = _table_id(var, freq)
    
    if source=="pangeo":
        dset = _fetch_cmip6_from_pangeo(model, experiment, member, var, table_id)
    elif source=="esgf":
        dset = _fetch_cmip6_from_esgf(model, experiment, member, var, table_id)


    dset = _standardize_coordinates(dset)
    # subset the data to the specified time period
    if start_date:
        dset = dset.sel(time=slice(start_date, end_date))

    # if an area is specified, subset the data to that area
    # assuming the area is given as [lat_S,lat_N,lon_W,lon_E]
    if area is not None:
        if area=="CA" or area=="california":
            mask = states.mask(dset.lon, dset.lat, wrap_lon=True)
            dset=dset.where(mask==4,drop=True)
        elif area=="global" or area==None:
            dset=dset
        else:        
            # check if lon is in 0-360 or -180-180, and adjust the area bounds accordingly
            if dset.lon.min() <= 0:
                area = [area[0], area[1], area[2]+360, area[3]+360]
            dset = dset.sel(lat=slice(area[0], area[1]), lon=slice(area[2], area[3]))

    # spatially average into a timeseries
    if timeseries:

        dset_area = _area_grid(dset['lat'], dset['lon'])# total area
        total_area = dset_area.sum(['lat','lon'])# temperature weighted by grid-cell area
        dset[var] = (dset[var]*dset_area) / total_area# area-weighted mean temperature
        dset = dset.sum(['lat','lon'])


    return dset



def get_change(model,experiments, member, var, freq,
               area=None, hist_start_date=None, hist_end_date=None,
               fut_start_date=None, fut_end_date=None, source='pangeo'):
    """inputs a model, member, variable, and a list if future scenarios (e.g. ['ssp370','ssp585']).
    uses get_cmip6() to load the historical and future data, then calculates the change between the two.
    returns a list of changes for each of the input scenarios."""

    changes= []
    # load the historical data
    hist = get_cmip6(model, 'historical', member, var, freq, area=area, start_date=hist_start_date, end_date=hist_end_date,timeseries=True, source=source)
    # load the future data
    for expt in experiments:
        try:
            fut = get_cmip6(model, expt, member, var, freq, area=area, start_date=fut_start_date, end_date=fut_end_date, timeseries=True, source=source)
            change = np.float(fut.mean(dim='time')[var].values) - np.float(hist.mean(dim='time')[var].values)

        except:
            print('no data for ' + model + ' ' + expt + ' ' + member)
            change = np.nan
            continue
        
        # add the change to a list
        changes.append(change)
    # return the list of changes
    return changes



############################ ESGF SEARCH FUNCTION ############################
# Author: Unknown
# original version from a word document published by ESGF
# https://docs.google.com/document/d/1pxz1Kd3JHfFp8vR2JCVBfApbsHmbUQQstifhGNdc6U0/edit?usp=sharing

# API AT: https://github.com/ESGF/esgf.github.io/wiki/ESGF_Search_REST_API#results-pagination

def _parse_files_into_df(cat):
    '''takes in a search result from esgf_search and returns a dataframe with columns for
    node, var, table, model, experiment, member, grid, start date, end date'''
    cat_list = []
    for entry in cat:
        components = entry.split('/')
        node = components[2]
        components = components[-1].split('_')
        sdate = components[6].split('-')[0]
        edate = components[6].split('-')[1].split('.')[0]

        row = {'node':node, 'var':components[0], 'table':components[1], 'model':components[2], 'secnario':components[3], 'member':components[4], 
               'grid':components[5], 'start_date':sdate, 'end_date':edate, 'url':entry}
        cat_list.append(row)
    df = pd.DataFrame(cat_list, columns=['node', 'var', 'table', 'model', 'secnario', 'member', 'grid', 'start_date', 'end_date', 'url']) 
    return df


def summarize_esgf_data(model, experiment, member, var, freq):
    table_id = _table_id(var, freq)
    files = esgf_search(variable_id=var, source_id=model, experiment_id=experiment, member_id=member,
                        table_id=table_id,
                        latest=True)
    
    data_df = _parse_files_into_df(files)
    return data_df
                        


def esgf_search(server="https://esgf-node.llnl.gov/esg-search/search",
                files_type="OPENDAP", local_node=True, project="CMIP6",
                verbose=False, format="application%2Fsolr%2Bjson",
                use_csrf=False, **search):
    client = requests.session()
    payload = search
    payload["project"] = project
    payload["type"]= "File"
    if local_node:
        payload["distrib"] = "false"
    if use_csrf:
        client.get(server)
        if 'csrftoken' in client.cookies:
            # Django 1.6 and up
            csrftoken = client.cookies['csrftoken']
        else:
            # older versions
            csrftoken = client.cookies['csrf']
        payload["csrfmiddlewaretoken"] = csrftoken

    payload["format"] = format

    offset = 0
    numFound = 10000
    all_files = []
    files_type = files_type.upper()
    while offset < numFound:
        payload["offset"] = offset
        url_keys = []
        for k in payload:
            url_keys += ["{}={}".format(k, payload[k])]

        url = "{}/?{}".format(server, "&".join(url_keys))
        #print(url)
        r = client.get(url)
        r.raise_for_status()
        resp = r.json()["response"]
        numFound = int(resp["numFound"])
        resp = resp["docs"]
        offset += len(resp)
        for d in resp:
            if verbose:
                for k in d:
                    print("{}: {}".format(k,d[k]))
            url = d["url"]
            for f in d["url"]:
                sp = f.split("|")
                if sp[-1] == files_type:
                    all_files.append(sp[0].split(".html")[0])
    return sorted(all_files)




def _earth_radius(lat):
    '''
    calculate radius of Earth assuming oblate spheroid
    defined by WGS84
    
    Input
    ---------
    lat: vector or latitudes in degrees  
    
    Output
    ----------
    r: vector of radius in meters
    
    Notes
    -----------
    WGS84: https://earth-info.nga.mil/GandG/publications/tr8350.2/tr8350.2-a/Chapter%203.pdf
    '''
    from numpy import deg2rad, sin, cos

    # define oblate spheroid from WGS84
    a = 6378137
    b = 6356752.3142
    e2 = 1 - (b**2/a**2)
    
    # convert from geodecic to geocentric
    # see equation 3-110 in WGS84
    lat = deg2rad(lat)
    lat_gc = np.arctan( (1-e2)*np.tan(lat) )

    # radius equation
    # see equation 3-107 in WGS84
    r = (
        (a * (1 - e2)**0.5) 
         / (1 - (e2 * np.cos(lat_gc)**2))**0.5 
        )

    return r


def _area_grid(lat, lon):
    """
    Calculate the area of each grid cell
    Area is in square meters
    
    Input
    -----------
    lat: vector of latitude in degrees
    lon: vector of longitude in degrees
    
    Output
    -----------
    area: grid-cell area in square-meters with dimensions, [lat,lon]
    
    Notes
    -----------
    Based on the function in
    https://github.com/chadagreene/CDT/blob/master/cdt/cdtarea.m
    """
    from numpy import meshgrid, deg2rad, gradient, cos
    from xarray import DataArray

    xlon, ylat = meshgrid(lon, lat)
    R = _earth_radius(ylat)

    dlat = deg2rad(gradient(ylat, axis=0))
    dlon = deg2rad(gradient(xlon, axis=1))

    dy = dlat * R
    dx = dlon * R * cos(deg2rad(ylat))

    area = dy * dx

    xda = DataArray(
        area,
        dims=["lat", "lon"],
        coords={"lat": lat, "lon": lon},
        attrs={
            "long_name": "area_per_pixel",
            "description": "area per pixel",
            "units": "m^2",
        },
    )
    return xda
