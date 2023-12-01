import os
import re
import xarray as xr
import pandas as pd
import numpy as np
import parse
import logging
from unitpy import Unit
from toolz import keyfilter, valfilter

import mordor
import mordor.utils
import mordor.futils

logger = logging.getLogger(__name__)

def _parse_unit(unit):
    # add ^ before numbers
    unit = re.sub("-?[0-9]","^\g<0>", unit)
    return unit

def to_l1a(
        fname: str,
        *,
        config=None,
        global_attrs=None):
    """
    Read logger raw file and parse it to xarray Dataset. Thereby, attributes and names are defined via cfmeta.json file and sun position values are calculated and added.

    Parameters
    ----------
    fname: str
        Path and filename of the raw logger file.
    config: dict
        Stores processing specific configuration.
            * cfjson -> path to cfmeta.json, the default is "../share/pyrnet_cfmeta.json"
            * stripminutes -> number of minutes to be stripped from the data at start and end,
                the default is 5.
    global_attrs: dict
        Additional global attributes for the Dataset. (Overrides cfmeta.json attributes)
    Returns
    -------
    xarray.Dataset
        Raw Logger data for one measurement periode.
    """
    config = mordor.utils.merge_config(config)
    gattrs, vattrs, vencode = mordor.futils.get_cfmeta(config)

    if global_attrs is not None:
        gattrs.update(global_attrs)

    # 1. Parse raw file
    # Parse table info
    fname_info = parse.parse(
        config["fname_out"].replace("%Y-%m-%d", "ti"),
        os.path.basename(fname)
    )
    table_config = mordor.utils.read_json(config["file_logger_tables"])[fname_info["table"]]
    usecols = np.argwhere([v is not None for v in table_config]).ravel()

    table_map = np.array([table_config[i] for i in usecols])

    names = []
    for i, v in enumerate(table_map[:, 0]):
        count = list(table_map[:i, 0]).count(v)
        names.append(v + f"_{count+1:d}" if count > 0 else v)

    # parse the data file
    dat = pd.read_csv(
        fname,
        skiprows=4,
        usecols=usecols,
        names=names,
        header=None,
        sep=','
    )

    dat["time"] = pd.to_datetime(dat['time'].astype(str), format='mixed')
    ds = dat.set_index("time").to_xarray()
    ds = ds.drop_vars(["record"])

    # add meta
    for name, troposid in zip(names,table_map[:,1]):
        if (name == "record") or (name == "time"):
            continue
        meta = mordor.utils.meta_lookup(config,troposID=troposid)
        # drop calibration meta
        meta = keyfilter(lambda x: not x.startswith('calibration'), meta)
        # drop None values
        meta = valfilter(lambda x: x is not None, meta)
        # update netcdf attributes
        ds[name].attrs.update(meta)

    ## Scale variables to correct units
    # identify logger units
    colunits = pd.read_csv(
        fname,
        skiprows=2,
        nrows=1,
        usecols=usecols,
        names=names,
        header=None,
        sep=','
    )
    colunits = colunits.set_index("time").to_xarray()
    colunits = colunits.drop_vars(["record"])

    # convert to cfmeta units
    for i,key in enumerate(names):
        if (key == "record") or (key == "time"):
            continue
        oldunit = colunits[key].values[0]
        newunit = vattrs[table_map[i, 0]]["units"]
        if oldunit == "%" and newunit == "1":
            ds[key].values = ds[key].values*1e-2
            continue

        oldunit = _parse_unit(oldunit)
        newunit = _parse_unit(newunit)

        oldval = ds[key].values * Unit(oldunit)
        ds[key].values = (oldval.to(newunit)).value

    # add coordinats if available in config
    if config["coordinates"] is not None:
        lat, lon, alt = config["coordinates"]
        if lat is not None:
            ds["lat"] = lat
        if lon is not None:
            ds["lon"] = lon
        if alt is not None:
            ds["altitude"] = alt


    # 2. Add global meta data
    now = pd.to_datetime(np.datetime64("now"))
    gattrs.update({
        'processing_level': 'l1a',
        'product_version': mordor.__version__,
        'history': f'{now.isoformat()}: Generated level l1a  by mordor version {mordor.__version__}; ',
    })
    ds.attrs.update(gattrs)

    # drop occurrence of duplicate sample values
    ds = ds.drop_duplicates("time")

    # add global coverage attributes
    ds = mordor.futils.update_coverage_meta(ds, timevar="time")

    # add attributes to Dataset
    for k,v in vattrs.items():
        if k not in ds.keys():
            continue
        # iterate over suffixed variables
        for ki in [key for key in ds if key.startswith(k)]:
            ds[ki].attrs.update(v)

    # add encoding to Dataset
    ds = mordor.futils.add_encoding(ds, vencode)

    return ds
