import os
from collections.abc import Iterable
from zoneinfo import ZoneInfo
import logging
import numpy as np
import pandas as pd
import xarray as xr
import datetime as dt
import importlib.resources
from toolz import keyfilter
import jstyleson as json
from addict import Dict as adict
from operator import itemgetter
from toolz import valfilter

import trosat.sunpos as sp

logger = logging.getLogger(__name__)

EPOCH_JD_2000_0 = np.datetime64("2000-01-01T12:00")
def to_datetime64(time, epoch=EPOCH_JD_2000_0):
    """
    Convert various representations of time to datetime64.

    Parameters
    ----------
    time : list, ndarray, or scalar of type float, datetime or datetime64
        A representation of time. If float, interpreted as Julian date.
    epoch : np.datetime64, default JD2000.0
        The epoch to use for the calculation

    Returns
    -------
    datetime64 or ndarray of datetime64
    """
    jd = sp.to_julday(time, epoch=epoch)
    jdms = np.int64(86_400_000*jd)
    return (epoch + jdms.astype('timedelta64[ms]')).astype("datetime64[ns]")

def tz_offset(zone: str, tz_reference: dt.datetime | None = None) -> int:
    if tz_reference is None:
        tz_reference = dt.datetime.now(dt.timezone.utc)
    elif tz_reference.tzinfo is None:
        tz_reference = tz_reference.replace(tzinfo=dt.timezone.utc)
    
    tz_target = ZoneInfo(zone)
    offset = tz_reference.astimezone(tz_target).utcoffset()
    return int(offset.total_seconds())

def offset_hhmm(seconds: int) -> str:
    sign = "+" if seconds >= 0 else "-"
    secs = abs(seconds)
    hh, mm = divmod(secs // 60, 60)
    return f"{sign}{hh:02d}:{mm:02d}"

def dt64_add_tz_offset(x, zone: str):
    offset = np.timedelta64(tz_offset(zone),'s')
    if isinstance(x, Iterable):
        if len(x) == 0:
            return np.array([]).astype("datetime64[us]")
        # handle if x is nested list
        if isinstance(x[0], Iterable):
            dt64 = []
            for xi in x:
                dt64.append([ np.datetime64(pd.to_datetime(t).tz_localize(None),"us") for t in xi])
            dt64 = np.array(dt64)
        else:
            dt64 = np.array([ np.datetime64(pd.to_datetime(t).tz_localize(None),"us") for t in x])
    else:
        dt64 = np.datetime64(pd.to_datetime(x).tz_localize(None),"us")

    return dt64 + offset

def dt64_sub_tz_offset(x, zone: str):
    offset = np.timedelta64(tz_offset(zone),'s')
    if isinstance(x, Iterable):
        if len(x) == 0:
            return np.array([]).astype("datetime64[us]")
        # handle if x is nested list
        if isinstance(x[0], Iterable):
            dt64 = []
            for xi in x:
                dt64.append([ np.datetime64(pd.to_datetime(t).tz_localize(None),"us") for t in xi])
            dt64 = np.array(dt64)
        else:
            dt64 = np.array([ np.datetime64(pd.to_datetime(t).tz_localize(None),"us") for t in x])
    else:
        dt64 = np.datetime64(pd.to_datetime(x).tz_localize(None),"us")

    return dt64 - offset

def round_to(base, x):
    """ Round x to a given base
    """
    return base * np.round(x/base, 0)

def read_json(fpath: str, *, object_hook: type = adict, cls = None) -> dict:
    """ Parse json file to python dict.
    """
    with open(fpath,"r") as f:
        js = json.load(f, object_hook=object_hook, cls=cls)
    return js

def pick(whitelist: list[str], d: dict) -> dict:
    """ Keep only whitelisted keys from input dict.
    """
    return keyfilter(lambda k: k in whitelist, d)

def omit(blacklist: list[str], d: dict) -> dict:
    """ Omit blacklisted keys from input dict.
    """
    return keyfilter(lambda k: k not in blacklist, d)

def get_var_attrs(d: dict) -> dict:
    """
    Parse cf-compliance dictionary.

    Parameters
    ----------
    d: dict
        Dict parsed from cf-meta json.

    Returns
    -------
    dict
        Dict with netcdf attributes for each variable.
    """
    get_vars = itemgetter("variables")
    get_attrs = itemgetter("attributes")
    vattrs = {k: get_attrs(v) for k,v in get_vars(d).items()}
    for k,v in get_vars(d).items():
        vattrs[k].update({
            "dtype": v["type"],
            "gzip":True,
            "complevel":6
        })
    return vattrs

def get_attrs_enc(d : dict) -> (dict,dict):
    """ Split variable attributes in attributes and encoding-attributes.
    """
    _enc_attrs = {
        "scale_factor",
        "add_offset",
        "_FillValue",
        "dtype",
        "zlib",
        "gzip",
        "complevel",
        "calendar",
    }
    # extract variable attributes
    vattrs = {k: omit(_enc_attrs, v) for k, v in d.items()}
    # extract variable encoding
    vencode = {k: pick(_enc_attrs, v) for k, v in d.items()}
    return vattrs, vencode

def get_default_config():
    """
    Get TARO default config
    """
    fn_config = os.path.join(
        importlib.resources.files("taro"),
        "conf/taro_config.json"
    )
    default_config = read_json(fn_config)

    # expand default file paths
    for key in default_config:
        if key.startswith("file"):
            default_config.update({
                key: os.path.join(
                    importlib.resources.files("taro"),
                    default_config[key]
                )
            })
    return default_config

def merge_config(config):
    """
    Merge config dictionary with taro default config
    """
    default_config = get_default_config()
    if config is None:
        config = default_config
    else:
        config = {**default_config, **config}
    return config


def init_logger(config):
    """
    Initialize Logging based on taro config
    """
    config = merge_config(config)
    fname = os.path.abspath(config["file_log"])

    # logging setup
    logging.basicConfig(
        filename=fname,
        encoding='utf-8',
        level=logging.DEBUG,
        format='%(asctime)s %(name)s %(levelname)s:%(message)s'
    )

def parse_calibration(cfile,troposID, cdate=None):
    """
    Parse calibration json file

    Parameters
    ----------
    cfile: str
        Path of the calibration.json
    cdate: list, ndarray, or scalar of type float, datetime or datetime64
        A representation of time. If float, interpreted as Julian date.
    Returns
    -------
    dict
        Calibration dictionary sorted by box number.
    """
    if cdate is not None:
        cdate = to_datetime64(cdate)
    calib = read_json(cfile)
    # parse calibration dates
    date_keys = [ key for key in calib if key.startswith("2") ]
    cdates = pd.to_datetime(date_keys, yearfirst=True).values
    isort = np.argsort(cdates)
    skeys = np.array(date_keys)[isort]

    ctimes, cfacs, cerrs = [], [], []
    # lookup calibration factors
    for i, key in enumerate(skeys):
        if troposID in calib[key]:
            ctimes.append(cdates[i])
            cfacs.append(calib[key][troposID][0])
            if calib[key][troposID][1] is None:
                cerrs.append(np.nan)
            else:
                cerrs.append(calib[key][troposID][1])
    if len(ctimes) == 0:
        return None

    ds = xr.Dataset(
        {
            "calibration_factor": ("time", np.array(cfacs)),
            "calibration_error": ("time", np.array(cerrs)),
            "calibration_factor_units": calib['units'][0],
            "calibration_error_units": calib['units'][1]
        },
        coords={
            "time": ("time",np.array(ctimes))
        }
    )

    # add temperature correction coefficients
    if ("temp_correction" in calib) and (troposID in calib["temp_correction"]):
        ds["temperature_correction_coef"] = calib["temp_correction"][troposID]

    if cdate is not None:
        ds = ds.sel(time=cdate, method='nearest')

    return ds


def meta_lookup(config, *, serial=None, troposID=None, date=None):
    assert (serial is not None) or (troposID is not None)
    config = merge_config(config)
    mapping = read_json(config["file_instrument_map"])

    outdict = {
        "device": None,
        "serial": serial,
        "troposID": troposID,
        "calibration_factor": None,
        "calibration_error": None,
        "calibration_date": None,
        "calibration_factor_units": None,
        "calibration_error_units": None,
    }

    if troposID is None:
        mappingres = valfilter(lambda x: serial == x["serial"], mapping)
        troposID = list(mappingres.keys())[0]
        outdict.update({"troposID":troposID})

    outdict.update(mapping[troposID])

    calibration = parse_calibration(config["file_calibration"],
                                    troposID=troposID,
                                    cdate=date)
    if calibration is not None:
        outdict.update({
            "calibration_factor": calibration.calibration_factor.values.tolist(),
            "calibration_error": calibration.calibration_error.values.tolist(),
            "calibration_date": [f"{date:%Y-%m-%d}" for date in pd.to_datetime(calibration.time.values)],
            "calibration_factor_units": calibration.calibration_factor_units.values,
            "calibration_error_units": calibration.calibration_error_units.values
        })
        if "temperature_correction_coef" in calibration:
            outdict.update({
                "temperature_correction_coef": calibration.temperature_correction_coef.values
            })

    return outdict
