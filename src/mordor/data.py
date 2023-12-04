import os
import re
import xarray as xr
import pandas as pd
import numpy as np
import parse
import logging
from unitpy import Unit
from toolz import keyfilter, valfilter, assoc_in
import trosat.sunpos as sp

import mordor
import mordor.utils
import mordor.futils

logger = logging.getLogger(__name__)

def _parse_unit(unit):
    unit = str(unit)
    if unit == "%":
        return ''
    # add ^ before numbers
    unit = re.sub("-?[0-9]","^\g<0>", unit)
    return unit

def _parse_quantity(unit):
    punit = _parse_unit(unit)
    if unit == "%":
        return 1e-2*Unit(punit)
    return 1.*Unit(punit)

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


def to_l1b(ds_l1a, resolution, *, config=None):
    config = mordor.utils.merge_config(config)
    gattrs, vattrs, vencode = mordor.futils.get_cfmeta(config)

    if ds_l1a.processing_level != "l1a":
        logger.warning(f"Is not a l1a file. Skip.")
        return None

    # 1. resample radiation data
    flx_vars = config["l1b_flux_variables"]
    methods = ['mean'] + config["l1b_resample_stats"]
    res = mordor.futils.resample(
        ds_l1a,
        freq=resolution,
        methods=methods,
        kwargs=dict(skipna=True)
    )
    ds_l1b = res[0]
    for i, method in enumerate(methods[1:]):
        for var in flx_vars:
            ds_l1b[f"{var}_{method}"] = res[i+1][var]
            ds_l1b[f"{var}_{method}"].attrs.update({
                "standard_name": f"{method}_"+ds_l1b[f"{var}_{method}"].attrs["standard_name"]
            })

    # 3. calibrate flux vars
    flx_vars_all = [key for key in ds_l1b if key.split('_')[0] in flx_vars]
    for var in flx_vars_all:
        troposID = ds_l1b[var].attrs["troposID"]
        calib = mordor.utils.parse_calibration(
            cfile=config["file_calibration"],
            troposID=troposID,
            cdate=ds_l1b.time.values[0]
        )


        cfac_unit = _parse_quantity(calib["calibration_factor_units"].values)
        cfac = calib["calibration_factor"].values * cfac_unit
        cdate = f"{pd.to_datetime(calib.time.values):%Y-%m}"
        # assume cfac unit in the form ( units_of_Voltage / (Wm-2) )
        cfac = ((1./cfac).to(f"W m^-2 {ds_l1b[var].attrs['units']}^-1")).value

        # Longwave calibration
        if "longwave" in ds_l1b[var].attrs["standard_name"]:
            # select sensor temperature
            dstemp = ds_l1b.filter_by_attrs(troposID=troposID)
            for key in dstemp:
                if key.startswith("sensor_temperature"):
                    temp_sensor = dstemp[key].values * Unit(dstemp[key].attrs['units'])
            # measured Voltage
            V0 = ds_l1b[var].values
            # temperature correction
            if "temperature_correction_coef" in calib:
                a, b, c = calib["temperature_correction_coef"].values
                T = (temp_sensor.to("degC")).value
                V0 *= (1. + a*(T**2) + b*T + c)
            # calibrate to W m-2
            ds_l1b[var].values = V0 * cfac + 5.670367e-8 * ((temp_sensor.to("K")).value**4)
            ds_l1b[var].attrs.update({
                "calibration_function": "flux (W m-2) = flux (V) * calibration_factor (W m-2 V-1) + 5.670367e-8 * (sensor_temperature (K))**4",
            })
        else:
            ds_l1b[var].values = ds_l1b[var].values*cfac
            ds_l1b[var].values[ds_l1b[var].values < 0] = 0.
            ds_l1b[var].attrs.update({
                "calibration_function": "flux (W m-2) = flux (V) * calibration_factor (W m-2 V-1)",
            })
        print(var,ds_l1b[var].values[0],ds_l1b[var].values[0]/cfac,cfac)
        # add new attributes and encoding to calibrated flux vars
        ds_l1b[var].attrs.update({
            "units": "W m-2",
            "calibration_factor": calib["calibration_factor"].values,
            "calibration_factor_units": calib["calibration_factor_units"].values,
            "calibration_error": calib["calibration_error"].values,
            "calibration_error_units": calib["calibration_error_units"].values,
            "calibration_date": cdate,
        })
        scale_factor = 1e-4
        add_offset = 0.
        valid_range = np.array([0, 2000])  # valid range in data units W m-2
        valid_range = ((valid_range - add_offset)/scale_factor).astype(int)
        vencode = assoc_in(vencode, [var,'valid_range'],list(valid_range))
        vencode = assoc_in(vencode, [var, 'scale_factor'], scale_factor)
        vencode = assoc_in(vencode, [var, 'add_offset'], add_offset)

    if ("lat" in ds_l1b) and ("lon" in ds_l1b):
        # 4. Calc and add sun position
        szen, sazi = sp.sun_angles(
            time=ds_l1b.time.values,
            lat=ds_l1b.lat.values,
            lon=ds_l1b.lon.values
        )
        szen = szen.squeeze()
        sazi = sazi.squeeze()

        esd = np.mean(sp.earth_sun_distance(ds_l1b.time.values))

        ds_l1b = ds_l1b.assign(
            {
                "szen": (("time"), szen),
                "sazi": (("time"), sazi),
                "esd":  esd
            }
        )
        # update attributes and encoding
        for key in ['szen', 'sazi', 'esd']:
            ds_l1b[key].attrs.update(vattrs[key])

    # add coordinats if available in config
    if config["coordinates"] is not None:
        lat, lon, alt = config["coordinates"]
        if lat is not None:
            ds_l1b["lat"] = lat
        if lon is not None:
            ds_l1b["lon"] = lon
        if alt is not None:
            ds_l1b["altitude"] = alt

    # add global coverage attributes
    ds_l1b = mordor.futils.update_coverage_meta(ds_l1b, timevar="time")
    ds_l1b.attrs["processing_level"] = 'l1b'
    now = pd.to_datetime(np.datetime64("now"))
    ds_l1b.attrs["history"] = ds_l1b.history + f"{now.isoformat()}: Generated level l1b  by mordor version {mordor.__version__}; "
    ds_l1b.attrs['product_version'] = mordor.__version__

    # update encoding
    ds_l1b = mordor.futils.add_encoding(ds_l1b, vencode=vencode)

    return ds_l1b