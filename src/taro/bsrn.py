from collections.abc import Iterable
import os
import re
import parse
import glob
from toolz import assoc_in,merge,merge_with,valmap,valfilter
import xarray as xr
import numpy as np
import pandas as pd
import datetime as dt
import importlib.resources
import matplotlib.pyplot as plt
import inspect
import gzip

import taro.utils
import taro.futils


def class2dict(cls):
    cdict = {}
    # Loop over members of the outer class; we only care about inner classes
    for _, inner in inspect.getmembers(cls(), predicate=inspect.isclass):
        # Skip built‑in attributes like __module__, __doc__, etc.
        for name, values in inspect.getmembers(inner):
            if not name.startswith("__") and not inspect.isclass(values):
                cdict.update({name:values})
    return cdict

class QUANTITIES:
    class RADIATION:
        surface_downwelling_shortwave_flux_in_air = 2
        surface_direct_along_beam_shortwave_flux_in_air = 3
        surface_diffuse_downwelling_shortwave_flux_in_air = 4
        surface_downwelling_longwave_flux_in_air = 5
    class OTHER:
        air_temperature = 21
        relative_humidity = 22
        air_pressure = 23

class QUANTITIES_RAW:
    surface_downwelling_shortwave_flux_in_air = "swd03",
    surface_direct_along_beam_shortwave_flux_in_air = "dirn"
    surface_diffuse_downwelling_shortwave_flux_in_air = "dif"
    surface_downwelling_longwave_flux_in_air = "lwd03"
    air_temperature = "temp2m"
    air_pressure = "press"
    relative_humidity = "rh"


class TOPOGRAPHY:
    """TABLE A5"""
    flat_urban=1
    flat_rural=2
    hilly_urban=3
    hilly_rural=4
    mountain_top_urban=5
    mountain_top_rural=6
    mountain_valley_urban=7
    mountain_valley_rural=8

class SURFACE:
    glacier_accumulation = 1
    glacier_ablation = 2
    iceshelf = 3
    sea_ice = 4
    water_river = 5
    water_lake = 6
    water_ocean = 7
    desert_rock = 8
    desert_sand = 9
    desert_gravel = 10
    concrete = 11
    asphalt = 12
    cultivated = 13
    tundra = 14
    grass = 15
    shrub = 16
    forest_evergreen = 17
    forest_deciduous = 18
    forest_mixed = 19
    rock = 20
    sand = 21

class DTC:
    shaded = 1
    ventilated = 2
    temperature = 3 # temperature measurement
    shaded_ventilated = 4
    shaded_temperature = 5
    ventilated_temperature = 6
    shaded_ventilated_temperature = 7
    other = 8

class BTC:
    circuit = 1 # manufacturer circuit compensation
    corrected_circuit = 2 # manufacturer circuit compensation with correction
    temperature = 3 # temperature measurement
    other = 4

class STATION:
    def __init__(self,station):
        self.station = station
    
    @property
    def id(self):
        mapping = {
            "actris-oscm": 87,
        }
        return mapping.get(self.station, 0)
    
    @property
    def abbr(self):
        mapping = {
            "actris-oscm": "MIN",
        }
        return mapping.get(self.station, "XXX")
    
    @property
    def pge_dome_temp_compensation(self):
        mapping = {
            "actris-oscm": DTC.shaded_ventilated,
        }
        return mapping.get(self.station, -1)
    
    @property
    def pge_body_temp_compensation(self):
        mapping = {
            "actris-oscm": BTC.temperature,
        }
        return mapping.get(self.station, -1)
    
    @property
    def surface_type(self):
        mapping = {
            "actris-oscm": SURFACE.concrete,
        }
        return mapping.get(self.station, -1)
    
    @property
    def topography(self):
        mapping = {
            "actris-oscm": TOPOGRAPHY.flat_rural,
        }
        return mapping.get(self.station, -1)
    
    @property
    def remarks_shading(self):
        mapping = {
            "actris-oscm": "SHADING WITH SHADOW-BALL",
        }
        return mapping.get(self.station, "XXX")
    
    @property
    def horizon(self):
        """Read the latest horizon file generated with the ASI software.
        Expecting a tab delimited ascii file:
         * empty lines will be skipped
         * first line comment or header (will be skipped)
         * first column azimuth (deg clockwise from south)
         * second column elevation (deg)
        """
        fn = os.path.join(
            importlib.resources.files("taro"),
            "share/horizon_{station}_{dt:%Y%m%d}.txt"
        )
        fname_pattern = re.sub(r"\{dt:[^{}]*\}", "*", fn)
        fname_pattern = fname_pattern.format(station=self.station)
        fnames = sorted(glob.glob(fname_pattern))
        if len(fnames)==0:
            return [0],[0], pd.to_datetime("2000-01-01")
        fname = fnames[-1] # choose the latest file
        finfo = parse.parse(fn,fname).named

        horizon_update_date = finfo["dt"]
        horizon_df = pd.read_csv(fname,sep="\s+",header=None,skiprows=1,skip_blank_lines=True)
        azi = horizon_df.values[:,0] # positive clockwise from south
        ele = horizon_df.values[:,1]

        # transform azi to clockwise positive from north
        azi += 180
        azi[azi>=360] -= 360

        # round to integer
        azi = np.round(azi,0).astype(int)
        ele = np.round(ele,0).astype(int)

        # rotate to start with north proceeding clockwise
        isort = np.argmin(azi)
        if isort !=0 :
            azi = np.concatenate((azi[isort:],azi[:isort]),axis=0)
            ele = np.concatenate((ele[isort:],ele[:isort]),axis=0)

        u,c = np.unique(azi,return_counts=True)
        if np.any(c>1):
            mask = c>1
            raise AttributeError(f"Horizon azimuth data has double values in {u[mask]}. Check file {fname}.")

        return azi,ele,horizon_update_date
    
    def plot_horizon(self):
        azi,ele,_ = self.horizon
        fig,ax = plt.subplots(1,1,subplot_kw={'projection': 'polar'})
        ax.fill_between(np.deg2rad(azi),90-ele,91)
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location(loc="N")
        return fig,ax


class BSRN:
    def __init__(self,input,bsrn_path='.',config=None,version=None):
        config = taro.utils.merge_config(config)
        self.config = config

        if os.path.isdir(input):
            input = os.path.join(input,"*.nc")
        if os.path.isfile(input):
            input = os.path.join(os.path.dirname(input),"*.nc")
        self.fnames = sorted(glob.glob(input))
        l1b, attrs_catalog = self.load_l1b()
        self._l1b = l1b
        self._attrs_catalog = attrs_catalog
        
        self.station = STATION(l1b.site)
        self.bsrn_path = os.path.abspath(bsrn_path)
        if version is None:
            # if file already exists, read its version and add 1
            if os.path.exists(self.s2a_fname):
                with gzip.open(self.s2a_fname,'rt') as txt:
                    _ = txt.readline()
                    l2 = txt.readline() # read second line with version information
                self.version = int(l2.split()[-1]) + 1
            else:
                self.version = 1
        else:
            self.version = version
    
    def load_l1b(self):
        def _update_attrs_catalog(catalog,new):
            """compare attributes dictionary and merge if an update happend.
            Expectes catalogs in the form {'var1':{**attrs,'time':datetime64}, 'var2'...}
            """
            def append(x):
                # check first element if iterable
                # make it iterable if needed
                # define 'a' the element to compare to the second input
                # 'a' should be the last element in the list
                if isinstance(x[0],list):
                    a = x[0][-1]
                    m = x[0]
                else:
                    a = x[0]
                    m = [x[0]]
                # compare and if equal just return the first element
                if a == x[1]:
                    return m
                else:
                    m.append(x[1])
                    return m
                
            # for each variable in the catalog, we compare the attributes and append if something has changed
            for var in catalog:
                res = merge_with(append, catalog[var],new[var])
                # time is always merged, so we compare the length of time to all attributes
                tlen = len(res["time"])
                res_same = valfilter(lambda x: len(x)!= tlen, res)
                res_update = valfilter(lambda x: len(x)== tlen, res)

                if len(res_update.keys())>1:
                    # there is more variables then time with some updates.
                    # So, we pad the not updated variables with their last entry to match the length
                    res_same = valmap(lambda x: x + [x[-1]], res_same)
                else:
                    # Only time receives an update, therefore we drop the last time entry
                    res_update = valmap(lambda x: x[:-1], res_update)

                var_merged =  merge(res_same,res_update)
                catalog = assoc_in(catalog,[var],var_merged)
            return catalog
        
        # get first file for inspection
        l1b = xr.load_dataset(self.fnames[0])
        keep_vars = self.config["bsrn_variables"]
        keep_vars += [var+'_min' for var in keep_vars]
        keep_vars += [var+'_max' for var in keep_vars]
        keep_vars += [var+'_std' for var in keep_vars]
        # keep pyrgeometer sensor temperature
        pges = list(l1b.filter_by_attrs(standard_name="surface_downwelling_longwave_flux_in_air").data_vars)
        pge_var = [ var for var in pges if var in self.config["bsrn_variables"] ][0]
        pgeID = l1b[pge_var].attrs["troposID"]
        pge_tmp_var = list(l1b.filter_by_attrs(standard_name="temperature_of_sensor").filter_by_attrs(troposID=pgeID).data_vars)[0]
        keep_vars.append(pge_tmp_var)

        drop_vars = [var for var in l1b if var not in keep_vars]

        # read some needed variables and then only keep variables for bsrn
        self.altitude = int(np.round(l1b.altitude.median(skipna=True).values,0))
        l1b = l1b.drop_vars(drop_vars)

        # check requirements for bsrn (calibrated, 1min resolution)
        if ("processing_level" not in l1b.attrs) or (l1b.processing_level != 'l1b'):
            raise AttributeError("Dataset must have a 'processing_level='l1b'' attribute.")
        # res = np.unique_counts(l1b.time.values.astype("datetime64[m]"))
        _,counts = np.unique(l1b.time.values.astype("datetime64[m]"),return_counts=True)
        if np.any(counts>1):
            raise AttributeError("Dataset must be in 1min resolution without double values.")
        
        # init attribute catalog
        timestr = f"{pd.to_datetime(l1b.time.values[0]):%Y-%m-%dT%H:%M}"
        attrs_catalog = { 
            var : {**valmap(str,l1b[var].attrs),"time":timestr} for var in l1b.data_vars
        }

        # load and merge rest of the files
        for fn in self.fnames[1:]:
            l1b_tmp =  xr.open_dataset(fn)
            l1b_tmp = l1b_tmp.drop_vars(drop_vars)

            timestr = f"{pd.to_datetime(l1b_tmp.time.values[0]):%Y-%m-%dT%H:%M}" 
            attrs_new = {
                var : {**valmap(str,l1b_tmp[var].attrs),"time":timestr} for var in l1b_tmp.data_vars
            }
            attrs_catalog = _update_attrs_catalog(attrs_catalog,attrs_new)

            l1b = xr.concat((l1b,l1b_tmp),dim='time',combine_attrs="override")
        
        return l1b,attrs_catalog


    @property
    def s2a_fname(self):
        """Define station-to-archive file name
        """
        station_abbr = self.station.abbr.lower()
        date = pd.to_datetime(self._l1b.time_coverage_start)
        fname = f"{station_abbr}{date:%m%y}.dat.gz"
        fname = os.path.join(self.bsrn_path,fname)
        return fname
    
    @property
    def raw_fname(self):
        """Define station-to-archive file name
        """
        station_abbr = self.station.abbr.lower()
        date = pd.to_datetime(self._l1b.time_coverage_start)
        fname = f"MON_{station_abbr}{date:%m%y}_avg.csv.gz"
        fname = os.path.join(self.bsrn_path,fname)
        return fname
    
    @property
    def s2a_fname_past(self):
        """Define station-to-archive file name
        """
        station_abbr = self.station.abbr.lower()
        date = pd.to_datetime(self._l1b.time_coverage_start)
        date -= pd.DateOffset(months=1)
        fname = f"{station_abbr}{date:%m%y}.dat.gz"
        fname = os.path.join(self.bsrn_path,fname)
        if os.path.exists(fname):
            return fname
        else:
            return None
    
    def add_unchanged_indicator(self,lines):
        if self.s2a_fname_past is None:
            return lines
        
        past_lines = []
        with gzip.open(self.s2a_fname_past,'rt') as txt:
            for i in range(200):
                past_lines.append(txt.readline())

        lr = ''
        past_lrs = []
        for line in past_lines:
            if line.startswith('*'):
                past_lrs.append(lr)
                lr = ''
                continue
            lr += line
            
        past_lrs = past_lrs[1:] # skip first (empty) entry

        lr = ''
        lrs = []
        line_nos = []
        for line_no,line in enumerate(lines[:200]):
            if line.startswith('*'):
                line_nos.append(line_no)
                lrs.append(lr)
                lr = ''
                continue
            lr += line
        lrs = lrs[1:] # skip first (empty) entry
        
        for i,(lr,lrp) in enumerate(zip(lrs,past_lrs)):
            if lr == lrp:
                lines[line_nos[i]] = lines[line_nos[i]].replace('C','U')

        return lines

    @property
    def lr_0001(self):
        date = pd.to_datetime(self._l1b.time_coverage_start)
        format = dict(
            station_id = self.station.id,
            month = date.month,
            year = date.year,
            version = self.version
        )
        qid,_ = self.quantities
        if len(qid)%8 != 0:
            qid += [-1]*(8-len(qid)%8)  # add -1 to fill up line (80 characters)
        
        lines = [
            "*C0001\n",
            " {station_id:2d} {month:2d} {year:4d} {version:2d}\n"
        ]

        idline = ""
        for i,id in enumerate(qid):
            idline += f" {id:9d}"
            if i%8 == 7:
                idline += "\n"
                lines.append(idline)
                idline = ""
     
        return [ l.format_map(format) for l in lines ]
    
    @property
    def lr_0002(self):
        config = {**self.config, "bsrn_stsci_tcp":"XXX", "bsrn_stdep_tcp":"XXX"}
        lines = [ 
            "*C0002\n",
            " -1 -1 -1\n",  # date when scientist changed (day, hour, min) (i2)
            "{bsrn_stsci_name:38.38s} {bsrn_stsci_phone:20.20s} {bsrn_stsci_fax:20.20s}\n",
            "{bsrn_stsci_tcp:15.15s} {bsrn_stsci_mail:50.50s}\n",
            "{bsrn_stsci_address:80.80s}\n",
            " -1 -1 -1\n", # date when deputy changed (day, hour, min) (i2)
            "{bsrn_stdep_name:38.38s} {bsrn_stdep_phone:20.20s} {bsrn_stdep_fax:20.20s}\n",
            "{bsrn_stdep_tcp:15.15s} {bsrn_stdep_mail:50.50s}\n",
            "{bsrn_stdep_address:80.80s}\n",
        ]
        return [ l.format_map(config) for l in lines ]
    
    @property
    def lr_0003(self):
        lines = [
            "*C0003\n",
            f"{'XXX':80.80s}\n"
        ]
        return lines
    
    @property
    def lr_0004(self):
        # get latitude and convert to (deg, 0=south, positive north) [0,180]
        lat = self._l1b.attrs["geospatial_lat_min"]
        if self._l1b.attrs["geospatial_lat_units"] == "degN":
            lat += 90.
        elif self._l1b.attrs["geospatial_lat_units"] == "degS":
            lat -= 90.
            lat *= -1.
        else:
            raise AttributeError("Attribute 'geospation_lat_units' is non of ['degN','degS'].")

        # get longitude and convert to (deg, 0=180W, positive east) [0,360)
        lon = self._l1b.attrs["geospatial_lon_min"]
        if self._l1b.attrs["geospatial_lon_units"] == "degE":
            lon += 180.
        elif self._l1b.attrs["geospatial_lon_units"] == "degW":
            lon -= 180.
            lon *= -1.
        else:
            raise AttributeError("Attribute 'geospation_lon_units' is non of ['degE','degW'].")

        lines = [
            "*C0004\n",
            " -1 -1 -1\n",  # date when station description has changed (day, hour, min) (i2)
            f" {self.station.surface_type:2d} {self.station.topography:2d}\n",
            "{bsrn_st_address:80.80s}\n",
            f"{'XXX':20.20} {'XXX':20.20s}\n", # station phone, station fax
            f"{'XXX':15.15} {'XXX':50.50s}\n", # station tcp, station mail
            f" {lat:7.3f} {lon:7.3f} {self.altitude:4d} XXXXX\n", # lat, lon, altitude, SYNOP id
            " -1 -1 -1\n",  # date when horizon has changed (day, hour, min) (i2)
        ]

        azis,eles,_ = self.station.horizon
        azis = list(azis)
        eles = list(eles)
        if len(azis)%11 != 0:
            azis += [-1]*(11-len(azis)%11)  # add -1 to fill up line (80 characters)
            eles += [-1]*(11-len(eles)%11)  # add -1 to fill up line (80 characters)
        hline = ""
        for i,(azi,ele) in enumerate(zip(azis,eles)):
            hline += f" {azi:3d} {ele:2d}"
            if i%11 == 10:
                hline += "\n"
                lines.append(hline)
                hline = ""

        return [ l.format_map(self.config) for l in lines ]
    
    @property
    def lr_0005(self):
        lines = [
            "*C0005\n",
            " -1 -1 -1 N\n",
            f"{'XXX':30.30s} {'XXX':25.25s} {0:03d} -1 -1 -1 -1 {'XXX':5.5s}\n",
            f"{'XXX':80.80}\n"
        ]
        return lines
    
    @property
    def lr_0006(self):
        lines = [
            "*C0006\n",
            " -1 -1 -1 N\n",
            f"{'XXX':30.30s} {'XXX':25.25s} {0:3d} {'XXX':5.5s}\n",
            f"{'XXX':80.80}\n"
        ]
        return lines
    
    @property
    def lr_0007(self):
        lines = [
            "*C0007\n",
            " -1 -1 -1\n",
            f"{'XXX':80.80}\n",
            f"{'XXX':80.80}\n",
            f"{'XXX':80.80}\n",
            f"{'XXX':80.80}\n",
            f"{'XXX':80.80}\n",
            "N N N N N N\n"
        ]
        return lines
    
    @property
    def lr_0008(self):
        meta_lines = [
            "{manufacturer:30.30s} {model:15.15s} {serial:18.18s} {date_purchase:8.8s} {wrmc_id:5d}\n",
            "{remarks:80.80s}\n",
            " {pge_bcc:2d} {pge_dcc:2d} {wvl1:7.3f} {bw1:7.3f} {wvl2:7.3f} {bw2:7.3f} {wvl3:7.3f} {bw3:7.3f} {max_zen_dni:2d} {min_ele_spc:2d}\n",
            "{calib_loc:30.30s} {calib_person:40.40s}\n",
            "{calib1_start:8.8s} {calib1_end:8.8s} {calib1_no:2d} {calib1_fac:12.4f} {calib1_err:12.4f}\n",
            "{calib2_start:8.8s} {calib2_end:8.8s} {calib2_no:2d} {calib2_fac:12.4f} {calib2_err:12.4f}\n",
            "{calib3_start:8.8s} {calib3_end:8.8s} {calib3_no:2d} {calib3_fac:12.4f} {calib3_err:12.4f}\n",
            "UNIT OF CAL. COEFF.: {calib_units:59.59s}\n",
            "{calib_remarks:80.80s}\n",
        ]
        meta_default = dict(
            manufacturer = 'XXX',
            model = 'XXX',
            serial = 'XXX',
            date_purchase = 'XXX',
            wrmc_id = -1,
            remarks = 'XXX',
            pge_bcc = -1,
            pge_dcc = -1,
            wvl1 = -1.,
            bw1 = -1.,
            wvl2 = -1.,
            bw2 = -1.,
            wvl3 = -1.,
            bw3 = -1.,
            max_zen_dni = -1,
            min_ele_spc = -1,
            calib_loc = 'XXX',
            calib_person = 'XXX',
            calib1_start = 'XXX',
            calib1_end = 'XXX',
            calib1_no = 1,
            calib1_fac = -1.,
            calib1_err = -1.,
            calib2_start = 'XXX',
            calib2_end = 'XXX',
            calib2_no = 1,
            calib2_fac = -1.,
            calib2_err = -1.,
            calib3_start = 'XXX',
            calib3_end = 'XXX',
            calib3_no = 1,
            calib3_fac = -1.,
            calib3_err = -1.,
            calib_units = 'XXX',
            calib_remarks = 'XXX'
        )

        # init lines
        lines = [
            "*C0008\n",
        ]
        
        # iterate catalog variables, only radiation related
        for var in self._attrs_catalog.values():
            if var["standard_name"][0] not in class2dict(QUANTITIES.RADIATION):
                continue

            # iterate all changes
            for i in range(len(var["standard_name"])):
                date_of_change = pd.to_datetime(var["time"][i])
                if i == 0:
                    lines += [ " -1 -1 -1 Y\n"]
                else:
                    lines += [ f" {date_of_change.day:2d} {date_of_change.hour:2d} {date_of_change.minute:2d} Y\n"]

                imeta = taro.utils.meta_lookup(
                    date = date_of_change,
                    troposID=var["troposID"][i],
                    config=self.config
                )

                if "device" in imeta:
                    imeta.update(dict(
                        manufacturer = imeta["device"].split(",")[-1].strip(),
                        model = imeta["device"].split(",")[0].split()[-1].strip()
                    ))
                if "date_purchase" in imeta:
                    imeta.update(dict(
                        date_purchase = f"{pd.to_datetime(imeta['date_purchase']):%m/%d/%y}"
                    ))

                imeta.update(dict(
                    calib1_fac = imeta["calibration_factor"],
                    calib1_no = imeta["calibration_repeats"],
                    calib1_err = imeta["calibration_factor"]*imeta["calibration_error"]*1e-2, # standard error in calibration factor units
                    calib1_start = f"{pd.to_datetime(imeta['calibration_period_start']):%m/%d/%y}",
                    calib1_end = f"{pd.to_datetime(imeta['calibration_period_end']):%m/%d/%y}",
                    calib_units = imeta["calibration_factor_units"],
                    calib_person = imeta["calibration_person"],
                    calib_loc = imeta["calibration_location"],
                    calib_remarks = imeta["calibration_remarks"],
                ))

                if var["standard_name"][i] == "surface_downwelling_longwave_flux_in_air":
                    imeta.update(
                        dict(
                            pge_bcc = self.station.pge_body_temp_compensation,
                            pge_dcc = self.station.pge_dome_temp_compensation
                        )
                    )
                    if "temperature_correction_coef" in imeta:
                        a,b,c = imeta["temperature_correction_coef"]
                        imeta.update(
                            dict(
                                remarks = f"BTC: U/U0=1{a:+.4e}T^2{b:+.4e}T{c:+.4e}"
                            )
                        )
                    if self.station.pge_dome_temp_compensation in [DTC.shaded,DTC.shaded_temperature,DTC.shaded_ventilated,DTC.shaded_ventilated_temperature]:
                        if "remarks" in imeta:
                            imeta.update(
                                dict(
                                    remarks = imeta["remarks"] + f"; {self.station.remarks_shading}"
                                )
                            )
                        else:
                            imeta.update(
                                dict(
                                    remarks = self.station.remarks_shading
                                )
                            )
                if var["standard_name"][i] == "surface_diffuse_downwelling_shortwave_flux_in_air":
                    imeta.update(
                            dict(
                                remarks = self.station.remarks_shading
                            )
                        )

                
                # merge with defaults
                imeta = {**meta_default, **imeta}

                # add lines
                lines += [ line.format_map(imeta) for line in meta_lines ]

        return lines
    
    @property
    def lr_0009(self):
        mapping = class2dict(QUANTITIES.RADIATION)
        lines= [
            "*C0009\n"
        ]
        # iterate catalog variables, only radiation related
        for var in self._attrs_catalog.values():
            if var["standard_name"][0] not in mapping:
                continue
            # iterate all changes:
            for i in range(len(var["standard_name"])):
                date_of_change = pd.to_datetime(var["time"][i])
                imeta = taro.utils.meta_lookup(
                    date = date_of_change,
                    troposID=var["troposID"][i],
                    config=self.config
                )
                if i == 0:
                    lines.append(f" -1 -1 -1 {mapping[var['standard_name'][i]]:9d} {imeta['wrmc_id']:5d} -1\n")
                else:
                    lines.append(f" {date_of_change.day:2d} {date_of_change.hour:2d} {date_of_change.minute:2d} {mapping[var['standard_name'][i]]:9d} {imeta['wrmc_id']:5d} -1\n")

        return lines
    
    @property
    def lr_0100(self):
        lines= [
            "*C0100\n"
        ]
        ghi_var = list(self._l1b.filter_by_attrs(standard_name="surface_downwelling_shortwave_flux_in_air"))[0]
        dhi_var = list(self._l1b.filter_by_attrs(standard_name="surface_direct_along_beam_shortwave_flux_in_air"))[0]
        dni_var = list(self._l1b.filter_by_attrs(standard_name="surface_downwelling_shortwave_flux_in_air"))[0]
        lw_var = list(self._l1b.filter_by_attrs(standard_name="surface_downwelling_longwave_flux_in_air"))[0]
        tair_var = list(self._l1b.filter_by_attrs(standard_name="air_temperature"))[0]
        rh_var = list(self._l1b.filter_by_attrs(standard_name="relative_humidity"))[0]
        pres_var = list(self._l1b.filter_by_attrs(standard_name="air_pressure"))[0]

        ds = self._l1b
        for i in range(ds.time.size):
            date = pd.to_datetime(ds.time.values[i])
            day = int(date.day)
            minute = int(date.hour*60 + date.minute)
            ghi = int(round(ds[ghi_var].values[i],0))
            ghi_min = int(round(ds[ghi_var+'_min'].values[i],0))
            ghi_max = int(round(ds[ghi_var+'_max'].values[i],0))
            ghi_std = float(round(ds[ghi_var+'_std'].values[i],1))
            dhi = int(round(ds[dhi_var].values[i],0))
            dhi_min = int(round(ds[dhi_var+'_min'].values[i],0))
            dhi_max = int(round(ds[dhi_var+'_max'].values[i],0))
            dhi_std = float(round(ds[dhi_var+'_std'].values[i],1))
            dni = int(round(ds[dni_var].values[i],0))
            dni_min = int(round(ds[dni_var+'_min'].values[i],0))
            dni_max = int(round(ds[dni_var+'_max'].values[i],0))
            dni_std = float(round(ds[dni_var+'_std'].values[i],1))
            lw = int(round(ds[lw_var].values[i],0))
            lw_min = int(round(ds[lw_var+'_min'].values[i],0))
            lw_max = int(round(ds[lw_var+'_max'].values[i],0))
            lw_std = float(round(ds[lw_var+'_std'].values[i],1))
            pres = int(round(ds[pres_var].values[i]/100.,0))
            rh = float(round(ds[rh_var].values[i]*100.,1))
            tair = float(round(ds[tair_var].values[i]-273.15,1))

            line = f" {day:2d} {minute:4d}"
            # ghi
            line += f"   {ghi:4d} {ghi_std:5.1f} {ghi_min:4d} {ghi_max:4d}"
            # dni
            line += f"   {dni:4d} {dni_std:5.1f} {dni_min:4d} {dni_max:4d}\n"
            lines.append(line)
            line = "        "
            # dhi
            line += f"   {dhi:4d} {dhi_std:5.1f} {dhi_min:4d} {dhi_max:4d}"
            # lw
            line += f"   {lw:4d} {lw_std:5.1f} {lw_min:4d} {lw_max:4d}"
            # air temp, rel hum, air press
            line += f"    {tair:5.1f} {rh:5.1f} {pres:4d}\n"
            lines.append(line)
        return lines

    @property
    def lr_4000(self):
        lines= [
            "*C4000\n"
        ]
        pgetemp_var = list(self._l1b.filter_by_attrs(standard_name="temperature_of_sensor"))[0]
        lw_var = list(self._l1b.filter_by_attrs(standard_name="surface_downwelling_longwave_flux_in_air"))[0]

        ds = self._l1b
        for i in range(ds.time.size):
            date = pd.to_datetime(ds.time.values[i])
            day = int(date.day)
            minute = int(date.hour*60 + date.minute)
            pgetmp = float(round(ds[pgetemp_var].values[i]-273.15,2))
            pgev0 = ds[lw_var].values[i] - 5.670367e-8*ds[pgetemp_var].values[i]**4
            pgev0 = float(round(pgev0,1))
            line = f" {day:2d} {minute:4d}"
            line+= f" {-99.99:6.2f} {-99.99:6.2f} {-99.99:6.2f} {pgetmp:6.2f} {pgev0:6.1f}"
            line+= f"  {-99.99:6.2f} {-99.99:6.2f} {-99.99:6.2f} {-99.99:6.2f} {-999.9:6.1f}\n"
            lines.append(line)
        return lines
    
    
    @property
    def quantities(self):
        mapping = class2dict(QUANTITIES)
        qvar,qid = [],[]

        for var in self._l1b:
            if self._l1b[var].attrs["standard_name"] in mapping:
                qvar.append(var)
                qid.append(mapping[self._l1b[var].attrs["standard_name"]])
        
        isort = np.argsort(qid)
        qid = [ qid[i] for i in isort ]
        qvar = [ qvar[i] for i in isort ]
        return qid,qvar

    def to_bsrn_raw(self):
        mapping = class2dict(QUANTITIES_RAW)

        unit_map = {
            "K" : (-273.15,1), # -> to degC
            "1" : (0,100.), # -> to %
            "Pa": (0,1e-2), # -> to hPa (mbar)
        }

        name_map = {}
        for var in self._l1b:
            unit = self._l1b[var].attrs["units"]
            if unit in unit_map:
                offset,scale = unit_map[unit]
                self._l1b[var].values = offset + self._l1b[var].values*scale

            sname = self._l1b[var].attrs["standard_name"]
            sname = sname.replace("min_","").replace("max_","").replace("std_","")
            name = mapping[sname]
            if var.endswith("min") or var.endswith("max") or var.endswith("std"):
                name += "_"+var.split("_")[-1]
            name_map.update({var:name})

        df = self._l1b.to_pandas().rename(columns=name_map)
        df = df.reset_index()
        # Create a date‑only column
        df.insert(0,"date", df["time"].dt.date)
        # Create a time‑only column
        df["time"] = df["time"].dt.time  
        
        df.to_csv(
            os.path.join(self.bsrn_path,"raw/",self.raw_fname),
            sep=',',
            index=False,
            compression="gzip"
        )

    def to_bsrn(self):

        lines = self.lr_0001
        lines += self.lr_0002
        lines += self.lr_0003
        lines += self.lr_0004
        lines += self.lr_0005
        lines += self.lr_0006
        lines += self.lr_0007
        lines += self.lr_0008
        lines += self.lr_0009
        lines += self.lr_0100
        lines += self.lr_4000

        lines = self.add_unchanged_indicator(lines)

        with gzip.open(self.s2a_fname,'wt') as txt:
            txt.writelines(lines)
