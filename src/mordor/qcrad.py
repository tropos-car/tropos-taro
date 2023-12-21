import numpy as np
from unitpy import Unit
import trosat.sunpos as sp

class QCCode:
    """ BSRN quality codes
    https://wiki.pangaea.de/wiki/BSRN_Toolbox#Quality_Check
    """
    below_physical = 2**0
    above_phyiscal = 2**1
    below_rare = 2**2
    above_rare = 2**3
    compare_to_low = 2**4
    compare_to_high = 2**5

class CONSTANTS:
    S0 = 1367  # W m-2
    k = 5.67*1e-8

class SNAMES:
    szen = "solar_zenith_angle"
    dni = "surface_direct_along_beam_shortwave_flux_in_air"
    ghi = "surface_downwelling_shortwave_flux_in_air"
    dhi = "surface_diffuse_downwelling_shortwave_flux_in_air"
    lwd = "surface_downwelling_longwave_flux_in_air"
    tair = "air_temperature"
    tdew = "dew_point_temperature"
    twb = "wet_bulb_temperature"
    tsens = "temperature_of_sensor"
    pair = "air_pressure"
    rh = "relative_humidity"
    qc = "quality_flag"
    freq = "frequency"

def quality_control(ds, lat=None, lon=None):
    def _init_qc(ds, var):
        qc_bits = [2**i for i in range(6)]
        ds[var + "_qc"] = ds[var].copy()
        ds[var + "_qc"].values *= 0
        ds[var + "_qc"].attrs.update({
            "standard_name": "quality_flag",
            "units": "-",
            "ancillary_variables": var,
            "valid_range": [0, np.sum(qc_bits)],
            "flag_masks": qc_bits,
            "flag_values": qc_bits,
            "flag_meanings": str(
                "below_physical_limit" + " " +
                "above_physical_limit" + " " +
                "below_rare_limit" + " " +
                "above_rare_limit" + " " +
                "comparison_to_low" + " " +
                "comparison_to_high"
            )
        })
        ds[var + "_qc"].encoding.update({
            "dtype": "u1",
            "_FillValue": 0
        })
        return ds

    # retrieve solar zenith angle from data
    szen = None
    for var in ds.filter_by_attrs(standard_name="solar_zenith_angle"):
        szen = ds[var].values*Unit(ds[var].attrs["units"])
        break
    # calculate solar zenith angle if not in data
    if szen is None:
        for var in ds.filter_by_attrs(standard_name="latitude"):
            lat = ds[var].values
            break
        for var in ds.filter_by_attrs(standard_name="longitude"):
            lon = ds[var].values
            break
        assert lat is not None
        assert lon is not None
        szen,_ = sp.sun_angles(ds.time.values, lat=lat, lon=lon)*Unit("degrees")

    mu0 = np.cos(szen.to("radian").value)
    mu0[mu0 < 0] = 0 #  exclude night
    szen = szen.value
    esd = sp.earth_sun_distance(ds.time.values)
    Sa = CONSTANTS.S0 / esd**2

    # GHI
    for var in ds.filter_by_attrs(standard_name=SNAMES.ghi):
        # init quality control variable
        ds = _init_qc(ds, var)
        # physical minimum
        mask = ds[var].values < -4
        ds[var+"_qc"].values[mask] += QCCode.below_physical
        # physical maximum
        mask = ds[var].values > ((Sa * 1.5 * mu0 ** 1.2) + 100)
        ds[var+"_qc"].values[mask] += QCCode.above_phyiscal
        # rare limit minimum
        mask = ds[var].values < -2
        ds[var + "_qc"].values[mask] += QCCode.below_rare
        # rare limit maximum
        mask = ds[var].values > ((Sa * 1.2 * mu0 ** 1.2) + 50)
        ds[var + "_qc"].values[mask] += QCCode.above_rare
        # ratio ghi / (dni*mu0 + dhi)
        index = 1
        for dnivar in ds.filter_by_attrs(standard_name=SNAMES.dni):
            dni = ds[dnivar].values
            for dhivar in ds.filter_by_attrs(standard_name=SNAMES.dhi):
                dhi = ds[dhivar].values
                sumsw = dni*mu0 + dhi
                ratio = np.ones(ds.time.size)
                ratio[sumsw > 50] = ds[var].values[sumsw > 50]/sumsw[sumsw > 50]
                thres_low = np.ones(ds.time.size)*0.92
                thres_low[szen>75] = 0.85
                thres_high = np.ones(ds.time.size) * 1.08
                thres_high[szen > 75] = 1.15
                mask_to_low = sumsw > 50
                mask_to_low *= ratio < thres_low
                mask_to_high = sumsw > 50
                mask_to_high *= ratio > thres_high
                if index == 1:
                    # comparison to low
                    ds[var + "_qc"].values[mask_to_low] += QCCode.compare_to_low
                    # comparison to high
                    ds[var + "_qc"].values[mask_to_high] += QCCode.compare_to_high
                    ds[var + "_qc"].attrs.update({
                        "comment": "comparison ratio GHI / (DNI*mu0 + DHI)",
                        "ancillary_variables": ds[var + "_qc"].attrs["ancillary_variables"] + f" {dnivar} {dhivar}"
                    })
                else:
                    ds[var + f"_qc_{index}"] = ds[var+"_qc"].copy()
                    # comparison to low
                    ds[var + f"_qc_{index}"].values[mask_to_low] += QCCode.compare_to_low
                    # comparison to high
                    ds[var + f"_qc_{index}"].values[mask_to_high] += QCCode.compare_to_high
                    ds[var + f"_qc_{index}"].attrs.update({
                        "comment": "comparison ratio GHI / (DNI*mu0 + DHI)",
                        "ancillary_variables": ds[var + f"_qc_{index}"].attrs["ancillary_variables"] + f" {dnivar} {dhivar}"
                    })
                index += 1

    # DHI
    for var in ds.filter_by_attrs(standard_name=SNAMES.dhi):
        # init quality control variable
        ds = _init_qc(ds, var)
        # physical minimum
        mask = ds[var].values < -4
        ds[var+"_qc"].values[mask] += QCCode.below_physical
        # physical maximum
        mask = ds[var].values > ((Sa * 0.95 * mu0**1.2) + 50)
        ds[var+"_qc"].values[mask] += QCCode.above_phyiscal
        # rare limit minimum
        mask = ds[var].values < -2
        ds[var + "_qc"].values[mask] += QCCode.below_rare
        # rare limit maximum
        mask = ds[var].values > ((Sa * 0.75 * mu0 ** 1.2) + 30)
        ds[var + "_qc"].values[mask] += QCCode.above_rare
        # ratio dhi / ghi
        index = 1
        for ghivar in ds.filter_by_attrs(standard_name=SNAMES.ghi):
            ghi = ds[ghivar].values
            ratio = np.ones(ds.time.size)
            ratio[ghi>50] = ds[var].values[ghi>50] / ghi[ghi>50]
            thres_low = np.zeros(ds.time.size)
            thres_high = np.ones(ds.time.size) * 1.05
            thres_high[szen > 75] = 1.10
            mask_to_low = ghi > 50
            mask_to_low *= ratio < thres_low
            mask_to_high = ghi > 50
            mask_to_high *= ratio > thres_high
            if index == 1:
                # comparison to low
                ds[var + "_qc"].values[mask_to_low] += QCCode.compare_to_low
                # comparison to high
                ds[var + "_qc"].values[mask_to_high] += QCCode.compare_to_high
                ds[var + "_qc"].attrs.update({
                    "comment": "comparison ratio DHI / GHI",
                    "ancillary_variables": ds[var + "_qc"].attrs["ancillary_variables"] + f" {ghivar}"
                })
            else:
                ds[var + f"_qc_{index}"] = ds[var+"_qc"].copy()
                # comparison to low
                ds[var + f"_qc_{index}"].values[mask_to_low] += QCCode.compare_to_low
                # comparison to high
                ds[var + f"_qc_{index}"].values[mask_to_high] += QCCode.compare_to_high
                ds[var + f"_qc_{index}"].attrs.update({
                    "comment": "comparison ratio DHI / GHI",
                    "ancillary_variables": ds[var + f"_qc_{index}"].attrs["ancillary_variables"] + f" {ghivar}"
                })
            index += 1

    # DNI
    for var in ds.filter_by_attrs(standard_name=SNAMES.dni):
        # init quality control variable
        ds = _init_qc(ds, var)
        # physical minimum
        mask = ds[var].values < -4
        ds[var+"_qc"].values[mask] += QCCode.below_physical
        # physical maximum
        mask = ds[var].values > Sa
        ds[var+"_qc"].values[mask] += QCCode.above_phyiscal
        # rare limit minimum
        mask = ds[var].values < -2
        ds[var + "_qc"].values[mask] += QCCode.below_rare
        # rare limit maximum
        mask = ds[var].values > ((Sa * 0.95 * mu0 ** 0.2) + 10)
        ds[var + "_qc"].values[mask] += QCCode.above_rare

    # LWD
    for var in ds.filter_by_attrs(standard_name=SNAMES.lwd):
        # init quality control variable
        ds = _init_qc(ds, var)
        # physical minimum
        mask = ds[var].values < 40
        ds[var+"_qc"].values[mask] += QCCode.below_physical
        # physical maximum
        mask = ds[var].values > 700
        ds[var+"_qc"].values[mask] += QCCode.above_phyiscal
        # rare limit minimum
        mask = ds[var].values < 60
        ds[var + "_qc"].values[mask] += QCCode.below_rare
        # rare limit maximum
        mask = ds[var].values > 500
        ds[var + "_qc"].values[mask] += QCCode.above_rare

        # LWD vs Air temperature
        for tvar in ds.filter_by_attrs(standard_name=SNAMES.tair):
            temp_air = ds[tvar].values * Unit(ds[tvar].attrs['units'])
            break
        temp_air = (temp_air.to('K')).value
        thres_low = 0.4 * CONSTANTS.k * temp_air**4
        thres_high = 25 + CONSTANTS.k * temp_air**4
        # comparison to low
        mask = ds[var].values < thres_low
        ds[var + "_qc"].values[mask] += QCCode.compare_to_low
        # comparison to high
        mask = ds[var].values > thres_high
        ds[var + "_qc"].values[mask] += QCCode.compare_to_high
        ds[var + "_qc"].attrs.update({
            "comment": "comparison to 5.67e-8*air_temperature**4",
            "ancillary_variables": ds[var + "_qc"].attrs["ancillary_variables"] + f" {tvar}"
        })

    return ds