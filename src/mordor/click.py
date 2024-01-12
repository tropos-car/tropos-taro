import os.path

import click
import logging
import parse
import xarray as xr
import pandas as pd
import numpy as np
import datetime as dt
import importlib.resources
import matplotlib.pyplot as plt

import mordor
import mordor.utils
import mordor.futils
import mordor.data
import mordor.plot


logger = logging.getLogger(__name__)

DEFAULT_CONFIG = fn_config = os.path.join(
        importlib.resources.files("mordor"),
        "conf/mordor_config.json"
    )
def _configure(config):
    if config is None:
        config = mordor.utils.get_default_config()
    else:
        config = mordor.utils.read_json(os.path.abspath(config))
        config = mordor.utils.merge_config(config)
    return config

# initialize commandline interface
@click.version_option()
@click.group("mordor")
def cli():
    pass

##########################
# Raw data handling
##########################
@cli.command("raw2daily")
@click.argument("input_path", nargs=1)
@click.argument("output_path", nargs=1)
@click.option("--config", "-c", type=click.Path(dir_okay=False, exists=True),
              help="Config file - will merge and override the default config.")
def raw2daily(input_path: str,
              output_path: str,
              config):
    """
    Moves data lines from the  continuous Loggernet files within input path (table names specified in config) to daily files in output path.
    The output path is suffixed according to config['path_sfx'] (default is '{dt:%Y/%m/}').
    The daily output files are named according to config['fname_out'] (default is 'mordor_{dt:%Y-%m-%d}_{campaign}_{table}_{datalvl}.c{collection:02d}.{sfx}').
    """
    config = _configure(config)
    mordor.utils.init_logger(config)

    logger.info("Call mordor.futils.raw2daily")
    mordor.futils.raw2daily(
       inpath=os.path.abspath(input_path),
       outpath=os.path.abspath(output_path),
       config=config
    )

##########################
# Processing functions
##########################
@cli.group("process")
def process():
    print("Process")

@process.command("l1a")
@click.argument("input_files", nargs=-1)
@click.argument("output_path", nargs=1)
@click.option("--config", "-c", type=click.Path(dir_okay=False, exists=True),
              help="Config file - will merge and override the default config.")
def process_l1a(input_files,
        output_path: str,
        config: str):
    """
    Process input files (LoggerNet TOA5 ASCII Table Data) to rsd-car netcdf.
    """
    config = _configure(config)
    mordor.utils.init_logger(config)

    with click.progressbar(input_files, label='Processing to l1a:') as files:
        for fn in files:
            logger.info("Call mordor.data.to_l1a")
            ds = mordor.data.to_l1a(
               fname=fn,
               config=config
            )
            if ds is None:
                logger.warning(f"Skip {fn}.")
                continue

            fname_info = parse.parse(
                config["fname_out"],
                os.path.basename(fn)
            ).named

            fname_info.update({
                "datalvl": "l1a",
                "sfx": "nc"
            })

            outfile = os.path.join(output_path,
                                   "{dt:%Y/%m/}",
                                   config['fname_out'])
            outfile = outfile.format_map(fname_info)

            mordor.futils.to_netcdf(
                ds=ds,
                fname=outfile,
                timevar="time"
            )

@process.command("l1b")
@click.argument("input_files", nargs=-1)
@click.argument("output_path", nargs=1)
@click.option("--resolution","-r",
              default='1min', show_default=True,
              help="Time resolution of output file.")
@click.option("--config", "-c", type=click.Path(dir_okay=False, exists=True),
              help="Config file - will merge and override the default config.")
def process_l1b(input_files,
                output_path: str,
                resolution:str,
                config: str):
    """
    Process input files (mordor l1a) to mordor l1b files.
    Input files require at least the Radiation table (SensorStatus and Meteorologie are then assumed to be in the same directory).
    Flux variables are calibrated and corrected for sensor temperature.
    Dataset is resampled to desired resolution.
    Sun coordinate parameters are added.
    """
    config = _configure(config)
    mordor.utils.init_logger(config)

    fdates = []
    for fn in input_files:
        finfo = parse.parse(
            config["fname_out"],
            os.path.basename(fn)
        )
        fdates.append(finfo["dt"])
    udays, indices = np.unique(fdates, return_inverse=True)


    with click.progressbar(udays, label='Processing to l1b:') as days:
        for i, day in enumerate(days):
            ifiles = np.argwhere(indices == i).ravel()
            files = [input_files[j] for j in ifiles]
            ds_l1a = mordor.futils.merge_with_rename(
                [xr.load_dataset(fn) for fn in files],
                dim="time",
                override=['lat','lon','altitude']
            )
            ds_l1a = ds_l1a.interpolate_na('time')

            logger.info("Call mordor.data.to_l1a")
            ds = mordor.data.to_l1b(
                ds_l1a,
                resolution=resolution,
                config=config
            )
            if ds is None:
                logger.warning(f"Skip {fn}.")
                continue

            fname_info = parse.parse(
                config["fname_out"],
                os.path.basename(files[0])
            ).named

            fname_info.update({
                "table": "complete",
                "resolution": resolution,
                "datalvl": "l1b",
            })

            outfile = os.path.join(output_path,
                                   "{dt:%Y/%m/}",
                                   config['fname_out'])
            outfile = outfile.format_map(fname_info)
            # for key in ds:
            #     if "time" not in ds[key].dims:
            #         continue
            #     print(key)
            #     dst=ds[key]
            mordor.futils.to_netcdf(
                ds=ds,
                fname=outfile,
                timevar="time"
            )

###########################
# MORDOR plot
###########################
@cli.group("quicklook")
def quickook():
    print("Make quicklooks")

@quickook.command("data")
@click.argument("input_files", nargs=-1)
@click.argument("output_path", nargs=1)
@click.option("--config", "-c", type=click.Path(dir_okay=False, exists=True),
              help="Config file - will merge and override the default config.")
@click.option("--dpi", type=int, nargs=1,
              default=300, show_default=True,
              help="DPI for output png-file.")
def ql_data(input_files, output_path, config,dpi):
    config = _configure(config)
    mordor.utils.init_logger(config)

    with click.progressbar(input_files, label='Make daily data quicklooks:') as files:
        for fn in files:
            ds_l1b = xr.load_dataset(fn)
            fig, axs = plt.subplots(2, 1, figsize=(8, 8))
            plots = ds_l1b.quicklooks.flux(ax=axs[0])
            pl, (ax, pax, rax) = ds_l1b.quicklooks.meteorology(ax=axs[1], legend=False)
            ax.legend(handles=pl, loc='lower right')

            fname_info = parse.parse(
                config["fname_out"],
                os.path.basename(fn)
            ).named

            fname_info.update({
                "table": "data",
                "sfx": "png"
            })

            outfile = os.path.join(output_path,
                                   "{dt:%Y/%m/}",
                                   config['fname_out'])
            outfile = outfile.format_map(fname_info)

            os.makedirs(os.path.dirname(outfile), exist_ok=True)
            fig.savefig(outfile, dpi=dpi)
            plt.close(fig)

@quickook.command("quality")
@click.argument("input_files", nargs=-1)
@click.argument("output_path", nargs=1)
@click.option("--config", "-c", type=click.Path(dir_okay=False, exists=True),
              help="Config file - will merge and override the default config.")
@click.option("--dpi", type=int, nargs=1,
              default=300, show_default=True,
              help="DPI for output png-file.")
def ql_quality(input_files, output_path, config, dpi):
    config = _configure(config)
    mordor.utils.init_logger(config)

    with click.progressbar(input_files, label='Make daily quality quicklooks:') as files:
        for fn in files:
            ds_l1b = xr.load_dataset(fn)
            fig, axs = plt.subplots(3, 1, figsize=(10, 12), constrained_layout=True,
                                    gridspec_kw={"height_ratios": [2, 1, 2]})

            pl_status, (ax_s1, ax_s2) = ds_l1b.quicklooks.status(ax=axs[0])
            ds_l1b.quicklooks.quality_range_dhi2ghi(ax=axs[1], ratio=True, kwargs={'alpha': 0.5})
            ds_l1b.quicklooks.quality_range_shading(ax=axs[1], ratio=True, kwargs={'alpha': 0.5, 'hatch': '//'})
            ds_l1b.quicklooks.quality_range_lwd2temp(ax=axs[1], ratio=True, kwargs={'alpha': 0.5})

            axs[1].legend(bbox_to_anchor=(1, 1))

            pl_flags = ds_l1b.quicklooks.quality_flags(ax=axs[2], freq='15min')

            fname_info = parse.parse(
                config["fname_out"],
                os.path.basename(fn)
            ).named

            fname_info.update({
                "table": "quality",
                "sfx": "png"
            })

            outfile = os.path.join(output_path,
                                   "{dt:%Y/%m/}",
                                   config['fname_out'])
            outfile = outfile.format_map(fname_info)

            os.makedirs(os.path.dirname(outfile), exist_ok=True)
            fig.savefig(outfile, dpi=dpi)
            plt.close(fig)

###########################
# MORDOR info
###########################
@cli.command("info")
@click.argument("ids", nargs=-1)
@click.option("--serial",is_flag=True, help="Print only serial number.")
@click.option("--tropos",is_flag=True, help="Print only TROPOS ID.")
@click.option("--device",is_flag=True, help="Print only device information.")
@click.option("--calibration", is_flag=True, help="Print only calibration information. If 'ids' are not set and --calibration is set: Prints calibration due overview of all instruments.")
@click.option("--config", "-c", type=click.Path(dir_okay=False, exists=True),
              help="Config file - will merge and override the default config.")
def info(ids:str, calibration:bool, serial:bool, tropos:bool, device:bool, config: dict):
    """Print MORDOR device information. If no option is selected, full information is printed.
    """
    config = _configure(config)
    if len(ids) == 0:
        cdate = dt.datetime.now()
        mapping = mordor.utils.read_json(config["file_instrument_map"])
        ids, stations, ctimes = [],[],[]
        for id in mapping:
            res = mordor.utils.parse_calibration(
                config["file_calibration"],
                troposID=id,
                cdate=cdate,
            )
            if res is None:
                if calibration:
                    # skip if no calibration information
                    continue
                ctimes.append(np.datetime64("NaT"))
            else:
                ctimes.append(res.time.values)
            ids.append(id)
            stations.append(mapping[id]['station'])
        times = np.array(ctimes).astype("datetime64[M]") + np.timedelta64(24, 'M')

        if calibration:
            click.echo("Calibrations due:")
            isort = np.argsort(ctimes)
            for i in isort:
                timedelta = pd.to_datetime(times[i]) - cdate#.astype("datetime64[M]")
                text = f"{pd.to_datetime(times[i]):%Y-%m}: {ids[i]} - {stations[i]} - {mapping[ids[i]]['device']}"
                if timedelta < dt.timedelta(days=0):
                    click.echo(click.style(text, fg='red'))
                elif timedelta < dt.timedelta(days=30*6):#np.timedelta64(6, 'M'):
                    click.echo(click.style(text, fg='yellow'))
                else:
                    click.echo(text)
        else:
            isort = np.argsort(stations)
            tmpstation = ""
            for i in isort:
                if stations[i]!=tmpstation:
                    click.echo(stations[i])
                    tmpstation = stations[i]
                text = f"    {ids[i]}: {mapping[ids[i]]['device']}"
                if not np.isnat(times[i]):
                    texttime =f" -  calibration due: {pd.to_datetime(times[i]):%Y-%m}"
                    if times[i] < np.datetime64("now"):
                        texttime = click.style(texttime,fg='red')
                    text += texttime
                click.echo(text)
        return

    # if nothing is set, echo all
    if not (serial or tropos or device or calibration):
        serial, tropos, device, calibration = True, True, True, True

    for i, id in enumerate(ids):
        if i != 0:
            click.echo("")

        if id.startswith('A2'):
            meta = mordor.utils.meta_lookup(config, troposID=id)
        else:
            meta = mordor.utils.meta_lookup(config, serial=id)

        if tropos:
            click.echo(f"TROPOS ID: {meta['troposID']}")
        if serial:
            click.echo(f"Serial No: {meta['serial']}")
        if device:
            click.echo(f"Device:    {meta['station']} - {meta['device']}")
        if calibration and meta['calibration_date'] is not None:
            if "temperature_correction_coef" in meta:
                click.echo("Temperature correction coefficients:")
                click.echo(f"    {meta['temperature_correction_coef']}")
            click.echo("Calibrations:")
            for d, f, e in zip(meta['calibration_date'],
                               meta['calibration_factor'],
                               meta['calibration_error']):
                click.echo(f"    {d}: {f:6.3f} ({meta['calibration_factor_units']}) +- {e:4.2f} ({meta['calibration_error_units']})")
            nextd = pd.to_datetime(meta['calibration_date'][-1]) + dt.timedelta(days=366*2)
            click.echo(f"Calibration due: {nextd:%Y-%m}!")



@click.version_option()
@click.group("asi16")
def cli_asi16():
    pass


def asi16_missing_dates(
        images_path: str,
        processed_path: str,
        config: dict):
    img_files = []
    img_dates = []
    for p, d, f in os.walk(images_path):
        img_files += [os.path.join(p, fi) for fi in f if fi.endswith(".jpg")]
    for fn in img_files:
        finfo = parse.parse(config['asi16_out'], os.path.basename(fn)).named
        img_dates.append(finfo["dt"])
    img_dates = np.unique(img_dates)

    pro_files = []
    pro_dates = []
    for p, d, f in os.walk(processed_path):
        pro_files += [os.path.join(p, fi) for fi in f if fi.endswith(".bmp")]
    for fn in pro_files:
        finfo = parse.parse(config['asi16_out'], os.path.basename(fn)).named
        pro_dates.append(finfo["dt"])
    pro_dates = np.unique(pro_dates)

    missing_dates = np.setdiff1d(img_dates, pro_dates, assume_unique=True)
    return missing_dates

@cli_asi16.command("status")
@click.argument("images_path", nargs=1)
@click.argument("processed_path", nargs=1)
@click.option("--config", "-c", type=click.Path(dir_okay=False, exists=True),
              help="Config file - will merge and override the default config.")
def asi16_check_processed(
        images_path: str,
        processed_path: str,
        config: str):

    config = _configure(config)
    missing_dates = asi16_missing_dates(
        images_path=images_path,
        processed_path=processed_path,
        config=config
    )
    mdates, counts = np.unique(missing_dates.astype("datetime64[D]"), return_counts=True)

    print("Unprocessed cases:")
    print("date      , cases")
    for date, count in zip(mdates,counts):
        print(f"{date}, {count:4d}")

@cli_asi16.command("move2raw")
@click.argument("images_path", nargs=1)
@click.argument("processed_path", nargs=1)
@click.argument("raw_path", nargs=1)
@click.option("--shot",'-s',nargs=1,
                default=11,type=int,
                show_default=True,
                help="Shot configuration id.")
@click.option("--config", "-c", type=click.Path(dir_okay=False, exists=True),
              help="Config file - will merge and override the default config.")
def asi16_move_unprocessed(
        images_path: str,
        processed_path: str,
        raw_path: str,
        shot: int,
        config: str):
    config = _configure(config)
    missing_dates = asi16_missing_dates(
        images_path=images_path,
        processed_path=processed_path,
        config=config
    )

    img_files = []
    for p, d, f in os.walk(images_path):
        img_files += [os.path.join(p, fi) for fi in f if fi.endswith(".jpg")]
    for fn in img_files:
        finfo = parse.parse(config['asi16_out'], os.path.basename(fn)).named
        if (finfo["dt"] in missing_dates) and (finfo["shot"] == shot):
            path_raw = os.path.join(raw_path, f"{finfo['dt']:%Y%m%d/}")
            fname_raw = config['asi16_raw'].format(
                dt=finfo["dt"],
                shot=finfo["shot"]
            )
            os.makedirs(path_raw, exist_ok=True)
            os.rename(fn,os.path.join(path_raw, fname_raw))

    #
    #
    # for date in missing_dates:
    #     path_img = os.path.join(images_path, f"{date:%Y/%m/%d/}")
    #     fname_img = config['asi16_out'].format(
    #         dt=pd.to_datetime(date),
    #         campaign=config['campaign'],
    #         shot=shot,
    #         sfx='jpg'
    #     )
    #     path_raw = os.path.join(raw_path,f"{date:%Y%m%d/}")
    #     fname_raw = config['asi16_raw'].format(
    #         dt=pd.to_datetime(date),
    #         shot=shot
    #     )
    #     os.makedirs(path_raw, exist_ok=True)
    #     os.rename(
    #         os.path.join(path_img, fname_img),
    #         os.path.join(path_raw, fname_raw)
    #     )
