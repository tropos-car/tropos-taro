import os.path

import click
import logging
import parse
import pandas as pd
import numpy as np
import datetime as dt
import importlib.resources

import mordor
import mordor.utils
import mordor.futils
import mordor.data


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
                config["fname_out"].replace("%Y-%m-%d", "ti"),
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
            click.echo("Calibrations:")
            for d, f, e in zip(meta['calibration_date'],
                               meta['calibration_factor'],
                               meta['calibration_error']):
                click.echo(f"    {d}: {f:6.3f} ({meta['calibration_factor_units']}) +- {e:4.2f} ({meta['calibration_error_units']})")
            nextd = pd.to_datetime(meta['calibration_date'][-1]) + dt.timedelta(days=366*2)
            click.echo(f"Calibration due: {nextd:%Y-%m}!")
