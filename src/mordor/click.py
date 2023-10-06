import os.path

import click
import logging

from . import utils as mordorutils
from . import futils as mordorfutils


# logging setup
logging.basicConfig(
    filename='mordor.log',
    encoding='utf-8',
    level=logging.DEBUG,
    format='%(asctime)s %(name)s %(levelname)s:%(message)s'
)
logger = logging.getLogger(__name__)

@click.group("mordor")
def cli():
    pass


@click.command("raw2daily")
@click.argument("input_path", nargs=1)
@click.argument("output_path", nargs=1)
@click.option("--config","-c",
              nargs=1,
              help="Specify config file - will merge and override the default config.")
def raw2daily(input_path: str,
              output_path: str,
              config: str):
    """
    Moves data lines from the  continuous Loggernet files within input path (table names specified in config) to daily files in output path.
    The output path is suffixed according to config['path_sfx'] (default is '{dt:%Y/%m/}').
    The daily output files are named according to config['fname_out'] (default is 'mordor_{dt:%Y-%m-%d}_{campaign}_{table}_{datalvl}.c{collection:02d}.{sfx}').
    """
    config =  mordorutils.read_json(os.path.abspath(config))
    mordorfutils.raw2daily(
       inpath=os.path.abspath(input_path),
       outpath=os.path.abspath(output_path),
       config=config
    )


cli.add_command(raw2daily)


