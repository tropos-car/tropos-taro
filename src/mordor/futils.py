import os
import pandas as pd
import numpy as np
import logging

import mordor.utils


# logging setup
logging.basicConfig(
    filename='mordor.log',
    encoding='utf-8',
    level=logging.DEBUG,
    format='%(asctime)s %(name)s %(levelname)s:%(message)s'
)
logger = logging.getLogger(__name__)

def raw2daily(inpath, outpath, tables=None, config=None):
    config = mordor.utils.merge_config(config)

    # if None, process all available tables (from config)
    if tables is None:
        tables = config["logger_tables"]

    for table in tables:
        infname = os.path.join(inpath, config["fname_raw"].format(loggername=config["logger_name"], table=table))
        days = pd.read_csv(
            infname,
            skiprows=4,
            header=None,
            usecols=[0],
            sep=',',
            dtype='S10',
        ).values
        # find unique days and first index
        udays, idays = np.unique(days, return_index=True)
        udays = pd.to_datetime(udays.astype(str))
        # add header index
        idays += 4
        # append None for slicing -> slice(idays[i],idays[i+1])
        idays = list(idays) + [None]

        # Retrieve header from original file
        with open(infname,'r') as txt:
            datalines = txt.readlines()

        N = len(datalines)

        logging.debug(f"Start write {N-4} data lines from {infname}:")
        logging.debug(f"Period {datalines[4].split(',')[0]} to {datalines[-1].split(',')[0]} ")

        # split into daily files
        for i,uday in enumerate(udays):
            # Output filename
            outfname = os.path.join(
                outpath,
                config["path_sfx"],
                config["fname_out"]
            ).format(dt=uday, datalvl='l1a',sfx='dat',table=table,**config)
            # create directory structure
            os.makedirs(os.path.dirname(outfname), exist_ok=True)
            # write daily file
            with open(outfname,'w') as txt:
                # write header
                txt.writelines(datalines[:4])
                # write content
                islice = slice(idays[i],idays[i+1])
                txt.writelines(datalines[islice])
            logging.debug(f".. Written {len(datalines[islice])} data lines to {outfname}.")

        # remove lines from original file after writing
        with open(infname,'r') as txt:
            datalines_complete = txt.readlines()


        datalines_new = datalines_complete[:4]
        if len(datalines_complete) > N:
            datalines_new += datalines_complete[N:]

        txt = open(infname, 'w')
        txt.writelines(datalines_new)
        txt.close()
        logging.debug(f"Removed {N - 4} data lines from {infname}.")

