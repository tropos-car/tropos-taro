import os
import pandas as pd
import numpy as np

import mordor.utils

def raw2daily(pf, tables=None, config=None):
    config = mordor.utils.merge_config(config)

    # if None, process all available tables (from config)
    if tables is None:
        tables = config["logger_tables"]

    for table in tables:
        days = pd.read_csv(
            os.path.join(pf, config["fname_raw"].format(loggername=config["logger_name"], table=table)),
            skiprows=4,
            header=None,
            usecols=[0],
            sep=',',
            dtype='S10',
        ).values

        udays, idays = np.unique(days, return_index=True)

        # create directory structure

        # split into daylie files