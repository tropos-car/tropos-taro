#https://www.freecodecamp.org/news/postgresql-in-python/

# info:   Access DB, extract cloud coverage data, write to netcdf.
# author: RHengst
# date:   2023-11-11

import os
import sys
import argparse
import configparser
import datetime
import logging
from trosat import cfconv as cf
import pandas as pd
import numpy as np
from netCDF4 import date2num
import psycopg2

try: 
    from tropos_uv import get_config
except:
    print("import local get_config")
    from mordor import get_config



"""
loop iterates the main function 
"""
def loop(args, config, logger,):
    logger.info('Start cloudiness processing   ')
    dates = pd.date_range(args.startdate.strftime("%Y\%m\%d"), args.stoppdate.strftime("%Y\%m\%d"), freq="1D", name=str, normalize=False)
    for step_startdate in dates:
        step_stoppdate = step_startdate + datetime.timedelta(days=1)

        #call function
        execute(args, config, logger, step_startdate, step_stoppdate)
    logger.info('End cloudiness processing     ')

"""
main fnction 
"""
def execute(args, config, logger, step_startdate, step_stoppdate):

    """Connection to DB"""
    conn = psycopg2.connect(\
           database=config.get("DATABASE","DB_NAME"),\
           host=config.get("DATABASE","DB_HOST"),\
           user=config.get("DATABASE","DB_USER"),\
           password=config.get("DATABASE","DB_PASS"),\
           port=config.get("DATABASE","DB_PORT")\
           )

    cursor = conn.cursor()
    
    "SQL Query"""
    sql="select cloudiness, humidity, temperature, timstmp from public.eval_numeric \
        WHERE timstmp >= '" + step_startdate.strftime("%Y-%m-%d") + "' \
        AND timstmp < '" + step_stoppdate.strftime("%Y-%m-%d") + "' \
        ORDER BY timstmp ASC;"

    cursor.execute(sql)
    
    """Get data from DB, store to datafame"""
    df = pd.DataFrame(cursor.fetchall())
    
    """Check data"""
    if df.empty:
        logger.info('No record in database at date : ' + step_startdate.strftime("%Y-%m-%d") )
        return None
    
    """Add header"""
    df.columns =[desc[0] for desc in cursor.description]

    """Load content of json_file to python variable cfjson"""
    cfjson=cf.read_cfjson(args.jsonfile)

    """Add dynamic global attributes"""
    cfjson["attributes"]["startdate"] = str(df.iloc[0]["timstmp"])
    cfjson["attributes"]["stoppdate"] = str(df.iloc[-1]["timstmp"])
    cfjson["attributes"]["created"]   = str(datetime.date.today())

    """Write dict data to numpy array https://www.quora.com/Is-it-possible-to-convert-a-Python-set-and-dictionary-to-a-NumPy-array """
    np_datetime = np.array(list(df["timstmp"]))

    """Prepare time variable to store seconds since ... in netCDF-File """
    second_since = date2num(np_datetime, cfjson['variables']['time']['attributes']['units'])
    logger.info('Processed date    : ' + step_startdate.strftime("%Y-%m-%d") )
    logger.info('Number of records : ' + str(len(second_since)) )
  

    """Create dictionary for mapping the data and concatenate the variables names and location """
    mapping_table={
          "cloudiness":{"df":"cloudiness"},
          "temperature":{"df":"temperature"},
          "humidity":{"df":"humidity"},
          "time":{"calculated":second_since}
          }

    """Introduce the size of the dimensions for each case """  
    cfjson.setDim('time', len(df["timstmp"]))
   
    """Add data to the variables """
    for netcdf_variable , mapping_table_value in mapping_table.items():
        for attribution , local_variable in mapping_table_value.items():
            if attribution == 'df':
                cfjson.setData(netcdf_variable,df[local_variable])
            elif attribution == 'calculated':
                cfjson.setData(netcdf_variable,local_variable)
   
   
    if not os.path.isdir( config.get('PATHFILE','PATHFILE_NETCDF') ):
        logger.warning('Output directory  not exists   : ' + config.get('PATHFILE','PATHFILE_NETCDF') )
        exit()

    netcdf_path_file = config.get('PATHFILE','PATHFILE_NETCDF') + step_startdate.strftime("%Y\%m\%d") + step_startdate.strftime("\%Y%m%d") + '_cloudcoverage_asi16_oscm.nc'

   
    if not os.path.isdir( os.path.dirname(netcdf_path_file) ):
        os.makedirs( os.path.dirname(netcdf_path_file) )
        logger.info('Create directory     : ' + os.path.dirname(netcdf_path_file) )


    """Create and save the netCDF-File """
    f = cf.create_file(netcdf_path_file , cfdict=cfjson)
    
    f.close()
    logger.info('File created : ' + netcdf_path_file )

####




"""
getting args, set logger, load configs
"""
def adjust(argv):
    startdate=datetime.date.today()

    """Get name of directory where main script is located"""
    current_dirname = os.path.dirname(os.path.realpath(__file__))
    
    """Get the name of the directory from where the script was executed"""
    exec_dirname = os.getcwd()
    
    """Define log_path_file + create dir"""
    log_path_file = exec_dirname + "\cloud_coverage.log"
    json_file  = os.path.dirname(os.path.realpath(__file__))  + '\conf\cloudiness_js_meta.json.template'

    """for calling the function from the terminal"""
    parser = argparse.ArgumentParser(description='Access DB, extract cloud coverage data, write to daily netcdf file(s).') 
    parser.add_argument('-s', required=False, type=str, dest='id', default=startdate.strftime("%Y%m%d"),
                    help='processing start date as 20190107 (y:2019 m:01 d:07)')
    parser.add_argument('-e', required=False, type=str, dest='fd', default=None,
                    help='processing end date as 20190412 (y:2019 m:04 d:12)')
    parser.add_argument('--configfile', required=True, type=str, dest='your_config_file',
                    help='config path and file name')

    parser.add_argument('--logfile', default=log_path_file, dest='logfile',
                    help="define logfile (default: " +  log_path_file + ") ")
    parser.add_argument('--loglevel', default='INFO', dest='loglevel',
                    help="define loglevel to output screen INFO (default) | WARNING | ERROR ")
    parser.add_argument('--jsonfile', default=json_file, dest='jsonfile',
                    help="define jsonfile (default: " + json_file + ") ")
    args = parser.parse_args()

    """Create directory to store logfile if necessary"""
    log_path_file = args.logfile
    if not os.path.isdir(  os.path.dirname( log_path_file ) ):
        os.makedirs( os.path.dirname( log_path_file ) )
        print('Create directory     : ' + os.path.dirname(log_path_file) )
    
    """Create logger with 'Cloudiness"""
    logger = logging.getLogger('cloudiness')
    logger.setLevel(logging.DEBUG)
    
    """Create file handler which logs even debug messages"""
    fh = logging.FileHandler( log_path_file )
    fh.setLevel(logging.DEBUG)

    """Create/check level"""
    screen_level = logging.getLevelName(args.loglevel)
    
    """Create console handler with a higher log level"""
    ch = logging.StreamHandler()
    #ch.setLevel(logging.WARNING)
    ch.setLevel(screen_level)
    
    """Create formatter and add it to the handlers"""
    formatter = logging.Formatter(fmt='%(asctime)s | %(name)-24s | %(levelname)-8s | %(message)s | %(module)s (%(lineno)d)', datefmt='%Y-%m-%d %H:%M:%S',)
    
    """Add formatter to the handlers"""
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    """Add the handlers to the logger"""
    logger.addHandler(fh)
    logger.addHandler(ch)

    """Add first log mesage"""
    logger.debug('Init logger')
    

    args.startdate = datetime.datetime.strptime(args.id, '%Y%m%d')
    if args.fd:
        args.stoppdate = datetime.datetime.strptime(args.fd, '%Y%m%d')
    else:
        args.stoppdate = args.startdate

    """Check if configs exists"""
    default_config_file = os.path.dirname(os.path.realpath(__file__))  + '\conf\\config.ini.template'
    your_config_file    = args.your_config_file

    if not os.path.isfile( default_config_file ):
        logger.warning('default config file not exists: ' + default_config_file)
        quit()
    if not os.path.isfile( your_config_file ):
        logger.warning('local config file not exists: ' + your_config_file)
        quit()

    """Get summarised config"""
    config = get_config.main(default_config_file, your_config_file)

    if not os.path.isfile( args.jsonfile ):
        logger.error( 'File json not exists: '+ args.jsonfile )
        quit()

    """Call loop function """
    loop(args, config, logger)

if __name__ == "__main__":
  adjust(sys.argv[1:])

