#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
The module read and compares two ini config files and returns a merged config object (configparser)

The tasks of the module:
* read config1, config2
* set config1 as source/final config
* if value at existing keys in config2 differs from config1, update final config
* if config2 introduce new key (and value) skip it, because only config1 keys are allowed

"""

__author__ = 'Rico Hengst'
__created__ = '17.04.21'

import configparser
import os
import logging
import copy
import datetime



""" Create logger, name important """
module_logger = logging.getLogger('uv-processing.config')


""" Function to read ini config file """
def get_config_object(config_object ) -> configparser.ConfigParser:
    if isinstance(config_object, configparser.ConfigParser):
        module_logger.info("config object is already a configparser.ConfigParser")
        return config_object
        
    if isinstance(config_object, str):
        if not os.path.isfile( config_object ):
            module_logger.error("config_object is a string, so a filename is expected, but filename not exists '" + config_object + "'")
            quit()
        else:
            # parse file
            module_logger.info("Read config file '" + config_object + "'")
            #config = configparser.ConfigParser(  ) # magic interpolation !
            config = configparser.RawConfigParser()
            config.read(config_object)
            return config
            



def main(c1,c2):
    
    """ Get configs """
    config1 = get_config_object(c1)
    config2 = get_config_object(c2)
    
    """ Set copy config """
    config_final = copy.deepcopy( config1 )
    
    # iteration via all sections in config1
    for section in config1.sections():
        # iteration via all keys in section
        for key1 in config1[section]:
            # check if key in config1section exists as key in config2section
            if key1 in config2[section]:
                # update config_final
                if config1[section][key1] != config2[section][key1]:
                    #config_final[section][key1] = config2[section][key1]
                    config_final.set(section, key1, config2[section][key1] )
                    #config_final.set(section, key1, eval( config2[section][key1] )  )
                    
                    module_logger.info("Update config section.key '" + section + "'.'" + key1 + "':  '" + config1[section][key1] + "' by '" + config2[section][key1] + "'")
            else:
                module_logger.info("Section.key is missing in config2 '" + section + "'.'" + key1 +"', but is not required")
        
        for key2 in config2[section]:
            # check if key in config2section exists as key in config1section
            if key2 not in config1[section]:
                module_logger.warning("Skip key value pair cause section key not exists in config1 '" + section + "'.'" + key2 +"'")
                
    # return
    return config_final
    
    
    
if __name__ == "__main__":
    print("run another_script.py as a script...")
    main('example1.ini', 'example2.ini')  # calls the main function



#config_final.write(sys.stdout)



    
