import os
import logging
import importlib.resources
from toolz import keyfilter
import jstyleson as json
from addict import Dict as adict
from operator import itemgetter

logger = logging.getLogger(__name__)

def read_json(fpath: str, *, object_hook: type = adict, cls = None) -> dict:
    """ Parse json file to python dict.
    """
    with open(fpath,"r") as f:
        js = json.load(f, object_hook=object_hook, cls=cls)
    return js

def pick(whitelist: list[str], d: dict) -> dict:
    """ Keep only whitelisted keys from input dict.
    """
    return keyfilter(lambda k: k in whitelist, d)

def omit(blacklist: list[str], d: dict) -> dict:
    """ Omit blacklisted keys from input dict.
    """
    return keyfilter(lambda k: k not in blacklist, d)

def get_var_attrs(d: dict) -> dict:
    """
    Parse cf-compliance dictionary.

    Parameters
    ----------
    d: dict
        Dict parsed from cf-meta json.

    Returns
    -------
    dict
        Dict with netcdf attributes for each variable.
    """
    get_vars = itemgetter("variables")
    get_attrs = itemgetter("attributes")
    vattrs = {k: get_attrs(v) for k,v in get_vars(d).items()}
    for k,v in get_vars(d).items():
        vattrs[k].update({
            "dtype": v["type"],
            "gzip":True,
            "complevel":6
        })
    return vattrs

def get_attrs_enc(d : dict) -> (dict,dict):
    """ Split variable attributes in attributes and encoding-attributes.
    """
    _enc_attrs = {
        "scale_factor",
        "add_offset",
        "_FillValue",
        "dtype",
        "zlib",
        "gzip",
        "complevel",
        "calendar",
    }
    # extract variable attributes
    vattrs = {k: omit(_enc_attrs, v) for k, v in d.items()}
    # extract variable encoding
    vencode = {k: pick(_enc_attrs, v) for k, v in d.items()}
    return vattrs, vencode

def get_default_config():
    """
    Get MORDOR default config
    """
    fn_config = os.path.join(
        importlib.resources.files("mordor"),
        "conf/mordor_config.json"
    )
    default_config = read_json(fn_config)

    # expand default file paths
    for key in default_config:
        if key.startswith("file"):
            default_config.update({
                key: os.path.join(
                    importlib.resources.files("mordor"),
                    default_config[key]
                )
            })
    return default_config

def merge_config(config):
    """
    Merge config dictionary with MORDOR default config
    """
    default_config = get_default_config()
    if config is None:
        config = default_config
    else:
        config = {**default_config, **config}
    return config


def init_logger(config):
    """
    Initialize Logging based on MORDOR config
    """
    config = merge_config(config)
    fname = os.path.abspath(config["file_log"])

    # logging setup
    logging.basicConfig(
        filename=fname,
        encoding='utf-8',
        level=logging.DEBUG,
        format='%(asctime)s %(name)s %(levelname)s:%(message)s'
    )
