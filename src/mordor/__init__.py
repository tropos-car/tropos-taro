
from . import _version
__version__ = _version.get_versions()['version']

import logging
logging.getLogger(__name__).addHandler(logging.NullHandler())