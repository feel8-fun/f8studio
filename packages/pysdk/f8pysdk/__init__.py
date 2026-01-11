from .generated import *
from .schema_helpers import *

def _proxy_getattr(self, name):
    return getattr(self.root, name)

F8DataTypeSchema.__getattr__ = _proxy_getattr
F8JsonValue.__getattr__ = _proxy_getattr