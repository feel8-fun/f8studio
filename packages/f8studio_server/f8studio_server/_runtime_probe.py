"""Inspect an interpreter without importing extension code or third-party modules."""
from __future__ import annotations

from importlib import metadata
import json
import os
import platform
import sys


def main() -> None:
    distributions: dict[str, object] = {}
    for distribution in metadata.distributions():
        name = distribution.metadata.get('Name')
        if name:
            distributions.setdefault(name, {
                'version': distribution.version,
                'requires': distribution.requires or [],
                'extras': distribution.metadata.get_all('Provides-Extra') or [],
            })
    implementation = sys.implementation.version
    implementation_version = f'{implementation.major}.{implementation.minor}.{implementation.micro}'
    if implementation.releaselevel != 'final':
        implementation_version += f'{implementation.releaselevel[0]}{implementation.serial}'
    print(json.dumps({
        'pythonVersion': platform.python_version(),
        'markers': {
            'implementation_name': sys.implementation.name,
            'implementation_version': implementation_version,
            'os_name': os.name,
            'platform_machine': platform.machine(),
            'platform_release': platform.release(),
            'platform_system': platform.system(),
            'platform_version': platform.version(),
            'platform_python_implementation': platform.python_implementation(),
            'python_full_version': platform.python_version(),
            'python_version': f'{sys.version_info.major}.{sys.version_info.minor}',
            'sys_platform': sys.platform,
        },
        'distributions': distributions,
        'modules': sorted(set(metadata.packages_distributions()) | sys.stdlib_module_names),
    }))


if __name__ == '__main__':
    main()
