"""Contracts for the assembled workspace, validated by the root CI."""
from pathlib import Path
import msgspec
import yaml
from f8platform.extension_models import ExtensionCatalog
from f8pysdk.service_runtime_tools.inventory.index import read_service_index


def test_official_extension_environment_names_match_all_platform_service_launches() -> None:
    config = Path(__file__).resolve().parents[1] / 'build/workspace/config'
    extensions = msgspec.json.decode((config / 'extensions.json').read_bytes(), type=ExtensionCatalog)
    index = read_service_index(config / 'service-index.json')
    services = {item.serviceClass: item for item in index.services}
    for manifest in extensions.extensions:
        if manifest.runtime.kind == 'workspace':
            for name in manifest.service_classes:
                for relative in services[name].manifests.values():
                    launch = yaml.safe_load(Path(relative.replace('${F8_PACKAGE_ROOT}', str(config.parents[2]))).read_text())['launch']
                    assert launch['command'] == 'pixi'
                    assert launch['args'][:3] == ['run', '-e', manifest.runtime.environment], name


def test_repository_catalog_owns_every_registered_service() -> None:
    root = Path(__file__).resolve().parents[1] / 'build/workspace'
    catalog = msgspec.json.decode((root / 'config/extensions.json').read_bytes(), type=ExtensionCatalog)
    index = read_service_index(root / 'config/service-index.json')
    classes = [name for extension in catalog.extensions for name in extension.service_classes]
    assert len(classes) == len(set(classes))
    assert set(classes) == {item.serviceClass for item in index.services}

