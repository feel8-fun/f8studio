from __future__ import annotations

import hashlib
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import urllib.parse
import urllib.request
import zipfile

from .errors import InvalidRequestError
from .extension_models import ExtensionImportRequest


MAX_ARCHIVE_BYTES = 2 * 1024**3
MAX_EXTRACTED_BYTES = 8 * 1024**3


def prepare_artifact(request: ExtensionImportRequest, root: Path) -> Path:
    if not re.fullmatch(r'[0-9a-f]{64}', request.sha256):
        raise InvalidRequestError('Extension SHA-256 must contain 64 lowercase hexadecimal characters')
    parsed = urllib.parse.urlsplit(request.url)
    if parsed.scheme != 'https' or not parsed.hostname or parsed.username or parsed.password:
        raise InvalidRequestError('Extension package URL must use HTTPS without embedded credentials')
    payload = root / 'payloads' / request.sha256
    if (payload / 'config/extensions.json').is_file():
        return payload
    cache = root / 'packages'
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / f'{request.sha256}.zip'
    if not archive.is_file():
        download = archive.with_suffix('.download')
        try:
            with urllib.request.urlopen(request.url, timeout=30) as response, download.open('wb') as destination:
                if urllib.parse.urlsplit(str(response.geturl())).scheme != 'https':
                    raise InvalidRequestError('Extension download redirected away from HTTPS')
                total = 0
                digest = hashlib.sha256()
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_ARCHIVE_BYTES:
                        raise InvalidRequestError('Extension archive exceeds the 2 GiB download limit')
                    destination.write(chunk)
                    digest.update(chunk)
            if digest.hexdigest() != request.sha256:
                raise InvalidRequestError('Extension archive SHA-256 does not match')
            download.replace(archive)
        finally:
            download.unlink(missing_ok=True)
    else:
        with archive.open('rb') as source:
            if hashlib.file_digest(source, 'sha256').hexdigest() != request.sha256:
                raise InvalidRequestError('Cached extension archive SHA-256 does not match')
    staging = payload.with_name(f'{request.sha256}.staging')
    staging.mkdir(parents=True, exist_ok=True)
    staging_root = staging.resolve()
    try:
        with zipfile.ZipFile(archive) as package:
            files = package.infolist()
            if len(files) > 20000 or sum(item.file_size for item in files) > MAX_EXTRACTED_BYTES:
                raise InvalidRequestError('Extension archive exceeds the extraction limit')
            names: set[str] = set()
            for item in files:
                relative = PurePosixPath(item.filename)
                if (relative.is_absolute() or '..' in relative.parts or '\\' in item.filename
                        or ':' in item.filename or not relative.parts or item.filename in names):
                    raise InvalidRequestError(f'Invalid extension archive path: {item.filename}')
                names.add(item.filename)
                mode = item.external_attr >> 16
                if stat.S_ISLNK(mode) or stat.S_IFMT(mode) not in {0, stat.S_IFREG, stat.S_IFDIR}:
                    raise InvalidRequestError(f'Unsupported extension archive file: {item.filename}')
                target = staging.joinpath(*relative.parts)
                if not target.resolve().is_relative_to(staging_root):
                    raise InvalidRequestError(f'Extension archive path escapes its root: {item.filename}')
                if item.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with package.open(item) as source, target.open('wb') as destination:
                    shutil.copyfileobj(source, destination)
                if mode & 0o111:
                    target.chmod(0o755)
        for name in ('config/extensions.json', 'config/service-index.json'):
            if not (staging / name).is_file():
                raise InvalidRequestError(f'Extension archive is missing {name}')
        staging.replace(payload)
    finally:
        if staging.is_dir():
            shutil.rmtree(staging)
    return payload
