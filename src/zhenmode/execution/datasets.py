"""Retrieve checksum-pinned OMIP forcing without relabelling another release."""
from __future__ import annotations

import hashlib
import http.client
import json
import os
import shutil
import urllib.request
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from zhenmode.provenance.sources import load_json, sha256_file

JRA_VARIABLES = ('uas', 'vas', 'tas', 'huss', 'psl', 'rsds', 'rlds',
                 'prra', 'prsn', 'friver', 'licalvf')


def select_jra_files(catalog, year):
    """Select actual ESGF file records; reject ambiguous versions/replicas."""
    if type(year) is not int or not 1958 <= year <= 2018:
        raise ValueError('year must be an integer in the frozen 1958-2018 interval')
    response = catalog['response']
    if len(response['docs']) != response['numFound'] or response.get('start', 0) != 0:
        raise ValueError('input catalog is incomplete; fetch all file records first')
    selected = {}
    for variable in JRA_VARIABLES:
        rows = [r for r in response['docs']
                if r.get('source_id') == ['MRI-JRA55-do-1-4-0']
                and r.get('variable_id') == [variable]
                and r.get('project') == ['input4MIPs'] and r.get('type') == 'File'
                and r.get('source_version') == ['1.4.0']
                and r.get('version') == 20190429 and r.get('latest') is True
                and r.get('deprecated') is False and r.get('retracted') is False
                and f'_gr_{year}0101' in r.get('title', '')]
        if not rows or any(r.get('checksum_type') != ['SHA256'] for r in rows):
            raise ValueError(f'missing published SHA256 identity for {variable}/{year}')
        identities = {(r['title'], r['size'], tuple(r['checksum'])) for r in rows}
        if len(identities) != 1:
            raise ValueError(f'ambiguous published identities for {variable}/{year}')
        name, size, checksums = identities.pop()
        if Path(name).name != name or type(size) is not int or size <= 0:
            raise ValueError('invalid published filename/size')
        if len(checksums) != 1 or len(checksums[0]) != 64 or any(c not in '0123456789abcdef' for c in checksums[0]):
            raise ValueError('invalid published SHA256')
        urls = sorted({entry.split('|')[0] for r in rows for entry in r['url']
                       if entry.endswith('|HTTPServer') and entry.startswith('https://')})
        if not urls:
            raise ValueError(f'no published HTTPS download for {variable}/{year}')
        selected[variable] = {'filename': name, 'bytes': size, 'sha256': checksums[0],
                              'source_url': urls[0], 'dataset_id': rows[0]['dataset_id']}
    return selected


@contextmanager
def _acquisition_lock(destination, year):
    """Exclusive writer; a crashed lock needs inspection, never silent takeover."""
    lock = destination / f'acquisition-{year}.lock'
    owner = json.dumps({'attempt': uuid.uuid4().hex, 'pid': os.getpid()})
    try:
        with lock.open('x', encoding='utf-8') as stream:
            stream.write(owner)
    except FileExistsError as error:
        raise ValueError(f'acquisition writer lock exists: {lock}; inspect the owning process before retrying') from error
    try:
        yield
    finally:
        if lock.read_text(encoding='utf-8') == owner:
            lock.unlink()


def fetch_jra(catalog_path, destination, year, *, opener=urllib.request.urlopen):
    """Stream one verified file at a time; resume HTTP ranges and retain receipts.

    This downloads original annual files, not model-ready remapped inputs.
    Successful IO never establishes mechanism or climate qualification.
    """
    select_jra_files(load_json(catalog_path), year)
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    with _acquisition_lock(destination, year):
        return _fetch_jra(catalog_path, destination, year, opener=opener)


def _fetch_jra(catalog_path, destination, year, *, opener):
    files = select_jra_files(load_json(catalog_path), year)
    receipt_path = destination / f'acquisition-{year}.json'
    previous = None
    if receipt_path.exists():
        previous = load_json(receipt_path)
        if previous['catalog_sha256'] != sha256_file(catalog_path) or previous['files'] != files:
            raise ValueError('destination acquisition identity differs from this catalog')
    required = sum(info['bytes'] for info in files.values()
                   if not (destination / info['filename']).exists())
    if shutil.disk_usage(destination).free < required + 2 * 1024**3:
        raise ValueError('insufficient free disk space for originals plus 2 GiB reserve')
    receipt = {'schema_version': 1, 'product': 'JRA55-do', 'version': '1.4.0',
               'catalog_sha256': sha256_file(catalog_path), 'year': year, 'files': files,
               'execution_status': 'running', 'verified': {}, 'failed': {},
               'model_ready': False, 'climate_qualification': False}
    receipt['attempt_id'] = uuid.uuid4().hex
    history = destination / 'acquisition-attempts'
    history.mkdir(exist_ok=True)
    journal = history / f'{year}-{receipt["attempt_id"]}.jsonl'
    # The current status is a replaceable index. Its attempt journal is
    # append-only: retrying must not erase an earlier failure or its evidence.
    if previous is not None:
        with journal.open('x', encoding='utf-8') as stream:
            stream.write(json.dumps({'event': 'previous_status', 'receipt': previous}, allow_nan=False) + '\n')

    def record():
        receipt['recorded_at_utc'] = datetime.now(timezone.utc).isoformat()
        with journal.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps({'event': 'status', 'receipt': receipt}, allow_nan=False) + '\n')
        temporary = receipt_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n', encoding='utf-8')
        temporary.replace(receipt_path)

    record()
    try:
        for variable, info in files.items():
            target = destination / info['filename']
            if target.exists():
                if target.stat().st_size != info['bytes'] or sha256_file(target) != info['sha256']:
                    raise ValueError(f'existing input fails published identity: {target.name}')
            else:
                partial = target.with_suffix('.nc.partial')
                size = partial.stat().st_size if partial.exists() else 0
                if size == info['bytes'] and sha256_file(partial) == info['sha256']:
                    partial.replace(target)
                    receipt['verified'][variable] = info['sha256']
                    record()
                    continue
                if size >= info['bytes']:
                    raise ValueError(f'previous partial has invalid size: {partial.name}')
                headers = {'User-Agent': 'ZhenMode-input-acquisition/1'}
                if size:
                    headers['Range'] = f'bytes={size}-'
                request = urllib.request.Request(info['source_url'], headers=headers)
                checksum = hashlib.sha256()
                if size:
                    with partial.open('rb') as existing:
                        while chunk := existing.read(1024**2):
                            checksum.update(chunk)
                with opener(request, timeout=30) as response, partial.open('ab' if partial.exists() else 'xb') as stream:
                    if urlsplit(response.geturl()).scheme != 'https':
                        raise ValueError('download redirected outside HTTPS')
                    if size:
                        prefix = f'bytes {size}-{info["bytes"]-1}/'
                        allowed = {prefix + str(info['bytes']), prefix + '*'}
                        if response.status != 206 or response.headers.get('Content-Range') not in allowed:
                            raise ValueError('server did not honor the exact published resume range')
                    while chunk := response.read(1024**2):
                        size += len(chunk)
                        if size > info['bytes']:
                            raise ValueError('download exceeds published size')
                        checksum.update(chunk)
                        stream.write(chunk)
                        if size % (32 * 1024**2) < 1024**2:
                            receipt['progress'] = {'variable': variable, 'bytes': size}
                            record()
                if size != info['bytes']:
                    raise ValueError(f'download incomplete: {target.name}: {size}/{info["bytes"]} bytes; partial retained')
                if checksum.hexdigest() != info['sha256']:
                    raise ValueError(f'download fails published identity: {target.name}')
                partial.replace(target)
            receipt['verified'][variable] = info['sha256']
            record()
        receipt['execution_status'] = 'completed'
    except (OSError, ValueError, http.client.HTTPException, KeyboardInterrupt) as error:
        receipt['execution_status'] = 'failed'
        receipt['failed'][variable] = str(error) or type(error).__name__
        partial = (destination / files[variable]['filename']).with_suffix('.nc.partial')
        if partial.exists():
            receipt['progress'] = {'variable': variable, 'bytes': partial.stat().st_size}
        record()
        raise
    record()
    return receipt
