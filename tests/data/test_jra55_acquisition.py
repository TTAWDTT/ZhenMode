"""Published identities, complete catalogs and failed downloads stay distinct."""
import hashlib
import io
import json

import pytest

from zhenmode.execution.datasets import JRA_VARIABLES, fetch_jra, select_jra_files


def catalog(payload=b'ocean'):
    rows = [{'source_id': ['MRI-JRA55-do-1-4-0'], 'source_version': ['1.4.0'],
             'variable_id': [v], 'project': ['input4MIPs'], 'type': 'File',
             'version': 20190429, 'latest': True, 'deprecated': False, 'retracted': False,
             'title': f'{v}_gr_19580101-19581231.nc', 'size': len(payload),
             'checksum_type': ['SHA256'], 'checksum': [hashlib.sha256(payload).hexdigest()],
             'url': [f'https://example.org/{v}.nc|application/netcdf|HTTPServer'],
             'dataset_id': v} for v in JRA_VARIABLES]
    return {'response': {'numFound': len(rows), 'start': 0, 'docs': rows}}


@pytest.mark.parametrize('change', ['missing', 'ambiguous', 'version', 'checksum', 'unsafe_name'])
def test_unusable_catalog_is_rejected(change):
    value = catalog()
    row = value['response']['docs'][0]
    if change == 'missing':
        value['response']['numFound'] += 1
    elif change == 'ambiguous':
        value['response']['docs'].append(row | {'checksum': ['1' * 64]})
        value['response']['numFound'] += 1
    elif change == 'version':
        row['source_version'] = ['1.6.0']
    elif change == 'checksum':
        row['checksum_type'] = ['MD5']
    else:
        row['title'] = '../uas_gr_19580101-19581231.nc'
    with pytest.raises(ValueError):
        select_jra_files(value, 1958)


def test_download_cache_and_corruption(tmp_path):
    path = tmp_path / 'catalog.json'
    path.write_text(json.dumps(catalog()))

    class Response(io.BytesIO):
        def geturl(self):
            return 'https://example.org/file.nc'

    def opener(request, timeout):
        assert timeout == 30 and request.full_url.startswith('https://')
        return Response(b'ocean')

    destination = tmp_path / 'data'
    result = fetch_jra(path, destination, 1958, opener=opener)
    assert result['execution_status'] == 'completed'
    assert len(result['verified']) == 11
    assert not result['model_ready'] and not result['climate_qualification']
    result = fetch_jra(path, destination, 1958, opener=lambda *a, **kw: pytest.fail('cache fetch'))
    assert result['execution_status'] == 'completed'
    original = destination / result['files']['uas']['filename']
    original.write_bytes(b'wrong')
    with pytest.raises(ValueError, match='existing input'):
        fetch_jra(path, destination, 1958, opener=opener)
    assert original.read_bytes() == b'wrong'
    assert json.loads((destination / 'acquisition-1958.json').read_text())['execution_status'] == 'failed'
    attempts = list((destination / 'acquisition-attempts').glob('*.jsonl'))
    assert len(attempts) == 3
    terminal = [json.loads(p.read_text().splitlines()[-1])['receipt']['execution_status'] for p in attempts]
    assert sorted(terminal) == ['completed', 'completed', 'failed']


def test_failed_partial_retained_and_resume_requires_exact_range(tmp_path):
    path = tmp_path / 'catalog.json'
    path.write_text(json.dumps(catalog()))

    class Response(io.BytesIO):
        def geturl(self):
            return 'https://example.org/file.nc'

    with pytest.raises(ValueError, match='published identity'):
        fetch_jra(path, tmp_path / 'data', 1958, opener=lambda *a, **kw: Response(b'wrong'))
    partials = list((tmp_path / 'data').glob('*.partial'))
    assert len(partials) == 1 and partials[0].read_bytes() == b'wrong'
    # A completed but corrupt partial cannot be promoted or overwritten.
    with pytest.raises(ValueError, match='invalid size'):
        fetch_jra(path, tmp_path / 'data', 1958)


@pytest.mark.parametrize('total', ['5', '*'])
def test_interrupted_stream_records_failure_and_verified_resume(tmp_path, total):
    import http.client

    path = tmp_path / 'catalog.json'
    path.write_text(json.dumps(catalog()))
    destination = tmp_path / 'data'

    class Broken(io.BytesIO):
        def geturl(self):
            return 'https://example.org/file.nc'

        def read(self, n):
            raise http.client.IncompleteRead(b'oce')

    with pytest.raises(http.client.HTTPException):
        fetch_jra(path, destination, 1958, opener=lambda *a, **kw: Broken())
    assert json.loads((destination / 'acquisition-1958.json').read_text())['execution_status'] == 'failed'
    partial = next(destination.glob('*.partial'))
    partial.write_bytes(b'oce')

    class Response(io.BytesIO):
        status = 206
        headers = {'Content-Range': f'bytes 3-4/{total}'}

        def geturl(self):
            return 'https://example.org/file.nc'

    def opener(request, timeout):
        return Response(b'an' if request.get_header('Range') == 'bytes=3-' else b'ocean')

    result = fetch_jra(path, destination, 1958, opener=opener)
    assert result['execution_status'] == 'completed' and len(result['verified']) == 11
    attempts = list((destination / 'acquisition-attempts').glob('*.jsonl'))
    assert len(attempts) == 2
    assert any(json.loads(p.read_text().splitlines()[-1])['receipt']['execution_status'] == 'failed'
               for p in attempts)


def test_concurrent_writer_refused_before_touching_download_or_status(tmp_path):
    path = tmp_path / 'catalog.json'
    path.write_text(json.dumps(catalog()))
    destination = tmp_path / 'data'

    class Response(io.BytesIO):
        def geturl(self):
            return 'https://example.org/file.nc'

    def opener(*args, **kwargs):
        before = (destination / 'acquisition-1958.json').read_bytes()
        with pytest.raises(ValueError, match='writer lock'):
            fetch_jra(path, destination, 1958, opener=lambda *a, **kw: pytest.fail('second writer'))
        assert (destination / 'acquisition-1958.json').read_bytes() == before
        return Response(b'ocean')

    assert fetch_jra(path, destination, 1958, opener=opener)['execution_status'] == 'completed'
    assert not (destination / 'acquisition-1958.lock').exists()


def test_interrupt_after_final_byte_recovers_only_checksum_valid_partial(tmp_path):
    path = tmp_path / 'catalog.json'
    path.write_text(json.dumps(catalog()))
    destination = tmp_path / 'data'
    destination.mkdir()
    name = select_jra_files(catalog(), 1958)['uas']['filename']
    (destination / name).with_suffix('.nc.partial').write_bytes(b'ocean')

    class Response(io.BytesIO):
        def geturl(self):
            return 'https://example.org/file.nc'

    def opener(request, timeout):
        assert not request.full_url.endswith('/uas.nc')
        return Response(b'ocean')

    assert fetch_jra(path, destination, 1958, opener=opener)['execution_status'] == 'completed'
    assert (destination / name).read_bytes() == b'ocean'
