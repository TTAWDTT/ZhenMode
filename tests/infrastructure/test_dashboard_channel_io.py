"""Offline PTY behavior controls; never open credentials or an SSH connection."""

import importlib.util
import inspect
import io
import sys
import types
from pathlib import Path

import pytest

from tests.support.paths import REPOSITORY_ROOT

ROOT = REPOSITORY_ROOT


class OutputCapture(io.StringIO):
    def reconfigure(self, **kwargs):
        pass


def load_bridge(relative, monkeypatch):
    """Load public bridge APIs with denied SSH and synthetic secret contents."""
    dependency = types.ModuleType('paramiko')

    def deny_connection(*args, **kwargs):
        raise AssertionError('test must never connect to SSH')

    dependency.SSHClient = deny_connection
    actual_open = io.open

    def controlled_open(path, *args, **kwargs):
        if Path(path).name == 'secret.json':
            return io.StringIO('{}')
        return actual_open(path, *args, **kwargs)

    with monkeypatch.context() as context:
        context.setitem(sys.modules, 'paramiko', dependency)
        context.setattr(sys, 'stdout', OutputCapture())
        context.setattr(sys, 'path', list(sys.path))
        context.setattr(io, 'open', controlled_open)
        spec = importlib.util.spec_from_file_location('offline_' + Path(relative).stem, ROOT / relative)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class Clock:
    def __init__(self):
        self.current = 0.
        self.sleeps = []

    def time(self):
        return self.current

    def sleep(self, duration):
        self.sleeps.append(duration)
        self.current += duration


class Channel:
    def __init__(self, clock, arrivals=()):
        self.clock = clock
        self.arrivals = list(arrivals)
        self.receives = []
        self.ready_calls = 0

    def recv_ready(self):
        self.ready_calls += 1
        return bool(self.arrivals and self.arrivals[0][0] <= self.clock.current)

    def recv(self, size):
        self.receives.append(size)
        return self.arrivals.pop(0)[1]


@pytest.fixture(params=['dashboard/cluster.py', 'scripts/publish_dashboard.py'])
def implementation(request, monkeypatch):
    return load_bridge(request.param, monkeypatch).drain


def run(implementation, monkeypatch, *, arrivals=(), timeout=4.):
    clock = Clock()
    channel = Channel(clock, arrivals)
    monkeypatch.setitem(implementation.__globals__, 'time', clock)
    output = implementation(channel, timeout)
    return output, channel, clock


def test_empty_channel_uses_original_timeout_and_poll_interval(implementation, monkeypatch):
    output, channel, clock = run(implementation, monkeypatch, timeout=.3)
    assert output == b'' and channel.receives == []
    assert clock.current == pytest.approx(.3)
    assert clock.sleeps == [.15, .15]


def test_burst_preserves_raw_bytes_and_resets_idle_deadline(implementation, monkeypatch):
    output, channel, clock = run(implementation, monkeypatch, arrivals=[(0., b'a'), (0., b'\xff\x00')])
    assert output == b'a\xff\x00'
    assert channel.receives == [65536, 65536]
    assert clock.current == pytest.approx(1.05)
    assert all(delay == .15 for delay in clock.sleeps)


def test_delayed_chunks_extend_deadline_from_last_receive(implementation, monkeypatch):
    output, channel, clock = run(implementation, monkeypatch, arrivals=[(.3, b'first'), (.9, b'last')])
    assert output == b'firstlast'
    assert channel.receives == [65536, 65536]
    assert clock.current == pytest.approx(1.95)


@pytest.mark.parametrize('timeout', [0., -1.])
def test_nonpositive_timeout_does_not_touch_channel(implementation, monkeypatch, timeout):
    output, channel, clock = run(implementation, monkeypatch, arrivals=[(0., b'unread')], timeout=timeout)
    assert output == b'' and channel.ready_calls == 0 and clock.sleeps == []


def test_channel_error_propagates(implementation, monkeypatch):
    clock = Clock()
    channel = Channel(clock)

    def failure():
        raise RuntimeError('closed channel')

    channel.recv_ready = failure
    monkeypatch.setitem(implementation.__globals__, 'time', clock)
    with pytest.raises(RuntimeError, match='closed channel'):
        implementation(channel)


def test_public_signature_and_distinct_ascii_policies(implementation, monkeypatch):
    signature = inspect.signature(implementation)
    assert list(signature.parameters) == ['sh', 'timeout']
    assert signature.parameters['timeout'].default == 4.
    cluster = load_bridge('dashboard/cluster.py', monkeypatch)
    publisher = load_bridge('scripts/publish_dashboard.py', monkeypatch)
    assert cluster.to_ascii(b'a\x00b') == 'ab'
    assert publisher.to_ascii(b'a\x00b') == 'a.b'


def test_shared_implementation_preserves_both_public_aliases(monkeypatch):
    cluster = load_bridge('dashboard/cluster.py', monkeypatch)
    publisher = load_bridge('scripts/publish_dashboard.py', monkeypatch)
    assert cluster.drain is publisher.drain
