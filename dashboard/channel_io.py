"""Raw-byte PTY draining for the dashboard bridge and snapshot publisher."""
import time


def drain(sh, timeout=4.0):
    buf = b""
    end = time.time() + timeout
    while time.time() < end:
        if sh.recv_ready():
            buf += sh.recv(65536)
            end = time.time() + 1.0
        else:
            time.sleep(0.15)
    return buf
