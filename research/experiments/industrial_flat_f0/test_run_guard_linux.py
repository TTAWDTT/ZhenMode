"""Small launcher checks only; no ocean model or benchmark is run."""
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch


path = Path(__file__).with_name('run_guard.py')
spec = importlib.util.spec_from_file_location('flat_guard', path)
guard = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard)


class GuardTests(unittest.TestCase):
    def test_new_session_descendant_is_counted(self):
        with tempfile.TemporaryDirectory(prefix='guard-test-', dir=Path.home()) as root:
            child = 'import time; data=bytearray(16*1024**2); time.sleep(.35)'
            code = ('import subprocess,sys; child=subprocess.Popen('
                    '[sys.executable,"-c",' + repr(child) + '],start_new_session=True);'
                    'child.wait()')
            result = guard.guarded_run([sys.executable, '-c', code], root, wall_s=5)
            self.assertEqual(result['exit_code'], 0)
            self.assertIsNone(result['stopped_reason'])
            self.assertGreaterEqual(result['descendants_identified'], 2)
            self.assertGreater(result['sampled_aggregate_peak_rss_bytes'], 20 * 1024**2)
            self.assertTrue(result['descendant_cleanup_confirmed'])

    def test_aggregate_output_stops_multiple_small_files(self):
        with tempfile.TemporaryDirectory(prefix='guard-test-', dir=Path.home()) as root:
            code = ('import pathlib,time;'
                    '[pathlib.Path(str(i)).write_bytes(b"x"*8192) for i in range(12)];'
                    'time.sleep(3)')
            result = guard.guarded_run([sys.executable, '-c', code], root,
                                       wall_s=5, output_limit=65536)
            self.assertEqual(result['stopped_reason'], 'aggregate output budget')

    def test_wall_timeout_stops_only_identified_run(self):
        with tempfile.TemporaryDirectory(prefix='guard-test-', dir=Path.home()) as root:
            result = guard.guarded_run([sys.executable, '-c', 'import time;time.sleep(3)'],
                                       root, wall_s=.25)
            self.assertEqual(result['stopped_reason'], 'wall timeout')
            self.assertNotEqual(result['exit_code'], 0)

    def test_unreadable_environment_still_cleans_direct_child(self):
        with tempfile.TemporaryDirectory(prefix='guard-test-', dir=Path.home()) as root:
            code = ('import ctypes,time; ctypes.CDLL(None).prctl(4,0,0,0,0);'
                    'time.sleep(3)')
            result = guard.guarded_run([sys.executable, '-c', code], root, wall_s=5)
            self.assertIn('monitoring failure', result['stopped_reason'])
            self.assertTrue(result['descendant_cleanup_confirmed'])

    def test_shared_output_budget_counts_the_other_model(self):
        with tempfile.TemporaryDirectory(prefix='guard-test-', dir=Path.home()) as root:
            run = Path(root) / 'second-model'
            run.mkdir()
            (Path(root) / 'first-model-output').write_bytes(b'x' * 32768)
            code = ('import pathlib,time;'
                    '[pathlib.Path(str(i)).write_bytes(b"x"*8192) for i in range(5)];'
                    'time.sleep(3)')
            result = guard.guarded_run([sys.executable, '-c', code], run,
                                       wall_s=5, output_limit=65536,
                                       output_budget_directory=root)
            self.assertEqual(result['stopped_reason'], 'aggregate output budget')

    def test_cleanup_scan_failure_still_reaps_child_and_reports_uncertainty(self):
        original = guard.user_processes

        def fail_cleanup(uid, *, require_environment=True):
            if not require_environment:
                raise PermissionError('injected cleanup read failure')
            return original(uid)

        with tempfile.TemporaryDirectory(prefix='guard-test-', dir=Path.home()) as root:
            with patch.object(guard, 'user_processes', side_effect=fail_cleanup):
                result = guard.guarded_run(
                    [sys.executable, '-c', 'import time;time.sleep(3)'], root, wall_s=.25)
            self.assertEqual(result['stopped_reason'], 'wall timeout')
            self.assertIsNotNone(result['exit_code'])
            self.assertFalse(result['descendant_cleanup_confirmed'])


if __name__ == '__main__':
    unittest.main()
