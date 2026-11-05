import os
import subprocess
import sys
import time
import unittest

from scripts.p0_windows_job import WindowsJob
from scripts.p1_job_supervision import wait_for_job


@unittest.skipUnless(os.name == "nt", "Windows 作业对象测试")
class TestJobSupervision(unittest.TestCase):
    @staticmethod
    def start(job, child_seconds):
        child = f"import time; time.sleep({child_seconds})"
        parent = f"import subprocess, sys; subprocess.Popen([sys.executable, '-c', {child!r}])"
        return job.start([sys.executable, "-B", "-c", parent],
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    def test_waits_for_child_after_parent_exits(self):
        with WindowsJob() as job:
            process = self.start(job, 0.4)
            begin = time.monotonic()
            timed_out, accounting = wait_for_job(job, process, begin + 5)
        self.assertFalse(timed_out)
        self.assertGreaterEqual(time.monotonic() - begin, 0.3)
        self.assertEqual(process.returncode, 0)
        self.assertEqual(accounting["active_processes"], 0)
        self.assertGreaterEqual(accounting["total_processes"], 2)

    def test_timeout_terminates_remaining_child(self):
        with WindowsJob() as job:
            process = self.start(job, 5)
            timed_out, accounting = wait_for_job(job, process, time.monotonic() + 0.4)
        self.assertTrue(timed_out)
        self.assertEqual(accounting["active_processes"], 0)


if __name__ == "__main__":
    unittest.main()
