"""在整次评价的时限内等待 Windows 作业中的全部进程退出。"""

import time


def wait_for_job(job, process, deadline):
    """返回是否超时及最终作业计数；超时后确认进程树已终止。"""
    timed_out = False
    while job.accounting()["active_processes"]:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            job.terminate()
            timed_out = True
            break
        time.sleep(min(0.05, remaining))
    if timed_out:
        termination_deadline = time.monotonic() + 10
        while job.accounting()["active_processes"]:
            if time.monotonic() >= termination_deadline:
                raise RuntimeError("评价超时后进程树未能完全终止。")
            time.sleep(0.05)
    process.wait()
    return timed_out, job.accounting()
