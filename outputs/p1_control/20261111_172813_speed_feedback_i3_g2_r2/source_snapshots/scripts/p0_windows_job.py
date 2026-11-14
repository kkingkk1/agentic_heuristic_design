"""Windows 作业对象：在恢复子进程前归组，并累计整个进程树的 CPU。"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
import subprocess


if os.name == "nt":
    class BasicLimits(ctypes.Structure):
        _fields_ = [("process_time", ctypes.c_int64), ("job_time", ctypes.c_int64),
                    ("flags", wintypes.DWORD), ("min_ws", ctypes.c_size_t),
                    ("max_ws", ctypes.c_size_t), ("active_limit", wintypes.DWORD),
                    ("affinity", ctypes.c_size_t), ("priority", wintypes.DWORD),
                    ("scheduling", wintypes.DWORD)]

    class IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in
                    ("read_ops", "write_ops", "other_ops", "read_bytes", "write_bytes", "other_bytes")]

    class ExtendedLimits(ctypes.Structure):
        _fields_ = [("basic", BasicLimits), ("io", IoCounters),
                    ("process_memory", ctypes.c_size_t), ("job_memory", ctypes.c_size_t),
                    ("peak_process_memory", ctypes.c_size_t), ("peak_job_memory", ctypes.c_size_t)]

    class Accounting(ctypes.Structure):
        _fields_ = [("user", ctypes.c_int64), ("kernel", ctypes.c_int64),
                    ("period_user", ctypes.c_int64), ("period_kernel", ctypes.c_int64),
                    ("faults", wintypes.DWORD), ("total_processes", wintypes.DWORD),
                    ("active_processes", wintypes.DWORD), ("terminated_processes", wintypes.DWORD)]

    class CpuRate(ctypes.Structure):
        _fields_ = [("flags", wintypes.DWORD), ("rate", wintypes.DWORD)]

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateJobObjectW.argtypes = [ctypes.c_void_p, wintypes.LPCWSTR]
    kernel.CreateJobObjectW.restype = wintypes.HANDLE
    kernel.SetInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
    kernel.SetInformationJobObject.restype = wintypes.BOOL
    kernel.QueryInformationJobObject.argtypes = [wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p,
                                                wintypes.DWORD, ctypes.c_void_p]
    kernel.QueryInformationJobObject.restype = wintypes.BOOL
    kernel.AssignProcessToJobObject.argtypes = [wintypes.HANDLE, wintypes.HANDLE]
    kernel.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel.TerminateJobObject.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel.TerminateJobObject.restype = wintypes.BOOL
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.restype = wintypes.BOOL
    ntdll = ctypes.WinDLL("ntdll")
    ntdll.NtResumeProcess.argtypes = [wintypes.HANDLE]
    ntdll.NtResumeProcess.restype = ctypes.c_long


class WindowsJob:
    """无 breakaway 权限；关闭句柄时终止尚存的作业内进程。"""

    def __init__(self):
        if os.name != "nt":
            raise RuntimeError("P0 进程树监督只支持 Windows。")
        self.handle = kernel.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = ExtendedLimits()
        limits.basic.flags = 0x00002000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not kernel.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            error = ctypes.WinError(ctypes.get_last_error())
            self.close()
            raise error

    def start(self, command, **kwargs):
        flags = 0x00000004 | subprocess.CREATE_NO_WINDOW  # CREATE_SUSPENDED
        process = subprocess.Popen(command, creationflags=flags, **kwargs)
        try:
            if not kernel.AssignProcessToJobObject(self.handle, int(process._handle)):
                raise ctypes.WinError(ctypes.get_last_error())
            status = ntdll.NtResumeProcess(int(process._handle))
            if status != 0:
                raise OSError(f"NtResumeProcess 失败：0x{status & 0xffffffff:08x}")
            return process
        except BaseException:
            process.kill()
            process.wait()
            raise

    def accounting(self):
        value = Accounting()
        if not kernel.QueryInformationJobObject(self.handle, 1, ctypes.byref(value), ctypes.sizeof(value), None):
            raise ctypes.WinError(ctypes.get_last_error())
        return {"user_seconds": value.user / 10_000_000,
                "kernel_seconds": value.kernel / 10_000_000,
                "cpu_seconds": (value.user + value.kernel) / 10_000_000,
                "total_processes": value.total_processes,
                "active_processes": value.active_processes,
                "terminated_processes": value.terminated_processes}

    def set_cpu_rate(self, rate):
        """将作业 CPU 限为整机的 rate/10000；仅供预算受限的独立检查。"""
        if not 1 <= rate <= 10000:
            raise ValueError("作业 CPU 比率必须在 1..10000。")
        value = CpuRate(0x1 | 0x4, rate)  # ENABLE | HARD_CAP
        if not kernel.SetInformationJobObject(self.handle, 15, ctypes.byref(value), ctypes.sizeof(value)):
            raise ctypes.WinError(ctypes.get_last_error())

    def terminate(self):
        if not kernel.TerminateJobObject(self.handle, 1):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        if self.handle:
            kernel.CloseHandle(self.handle)
            self.handle = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
