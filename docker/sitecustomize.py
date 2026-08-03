"""Cap CPU thread pools before torch initializes them.

RunPod attaches SSH sessions into the container without sshd or PAM and with a
non-login shell, so neither /etc/environment nor /etc/profile.d reaches them --
environment-based configuration cannot be relied on here. Python runs this
module at interpreter startup for every process, before torch is imported,
which makes it the one place the cap always applies.

PyTorch otherwise sizes its pools to the visible core count (128 on the
dual-socket EPYC pod host). The decode loop issues many small per-row tensor
ops, and coordinating that many threads costs more than the ops themselves --
worse across two NUMA nodes, and worse again under a cgroup CPU quota, where
the threads contend for a fraction of the cores they think they have. Left
uncapped this starves the GPU: measured ~5% duty cycle before, >90% after.

An explicit OMP_NUM_THREADS always wins; VISCONF_MAX_THREADS sets the ceiling.
"""

import os


def _cgroup_cpu_limit():
    try:
        with open("/sys/fs/cgroup/cpu.max", encoding="utf-8") as handle:
            quota, period = handle.read().split()[:2]
        if quota == "max":
            return None
        cores = int(quota) // int(period)
        return max(cores, 1)
    except Exception:
        return None


def _apply():
    if os.environ.get("OMP_NUM_THREADS"):
        return
    try:
        threads = int(os.environ.get("VISCONF_MAX_THREADS", "8"))
    except ValueError:
        threads = 8
    limit = _cgroup_cpu_limit()
    if limit is not None and limit < threads:
        threads = limit
    os.environ["OMP_NUM_THREADS"] = str(threads)
    os.environ.setdefault("MKL_NUM_THREADS", str(threads))


try:
    _apply()
except Exception:
    pass
