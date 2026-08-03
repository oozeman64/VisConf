#!/usr/bin/env bash
set -euo pipefail

image_repo=/opt/visconf
mounted_repo=/workspace/VisualPO/VisConf

if [[ -f "${VISCONF_REPO:-${mounted_repo}}/pyproject.toml" ]]; then
    repo_root="${VISCONF_REPO:-${mounted_repo}}"
else
    repo_root="${image_repo}"
fi

# The image ships an editable install whose .pth hardcodes the image copy's
# src directory, so every Python process picks up /opt/visconf unless it is
# overridden. Exporting PYTHONPATH below only reaches this script's children;
# sshd builds a fresh environment for each session, so `ssh pod '<command>'`
# would silently import the stale image copy instead of the mounted repo.
# Repointing the .pth fixes resolution at the path level, for every process,
# with no dependency on environment inheritance.
site_packages="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
for pth in "${site_packages}"/__editable__.visconf-*.pth; do
    [[ -e "${pth}" ]] || continue
    printf '%s\n' "${repo_root}/src" > "${pth}"
done

# PyTorch defaults its CPU thread pools to one thread per visible core (128 on
# a dual-socket EPYC host). The decode loop runs many small per-row tensor ops,
# and coordinating that many threads costs far more than the ops themselves --
# worse on a multi-socket host, where the pool spans both NUMA nodes, and on a
# cgroup-limited pod, where the threads contend for a fraction of those cores.
# Left unset, this starves the GPU: measured ~5% duty cycle before, >90% after.
if [[ -z "${OMP_NUM_THREADS:-}" ]]; then
    visconf_threads="${VISCONF_MAX_THREADS:-8}"
    if [[ -r /sys/fs/cgroup/cpu.max ]]; then
        read -r cpu_quota cpu_period < /sys/fs/cgroup/cpu.max || true
        if [[ "${cpu_quota:-max}" != "max" && "${cpu_period:-0}" -gt 0 ]]; then
            allowed_cores=$(( cpu_quota / cpu_period ))
            (( allowed_cores < 1 )) && allowed_cores=1
            (( allowed_cores < visconf_threads )) && visconf_threads="${allowed_cores}"
        fi
    fi
    export OMP_NUM_THREADS="${visconf_threads}"
fi
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${OMP_NUM_THREADS}}"

export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"

# sshd sessions inherit nothing from this script, but PAM reads /etc/environment
# for both interactive logins and `ssh pod '<command>'` runs (UsePAM yes plus
# pam_env.so in /etc/pam.d/sshd), which makes it the one channel that reaches
# every session type.
{
    grep -v -E '^(PYTHONPATH|OMP_NUM_THREADS|MKL_NUM_THREADS)=' /etc/environment 2>/dev/null || true
    printf 'PYTHONPATH="%s/src"\n' "${repo_root}"
    printf 'OMP_NUM_THREADS="%s"\n' "${OMP_NUM_THREADS}"
    printf 'MKL_NUM_THREADS="%s"\n' "${MKL_NUM_THREADS}"
} > /etc/environment.tmp && mv /etc/environment.tmp /etc/environment

cat > /etc/profile.d/visconf.sh <<PROFILE
export PYTHONPATH="${repo_root}/src\${PYTHONPATH:+:\${PYTHONPATH}}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS}"
PROFILE
chmod 0644 /etc/profile.d/visconf.sh

printf 'visconf: repo=%s threads=%s\n' "${repo_root}" "${OMP_NUM_THREADS}"

if [[ -n "${PUBLIC_KEY:-}" ]]; then
    mkdir -p /root/.ssh
    echo "${PUBLIC_KEY}" >> /root/.ssh/authorized_keys
    chmod 700 /root/.ssh
    chmod 600 /root/.ssh/authorized_keys
fi
# Started only once the files above exist, so the first session to connect
# already sees the resolved repo and thread settings.
/usr/sbin/sshd

cd "${repo_root}"

# A no-argument RunPod Pod should stay alive for SSH/tmux use. Supplying a
# VisConf subcommand runs the CLI; supplying an executable passes through.
if [[ "$#" -eq 0 ]]; then
    exec sleep infinity
fi

case "$1" in
    bash|sh|python|python3|visconf|sleep|tail|/bin/*|/usr/bin/*)
        exec "$@"
        ;;
    *)
        exec python -m visconf.cli "$@"
        ;;
esac
