#!/usr/bin/env bash
set -euo pipefail

if [[ -n "${PUBLIC_KEY:-}" ]]; then
    mkdir -p /root/.ssh
    echo "${PUBLIC_KEY}" >> /root/.ssh/authorized_keys
    chmod 700 /root/.ssh
    chmod 600 /root/.ssh/authorized_keys
fi
/usr/sbin/sshd

image_repo=/opt/visconf
mounted_repo=/workspace/VisualPO/VisConf

if [[ -f "${VISCONF_REPO:-${mounted_repo}}/pyproject.toml" ]]; then
    repo_root="${VISCONF_REPO:-${mounted_repo}}"
else
    repo_root="${image_repo}"
fi

export PYTHONPATH="${repo_root}/src${PYTHONPATH:+:${PYTHONPATH}}"
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
