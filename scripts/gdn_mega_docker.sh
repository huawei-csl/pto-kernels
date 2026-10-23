#!/usr/bin/env bash
# Reuse the prepared container; arguments are a command to execute in it.
set -euo pipefail
container=${GDN_CONTAINER:-gdn-mega-opt}
if ! docker inspect "$container" >/dev/null 2>&1; then
    docker run -d --rm --name "$container" --privileged --network=host --ipc=host \
        --shm-size=16g \
        --device=/dev/davinci0 --device=/dev/davinci1 --device=/dev/davinci2 --device=/dev/davinci3 \
        --device=/dev/davinci4 --device=/dev/davinci5 --device=/dev/davinci6 --device=/dev/davinci7 \
        --device=/dev/davinci_manager --device=/dev/hisi_hdc \
        -v /usr/local/sbin:/usr/local/sbin -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
        -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware \
        -v /etc/ascend_install.info:/etc/ascend_install.info \
        -v /var/queue_schedule:/var/queue_schedule -v "$HOME":/mounted_home \
        -w "/mounted_home/${PWD#"$HOME/"}" \
        quay.io/ascend/cann:9.1.0-910b-ubuntu22.04-py3.11 sleep infinity
    docker exec "$container" python -m pip install torch==2.10.0 torch-npu==2.10.0 pytest pybind11 cmake ninja
fi
exec docker exec -e PYTHONPATH=python -e OMP_NUM_THREADS=4 \
    -e NPU_DEVICE="${NPU_DEVICE:-npu:0}" "$container" "$@"
