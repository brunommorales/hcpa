# Base image aligned with R545+ drivers; update tag when upgrading drivers (ex.: 25.09 for R580+)
FROM nvcr.io/nvidia/tensorflow:25.02-tf2-py3

# ---- Create non-root user (matches host UID/GID to avoid root-owned outputs) ----
ARG UID=1000
ARG GID=1000
ARG USERNAME=app
RUN set -eux; \
    if ! getent group "${GID}" >/dev/null; then \
        groupadd -g "${GID}" "${USERNAME}"; \
    fi; \
    if id -u "${USERNAME}" >/dev/null 2>&1; then \
        usermod -u "${UID}" -g "${GID}" "${USERNAME}"; \
    elif id -u "${UID}" >/dev/null 2>&1; then \
        existing_user="$(id -nu "${UID}")"; \
        usermod -l "${USERNAME}" -d "/home/${USERNAME}" -m "${existing_user}"; \
        usermod -g "${GID}" "${USERNAME}"; \
    else \
        useradd -m -u "${UID}" -g "${GID}" -s /bin/bash "${USERNAME}"; \
    fi

# ---- Workspace defaults ----
RUN mkdir -p /workspace && chown -R ${UID}:${GID} /workspace
WORKDIR /workspace

# TensorFlow already default, but Keras 3 can switch backends via env; keep explicit for clarity
ENV KERAS_BACKEND=tensorflow

# ---- Python dependencies ----
COPY requirements /tmp/requirements.txt

RUN set -eux; \
    python -m pip install --upgrade pip; \
    if [ -f /etc/pip/constraint.txt ]; then \
        python -m pip install -r /tmp/requirements.txt -c /etc/pip/constraint.txt; \
        python -m pip install --extra-index-url https://developer.download.nvidia.com/compute/redist \
            -c /etc/pip/constraint.txt \
            nvidia-dali-cuda120 \
            nvidia-dali-tf-plugin-cuda120; \
    else \
        python -m pip install -r /tmp/requirements.txt; \
        python -m pip install --extra-index-url https://developer.download.nvidia.com/compute/redist \
            nvidia-dali-cuda120 \
            nvidia-dali-tf-plugin-cuda120; \
    fi; \
    rm -rf /root/.cache/pip

# ---- Runtime defaults ----
ENV TF_CPP_MIN_LOG_LEVEL=1 \
    TF_GPU_ALLOCATOR=cuda_malloc_async

# Switch to requested non-root user
USER ${USERNAME}

CMD ["/bin/bash"]
