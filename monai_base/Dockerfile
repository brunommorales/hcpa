ARG BASE_IMAGE=nvcr.io/nvidia/pytorch:24.12-py3
FROM ${BASE_IMAGE}

USER root

ARG UID=1000
ARG GID=1000
ARG USERNAME=app
RUN set -eux; \
    if ! getent group "${GID}" >/dev/null; then groupadd -g "${GID}" "${USERNAME}"; fi; \
    if id -u "${USERNAME}" >/dev/null 2>&1; then \
        usermod -u "${UID}" -g "${GID}" "${USERNAME}"; \
    elif id -u "${UID}" >/dev/null 2>&1; then \
        existing_user="$(id -nu "${UID}")"; \
        usermod -l "${USERNAME}" -d "/home/${USERNAME}" -m "${existing_user}"; \
        usermod -g "${GID}" "${USERNAME}"; \
    else \
        useradd -m -u "${UID}" -g "${GID}" -s /bin/bash "${USERNAME}"; \
    fi

WORKDIR /workspace
COPY requirements.txt /tmp/requirements.txt
RUN set -eux; \
    python -m pip install --upgrade pip; \
    python -m pip install -r /tmp/requirements.txt; \
    rm -rf /root/.cache/pip

ENV PYTHONPATH=/workspace \
    TF_CPP_MIN_LOG_LEVEL=1

USER ${USERNAME}
CMD ["/bin/bash"]
