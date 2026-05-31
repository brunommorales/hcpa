# DRIVER R545.x -> use 24.11; if upgrading to R580+, switch to 25.09
FROM nvcr.io/nvidia/pytorch:24.11-py3
# FROM nvcr.io/nvidia/pytorch:25.09-py3

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

RUN mkdir -p /workspace /home/users/bmmorales/projects && \
    ln -sfn /workspace /home/users/bmmorales/projects/hcpa && \
    chown -R ${UID}:${GID} /workspace

WORKDIR /workspace

ENV KERAS_BACKEND=torch
ENV PYTHONPATH=/workspace

COPY requirements-ngc.txt /tmp/requirements-ngc.txt

RUN set -eu; \
    if [ -f /etc/pip/constraint.txt ]; then C="-c /etc/pip/constraint.txt"; else C=""; fi; \
    python -m pip install --upgrade pip && \
    python -m pip install -r /tmp/requirements-ngc.txt $C && \
    rm -rf /root/.cache/pip

USER ${USERNAME}

CMD ["/bin/bash"]
