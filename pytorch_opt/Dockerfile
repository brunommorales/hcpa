# DRIVER R545.x → use 24.11; se atualizar p/ R580+, troque para 25.09
FROM nvcr.io/nvidia/pytorch:24.11-py3
# FROM nvcr.io/nvidia/pytorch:25.09-py3   # (quando tiver driver R580+)

# Cria usuário com seu UID/GID pra não gerar arquivos como root
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

# Workspace
RUN mkdir -p /workspace && chown -R ${UID}:${GID} /workspace
WORKDIR /workspace

# Keras 3 usando backend torch (se seu script usa Keras)
ENV KERAS_BACKEND=torch

# Copie deps de usuário (NÃO inclua torch/torchvision aqui)
COPY requirements-ngc.txt /tmp/requirements-ngc.txt

# Instala respeitando constraints **se existir** (25.03+); na 24.10 segue sem.
RUN set -eu; \
    if [ -f /etc/pip/constraint.txt ]; then C="-c /etc/pip/constraint.txt"; else C=""; fi; \
    python -m pip install --upgrade pip && \
    python -m pip install -r /tmp/requirements-ngc.txt $C && \
    # pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda120 && \
    rm -rf /root/.cache/pip

# Troca para não-root
USER ${USERNAME}

CMD ["/bin/bash"]
