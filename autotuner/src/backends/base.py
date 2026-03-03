"""
base.py — Classe base abstrata para backends de framework.
"""
from __future__ import annotations

import abc
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class BackendBase(abc.ABC):
    """
    Interface abstrata para backends (PyTorch, TensorFlow, MONAI).

    Cada backend é responsável por:
    1. Validar que a variante escolhida (path) existe e contém os arquivos necessários.
    2. Construir o comando de execução com os argumentos derivados da config atual.
    3. Aplicar ajustes online que o controller determinar.
    4. Coletar métricas do treino (parsear stdout/stderr ou CSV de saída).
    """

    STACK_NAME: str = ""  # "pytorch", "tensorflow", "monai"

    def __init__(self, variant_path: Path, mode: str):
        """
        Args:
            variant_path: Diretório raiz da variante (ex: /path/to/pytorch_base)
            mode: "base" ou "opt"
        """
        self.variant_path = Path(variant_path)
        self.mode = mode
        if not self.variant_path.is_dir():
            raise FileNotFoundError(f"Diretório da variante não encontrado: {self.variant_path}")

    @abc.abstractmethod
    def get_entry_point(self) -> str:
        """Retorna o path do script principal de treino."""
        ...

    @abc.abstractmethod
    def build_command(self, config: Dict[str, Any]) -> List[str]:
        """
        Constrói o comando de execução com args derivados da config.

        Args:
            config: Configuração atual do espaço derivado.

        Returns:
            Lista de args para subprocess (ex: ["python", "train.py", "--batch_size", "96"])
        """
        ...

    @abc.abstractmethod
    def config_to_cli_args(self, config: Dict[str, Any]) -> List[str]:
        """Converte config dict para lista de argumentos CLI."""
        ...

    @abc.abstractmethod
    def parse_epoch_metrics(self, output: str) -> Dict[str, Any]:
        """
        Parseia saída do treino para extrair métricas de uma época.

        Returns:
            Dict com chaves como 'train_loss', 'val_loss', 'val_auc', etc.
        """
        ...

    def validate(self) -> bool:
        """Verifica se a variante é executável."""
        entry = self.get_entry_point()
        return Path(entry).exists()

    def get_results_dir(self, config: Dict[str, Any]) -> Path:
        """Retorna diretório de resultados da variante."""
        results = config.get("results", config.get("results_dir", "./results"))
        return self.variant_path / results

    def _map_config_key(self, key: str) -> Optional[str]:
        """Mapeia chave do espaço derivado para argumento CLI da variante."""
        # Override nos backends específicos conforme necessário
        return f"--{key}"

    def _filter_applicable_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Filtra config para incluir apenas chaves aceitas pela variante."""
        # Subclasses devem implementar filtro específico
        return config
