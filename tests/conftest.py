"""Configurações de teste compartilhadas.

Garante que a raiz do projeto esteja no sys.path antes da importação dos pacotes.
Isso ajuda quando os testes são executados com pytest ou outras ferramentas que
não garantem que o diretório do projeto esteja no PYTHONPATH.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ROOT_STR = str(ROOT)

if ROOT_STR not in sys.path:
    # Inserir no começo para priorizar o código do repositório sobre pacotes instalados
    sys.path.insert(0, ROOT_STR)
