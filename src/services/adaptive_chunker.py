"""
Adaptive Chunker - Sistema de Chunking Inteligente e Adaptativo

Este módulo implementa um sistema de chunking que:
- Detecta automaticamente o tipo de conteúdo (tabela, fórmula, aviso, procedimento, etc)
- Define dinamicamente chunk_size e overlap ideal
- Funciona independentemente do nome do documento
- Mantém qualidade semântica e suporte multilíngue
"""

import re
from typing import Tuple, List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter


def detect_block_type(text: str) -> str:
    """
    Detecta o tipo de bloco textual com base em padrões estruturais.

    Args:
        text: Texto a ser analisado

    Returns:
        str: Tipo do bloco (warning, table, formula, procedure, conceptual, narrative)

    Tipos:
        - warning: Avisos de segurança (WARNING, CAUTION, ATENÇÃO)
        - table: Tabelas com dados estruturados
        - formula: Fórmulas matemáticas e expressões técnicas
        - procedure: Listas e procedimentos passo-a-passo
        - conceptual: Conteúdo técnico conceitual com numeração hierárquica
        - narrative: Texto contínuo descritivo
    """
    t = text.strip()
    text_low = t.lower()
    lines = [l.strip() for l in t.splitlines() if l.strip()]

    # 1. Detectar avisos de segurança (prioridade alta)
    if re.search(r"\b(warning|caution|aten[çc][aã]o|aviso|advert[êe]ncia)\b", text_low):
        return "warning"

    # 2. Detectar tabelas "óbvias" (palavra-chave ou pipes)
    if "tabela" in text_low or "table" in text_low or "tabla" in text_low or "|" in t:
        return "table"

    # 3. Detectar procedimentos e listas (2 ou mais passos)
    step_like = sum(
        1
        for l in lines
        if re.match(r'^(\d+[\.\)]\s+|[-*•]\s+)', l)
    )
    if step_like >= 2:
        return "procedure"

    # 4. Detectar tabelas numéricas (como as do LB5001, sem '|')
    # Heurística: bloco com >= 3 linhas e pelo menos 2 linhas "numéricas"
    numeric_like_lines = 0
    for line in lines:
        # Ignorar linhas que são claramente passos de procedimento
        if re.match(r'^(\d+[\.\)]\s+)', line):
            continue

        lower = line.lower()
        tokens = line.split()
        num_tokens = sum(1 for tok in tokens if re.search(r"\d", tok))
        has_hrs = "hrs" in lower  # 12000Hrs., 22,000Hrs., etc

        # Linha "numérica" se:
        # - começa com dígito, OU
        # - tem pelo menos 2 tokens com dígito, OU
        # - menciona horas (Hrs.)
        if re.match(r"^\d", line) or num_tokens >= 2 or has_hrs:
            numeric_like_lines += 1

    if len(lines) >= 3 and numeric_like_lines >= 2:
        return "table"

    # 5. Detectar fórmulas e conteúdo matemático
    # Requer símbolo matemático + dígito OU palavras típicas de função matemática
    has_math_symbol = re.search(r'[=<>×÷±/]|√|∑|Σ', t) is not None
    has_math_word = re.search(r'\b(cos|sen|sin|tan|log|ln)\b', text_low) is not None
    has_digit = re.search(r'\d', t) is not None

    if (has_math_symbol and has_digit) or has_math_word:
        return "formula"

    # 6. Detectar conteúdo conceitual técnico
    # Títulos hierárquicos (1.2.5, Section 3.1, etc)
    if re.match(r"^\d+(\.\d+)+\s", t):
        return "conceptual"

    # 7. Padrão: texto narrativo
    return "narrative"


def get_chunk_params(block_type: str) -> Dict[str, any]:
    """
    Retorna os parâmetros ideais de chunking para cada tipo de bloco.

    Args:
        block_type: Tipo do bloco (warning, table, formula, etc)

    Returns:
        dict: Parâmetros para RecursiveCharacterTextSplitter
              - chunk_size: tamanho do chunk
              - chunk_overlap: sobreposição entre chunks
              - separators: lista de separadores
    """
    params_map = {
        "table": {
            "chunk_size": 300,
            "chunk_overlap": 50,
            "separators": ["\n", "|", ";", ","]
        },
        "formula": {
            "chunk_size": 1200,
            "chunk_overlap": 150,
            "separators": ["\n\n", "\n", ".", ";"]
        },
        "conceptual": {
            "chunk_size": 1200,
            "chunk_overlap": 150,
            "separators": ["\n\n", "\n", ".", ";"]
        },
        "procedure": {
            "chunk_size": 400,
            "chunk_overlap": 80,
            "separators": ["\n", "\n\n", ".", ";"]
        },
        "warning": {
            "chunk_size": 250,
            "chunk_overlap": 20,
            "separators": ["WARNING:", "CAUTION:", "ATENÇÃO:", "AVISO:", "\n"]
        },
        "narrative": {
            "chunk_size": 800,
            "chunk_overlap": 100,
            "separators": ["\n\n", "\n", ".", ";", " "]
        }
    }

    return params_map.get(block_type, params_map["narrative"])


def split_text_dynamic(text: str) -> Tuple[str, List[str]]:
    """
    Divide o texto de forma adaptativa baseado no tipo de conteúdo detectado.

    Este é o método principal do chunking adaptativo. Ele:
    1. Detecta automaticamente o tipo de conteúdo
    2. Seleciona os parâmetros ideais de chunking
    3. Aplica o text splitter configurado
    4. Retorna o tipo detectado e os chunks gerados
    """
    # 1. Detectar tipo do bloco
    block_type = detect_block_type(text)

    # 2. Obter parâmetros de chunking para este tipo
    params = get_chunk_params(block_type)

    # 3. Criar splitter com os parâmetros
    splitter = RecursiveCharacterTextSplitter(**params)

    # 4. Dividir texto em chunks
    chunks = splitter.split_text(text)

    return block_type, chunks


def split_text_with_metadata(
    text: str,
    base_metadata: Dict = None,
    image_ids: str = None
) -> List[Dict]:
    """
    Divide o texto em chunks e enriquece cada chunk com metadados completos.

    Args:
        text: Texto a ser dividido
        base_metadata: Metadados base a serem adicionados (source_pdf, page, etc)
        image_ids: String CSV com IDs de imagens associadas (ex: "img1,img2")

    Returns:
        List[dict]: Lista de dicionários com 'content' e 'metadata'
    """
    if base_metadata is None:
        base_metadata = {}

    # Obter tipo e chunks
    block_type, chunks = split_text_dynamic(text)
    params = get_chunk_params(block_type)

    # Criar lista de chunks com metadados
    result = []
    for idx, chunk in enumerate(chunks):
        metadata = {
            **base_metadata,
            "chunk_type": block_type,
            "chunk_index": idx,
            "chunk_size_used": params["chunk_size"],
            "chunk_overlap_used": params["chunk_overlap"],
            "total_chunks": len(chunks),
            "has_images": bool(image_ids)
        }

        # Adicionar image_ids se houver
        if image_ids:
            metadata["image_ids"] = image_ids

        result.append({
            "content": chunk,
            "metadata": metadata
        })

    return result


# ============================================================================
# FUNÇÕES AUXILIARES PARA CASOS ESPECIAIS
# ============================================================================

def extract_table_with_header(text: str) -> List[Dict]:
    """
    Extração especializada para tabelas, mantendo cabeçalho em cada chunk.

    Para tabelas, cada linha deve preservar:
    - Título da tabela
    - Cabeçalhos das colunas
    - Dados da linha

    Args:
        text: Texto da tabela

    Returns:
        List[dict]: Chunks com header repetido
    """
    # Normalizar linhas (remover vazias e espaços extras)
    lines = [l.strip() for l in text.splitlines() if l.strip()]

    if not lines:
        return []

    title = ""
    header = ""
    data_lines: List[str] = []

    for line in lines:
        lower = line.lower()

        # Título da tabela (preferir linhas com 'table/tabela/tabla')
        if re.search(r"(table|tabela|tabla)", lower):
            title = line.strip()
            continue

        # Primeira linha com '|' é considerada header
        if "|" in line and not header:
            header = line.strip()
            continue

        # Demais linhas com '|' são dados
        if "|" in line:
            data_lines.append(line.strip())

    # Fallback: se não encontramos título por palavra-chave, usar primeira linha
    if not title:
        title = lines[0]

    # Se não há header com '|', não tentamos extrair dados nesse helper
    if not header or not data_lines:
        return []

    chunks: List[Dict] = []
    for data_line in data_lines:
        chunk_text = f"{title}\n{header}\n{data_line}"
        chunks.append({
            "content": chunk_text,
            "metadata": {
                "chunk_type": "table",
                "has_header": True,
                "title": title
            }
        })

    return chunks


def get_block_type_stats(texts: List[str]) -> Dict[str, int]:
    """
    Retorna estatísticas de tipos de blocos em um conjunto de textos.

    Útil para análise e debugging do pipeline.

    Args:
        texts: Lista de textos a analisar

    Returns:
        dict: Contagem de cada tipo de bloco
    """
    stats = {}
    for text in texts:
        block_type = detect_block_type(text)
        stats[block_type] = stats.get(block_type, 0) + 1

    return stats
