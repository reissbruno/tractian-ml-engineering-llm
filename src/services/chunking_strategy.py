"""
Estratégia de Chunking Inteligente para RAG

Este módulo implementa pré-processamento avançado de documentos Docling:
1. Agrupa elementos minúsculos em blocos semânticos maiores
2. Normaliza fórmulas matemáticas quebradas
3. Detecta e reconstrói sequências de fórmulas
4. Mantém contexto entre elementos relacionados

Objetivo: Resolver fragmentação excessiva que prejudica recuperação RAG.
"""

import logging
import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class SemanticChunk:
    """
    Representa um chunk semântico agrupado.

    Attributes:
        text: Texto completo do chunk
        page: Número da página
        chunk_type: Tipo do chunk (text, formula, table, mixed)
        element_ids: IDs dos elementos originais agrupados
        metadata: Metadados adicionais
    """
    text: str
    page: int
    chunk_type: str
    element_ids: List[int]
    metadata: Dict[str, Any]


class FormulaReconstructor:
    """
    Reconstrói fórmulas matemáticas quebradas em múltiplos elementos.

    Exemplo:
        [141] U
        [142] 2
        [143] P    =        (W)
        [144] R

        Resultado: "P = U² / R (W)"
    """

    # Padrões de fórmulas comuns
    MATH_PATTERNS = {
        'power': r'(\w+)\s*(\d+)',  # U 2 -> U²
        'division': r'(\w+)\s*/\s*(\w+)',  # U / R
        'multiplication': r'(\w+)\s*[.×]\s*(\w+)',  # U . I
        'sqrt': r'√\s*(\d+)',  # √3
        'equation': r'(\w+)\s*=\s*(.+)',  # P = ...
    }

    @staticmethod
    def is_formula_fragment(text: str) -> bool:
        """
        Verifica se o texto é um fragmento de fórmula.

        Args:
            text: Texto a verificar

        Returns:
            True se for fragmento de fórmula
        """
        text = text.strip()

        # Elementos muito curtos com caracteres matemáticos
        if len(text) <= 3 and any(c in text for c in '=()²³+-×/'):
            return True

        # Apenas números ou variáveis únicas
        if len(text) <= 2 and (text.isdigit() or text.isalpha()):
            return True

        # Contém operadores matemáticos isolados
        if text in ['=', '+', '-', '×', '/', '(', ')', '²', '³']:
            return True

        return False

    @staticmethod
    def reconstruct_formula(fragments: List[str]) -> str:
        """
        Reconstrói uma fórmula a partir de fragmentos.

        Args:
            fragments: Lista de fragmentos de texto

        Returns:
            Fórmula reconstruída
        """
        if not fragments:
            return ""

        # Juntar fragmentos com espaços
        reconstructed = " ".join(frag.strip() for frag in fragments if frag.strip())

        # Aplicar normalizações
        reconstructed = FormulaReconstructor._normalize_powers(reconstructed)
        reconstructed = FormulaReconstructor._normalize_operators(reconstructed)
        reconstructed = FormulaReconstructor._normalize_spacing(reconstructed)

        return reconstructed

    @staticmethod
    def _normalize_powers(text: str) -> str:
        """Normaliza potências: 'U 2' -> 'U²', 'I 2' -> 'I²'"""
        # Substituir número após variável por superscript
        text = re.sub(r'(\w)\s+2(?!\w)', r'\1²', text)
        text = re.sub(r'(\w)\s+3(?!\w)', r'\1³', text)
        return text

    @staticmethod
    def _normalize_operators(text: str) -> str:
        """Normaliza operadores: múltiplos espaços, etc."""
        # Normalizar multiplicação
        text = re.sub(r'\s*[.×]\s*', ' × ', text)
        # Normalizar divisão
        text = re.sub(r'\s*/\s*', ' / ', text)
        # Normalizar igualdade
        text = re.sub(r'\s*=\s*', ' = ', text)
        return text

    @staticmethod
    def _normalize_spacing(text: str) -> str:
        """Remove espaços múltiplos e limpa a string."""
        return re.sub(r'\s+', ' ', text).strip()


class SemanticChunker:
    """
    Agrupa elementos Docling em chunks semânticos maiores e mais coerentes.

    Estratégia:
    1. Agrupar elementos consecutivos da mesma página/seção
    2. Respeitar limite de caracteres (500-1000)
    3. Detectar e juntar fórmulas quebradas
    4. Manter contexto entre texto + fórmula + explicação
    """

    def __init__(
        self,
        min_chunk_size: int = 300,
        max_chunk_size: int = 1000,
        overlap_size: int = 100
    ):
        """
        Inicializa o chunker semântico.

        Args:
            min_chunk_size: Tamanho mínimo de chunk em caracteres
            max_chunk_size: Tamanho máximo de chunk em caracteres
            overlap_size: Sobreposição entre chunks vizinhos
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.overlap_size = overlap_size
        self.formula_reconstructor = FormulaReconstructor()

    def _is_minor_header(self, header: str) -> bool:
        """
        Verifica se o header é "menor" (não deve causar split de chunk).

        Headers menores como "Exemplo:", "Nota:", "Observação:" não devem
        iniciar uma nova seção conceitual, mas sim fazer parte da seção atual.

        Args:
            header: Texto do header

        Returns:
            True se for um header menor
        """
        if not header:
            return False

        h = header.strip().lower()

        # Remover dois-pontos final se houver
        if h.endswith(':'):
            h = h[:-1].strip()

        # Lista de headers menores que não devem causar split
        minor_headers = {
            'exemplo', 'exemplos',
            'nota', 'notas',
            'observação', 'observacao', 'observações', 'observacoes',
            'atenção', 'atencao',
            'importante',
            'dica', 'dicas',
            'aviso', 'avisos'
        }

        return h in minor_headers

    def group_elements(self, elements: List[Dict[str, Any]]) -> List[SemanticChunk]:
        """
        Agrupa elementos Docling em chunks semânticos.

        Args:
            elements: Lista de elementos extraídos do Docling

        Returns:
            Lista de chunks semânticos agrupados
        """
        if not elements:
            return []

        chunks = []
        current_buffer = []
        current_length = 0
        current_page = None
        current_section = None
        formula_buffer = []

        for idx, element in enumerate(elements):
            text = element.get('text', '').strip()
            page = element.get('page', 0)
            el_type = element.get('type', 'text')
            section = element.get('section_header', '')

            # Detectar fragmentos de fórmula
            is_formula_frag = self.formula_reconstructor.is_formula_fragment(text)

            # Se é fragmento de fórmula, adicionar ao buffer de fórmulas
            if is_formula_frag or el_type == 'formula':
                formula_buffer.append(text)
                continue

            # Se temos fórmulas acumuladas, reconstruir e adicionar
            if formula_buffer:
                reconstructed = self.formula_reconstructor.reconstruct_formula(formula_buffer)
                if reconstructed:
                    current_buffer.append({
                        'text': reconstructed,
                        'page': page,
                        'type': 'formula',
                        'element_id': idx
                    })
                    current_length += len(reconstructed)
                formula_buffer = []

            # Verificar se houve mudança de seção significativa
            # (ignorar headers menores como "Exemplo:", "Nota:", etc)
            section_changed = (
                current_section
                and section
                and section != current_section
                and not self._is_minor_header(section)
            )

            # Verificar se deve criar novo chunk
            should_split = (
                # Mudou de página
                (current_page is not None and page != current_page) or
                # Mudou de seção (apenas seções "grandes")
                section_changed or
                # Atingiu tamanho máximo
                (current_length + len(text) > self.max_chunk_size and current_length >= self.min_chunk_size)
            )

            if should_split and current_buffer:
                # Criar chunk do buffer atual
                chunk = self._create_chunk(current_buffer)
                if chunk:
                    chunks.append(chunk)

                # Manter overlap se possível
                if self.overlap_size > 0 and current_buffer:
                    overlap_text = self._get_overlap_text(current_buffer)
                    current_buffer = [{'text': overlap_text, 'page': page, 'type': 'text', 'element_id': idx}]
                    current_length = len(overlap_text)
                else:
                    current_buffer = []
                    current_length = 0

            # Adicionar elemento atual ao buffer
            if text:  # Só adicionar se tiver texto
                current_buffer.append({
                    'text': text,
                    'page': page,
                    'type': el_type,
                    'element_id': idx,
                    'section': section
                })
                current_length += len(text)
                current_page = page
                current_section = section

        # Processar fórmulas remanescentes
        if formula_buffer:
            reconstructed = self.formula_reconstructor.reconstruct_formula(formula_buffer)
            if reconstructed and current_buffer:
                current_buffer.append({
                    'text': reconstructed,
                    'page': current_page or 0,
                    'type': 'formula',
                    'element_id': len(elements)
                })

        # Criar último chunk
        if current_buffer:
            chunk = self._create_chunk(current_buffer)
            if chunk:
                chunks.append(chunk)

        logger.info(f"Agrupados {len(elements)} elementos em {len(chunks)} chunks semânticos")

        return chunks

    def _create_chunk(self, buffer: List[Dict[str, Any]]) -> Optional[SemanticChunk]:
        """
        Cria um chunk semântico a partir do buffer.

        Args:
            buffer: Buffer de elementos

        Returns:
            SemanticChunk ou None se buffer vazio
        """
        if not buffer:
            return None

        # Juntar textos com espaçamento adequado
        texts = []
        for item in buffer:
            text = item['text'].strip()
            if text:
                # Fórmulas em linha separada para destaque
                if item['type'] in ['formula', 'equation']:
                    texts.append(f"\n{text}\n")
                else:
                    texts.append(text)

        combined_text = " ".join(texts)
        combined_text = re.sub(r'\s+', ' ', combined_text).strip()

        # Não criar chunks muito pequenos
        if len(combined_text) < self.min_chunk_size and len(buffer) == 1:
            return None

        # Determinar tipo do chunk
        types = [item['type'] for item in buffer]
        if 'formula' in types or 'equation' in types:
            chunk_type = 'mixed' if 'text' in types else 'formula'
        elif 'table' in types:
            chunk_type = 'table'
        else:
            chunk_type = 'text'

        # Pegar página (primeira página do grupo)
        page = buffer[0]['page']

        # IDs dos elementos
        element_ids = [item['element_id'] for item in buffer]

        # Metadados
        metadata = {
            'num_elements': len(buffer),
            'types': list(set(types)),
            'section': buffer[0].get('section', ''),
            'has_formula': 'formula' in types or 'equation' in types
        }

        return SemanticChunk(
            text=combined_text,
            page=page,
            chunk_type=chunk_type,
            element_ids=element_ids,
            metadata=metadata
        )

    def _get_overlap_text(self, buffer: List[Dict[str, Any]]) -> str:
        """
        Obtém texto de overlap do final do buffer.

        Args:
            buffer: Buffer de elementos

        Returns:
            Texto de overlap
        """
        if not buffer:
            return ""

        # Pegar últimos N caracteres
        all_text = " ".join(item['text'] for item in buffer if item['text'])
        if len(all_text) <= self.overlap_size:
            return all_text

        return all_text[-self.overlap_size:]


def expand_context_with_neighbors(
    chunks: List[SemanticChunk],
    target_indices: List[int],
    n_before: int = 1,
    n_after: int = 1
) -> List[SemanticChunk]:
    """
    Expande contexto incluindo chunks vizinhos.

    Útil para recuperação RAG: mesmo que o chunk exato não entre no top_k,
    chunks vizinhos podem conter informação complementar crítica.

    Args:
        chunks: Lista de todos os chunks
        target_indices: Índices dos chunks recuperados
        n_before: Número de chunks anteriores a incluir
        n_after: Número de chunks posteriores a incluir

    Returns:
        Lista expandida de chunks com vizinhos
    """
    expanded_indices = set()

    for idx in target_indices:
        # Adicionar chunk atual
        expanded_indices.add(idx)

        # Adicionar vizinhos anteriores
        for i in range(1, n_before + 1):
            if idx - i >= 0:
                expanded_indices.add(idx - i)

        # Adicionar vizinhos posteriores
        for i in range(1, n_after + 1):
            if idx + i < len(chunks):
                expanded_indices.add(idx + i)

    # Retornar chunks na ordem original
    result = [chunks[i] for i in sorted(expanded_indices)]

    logger.info(f"Expandido {len(target_indices)} chunks para {len(result)} com vizinhos")

    return result
