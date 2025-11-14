"""
Pipeline de Ingestão com Docling

Este módulo implementa o pipeline completo de processamento de documentos PDF:
1. Extração estruturada com Docling (texto, tabelas, figuras)
2. Armazenamento de imagens no SQLite (base64)
3. Chunking adaptativo por tipo de conteúdo
4. Geração de embeddings multimodais (CLIP)
5. Armazenamento no ChromaDB
"""

import logging
import os
import uuid
import base64
from io import BytesIO
from datetime import datetime, timezone
from typing import List, Dict
from sqlalchemy.orm import Session

from docling.document_converter import DocumentConverter
from langchain_chroma import Chroma
from langchain_core.documents import Document as LangChainDocument
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from src.services.adaptive_chunker import split_text_with_metadata
from src.services.chunking_strategy import SemanticChunker, expand_context_with_neighbors
from src.auth.database import DocumentImage, Document

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SentenceTransformerEmbeddings(Embeddings):
    """
    Wrapper para usar Sentence Transformers com LangChain.

    Implementa a interface Embeddings do LangChain usando
    a biblioteca sentence-transformers nativa.
    """

    def __init__(self, model_name: str = "sentence-transformers/clip-ViT-B-32-multilingual-v1"):
        """
        Inicializa o modelo Sentence Transformer.

        Args:
            model_name: Nome do modelo no Hugging Face
        """
        self.model = SentenceTransformer(model_name)
        logger.info(f"Modelo Sentence Transformer carregado: {model_name}")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Gera embeddings para uma lista de documentos.

        Args:
            texts: Lista de textos

        Returns:
            Lista de embeddings (lista de floats)
        """
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        Gera embedding para uma query.

        Args:
            text: Texto da query

        Returns:
            Embedding (lista de floats)
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding.tolist()


def get_embedding_function():
    """
    Retorna a função de embeddings (CLIP multilíngue via Sentence Transformers).

    Returns:
        SentenceTransformerEmbeddings configurado
    """
    return SentenceTransformerEmbeddings(
        model_name="sentence-transformers/clip-ViT-B-32-multilingual-v1"
    )


def get_chroma_vectorstore(user_id: int):
    """
    Obtém ou cria um vector store Chroma para o usuário via LangChain.

    Args:
        user_id: ID do usuário

    Returns:
        Chroma vector store do LangChain
    """
    collection_name = f"user_{user_id}_documents"

    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=get_embedding_function(),
        persist_directory="./chroma_db"
    )

    return vectorstore


def save_image_to_db(
    image_pil,
    doc_id: str,
    page_number: int,
    caption: str,
    db: Session
) -> str:
    """
    Salva uma imagem no banco de dados SQLite.

    Args:
        image_pil: Imagem PIL do Docling
        doc_id: ID do documento
        page_number: Número da página
        caption: Legenda da imagem
        db: Sessão do banco de dados

    Returns:
        ID da imagem salva
    """
    image_id = str(uuid.uuid4())

    # Converter PIL Image para base64
    buffered = BytesIO()
    image_pil.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

    # Salvar no banco
    img_record = DocumentImage(
        id=image_id,
        document_id=doc_id,
        page_number=page_number,
        image_data=img_base64,
        image_format="png",
        caption=caption,
        created_at=datetime.now(timezone.utc)
    )

    db.add(img_record)
    db.commit()

    logger.info(f"Imagem {image_id} salva no banco (página {page_number})")

    return image_id


def extract_images_from_pdf_with_pymupdf(file_path: str, doc_id: str, db: Session) -> Dict[int, List[str]]:
    """
    Extrai TODAS as imagens do PDF usando PyMuPDF, página por página.

    Esta função complementa o Docling extraindo imagens que ele não consegue (vetoriais, embutidas, etc).

    Args:
        file_path: Caminho do arquivo PDF
        doc_id: ID do documento
        db: Sessão do banco de dados

    Returns:
        Dicionário mapeando número da página para lista de IDs de imagens:
        {1: ['img_id_1', 'img_id_2'], 2: ['img_id_3'], ...}
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        logger.error("PyMuPDF (fitz) não instalado. Execute: pip install PyMuPDF")
        return {}

    from PIL import Image

    page_images = {}  # {page_num: [image_id1, image_id2, ...]}

    try:
        pdf_document = fitz.open(file_path)
        logger.info(f"PyMuPDF: Extraindo imagens de {len(pdf_document)} páginas...")

        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            image_list = page.get_images(full=True)

            page_image_ids = []

            for img_index, img in enumerate(image_list):
                try:
                    xref = img[0]
                    base_image = pdf_document.extract_image(xref)
                    image_bytes = base_image["image"]

                    # Converter bytes para PIL Image
                    pil_image = Image.open(BytesIO(image_bytes))

                    # Converter para RGB se necessário
                    if pil_image.mode not in ('RGB', 'RGBA'):
                        pil_image = pil_image.convert('RGB')

                    # Salvar no banco
                    caption = f"Imagem extraída da página {page_num + 1}"
                    image_id = save_image_to_db(pil_image, doc_id, page_num + 1, caption, db)
                    page_image_ids.append(image_id)

                except Exception as e:
                    logger.warning(f"Erro ao extrair imagem {img_index} da página {page_num + 1}: {str(e)}")
                    continue

            if page_image_ids:
                page_images[page_num + 1] = page_image_ids

        pdf_document.close()

        total_images = sum(len(ids) for ids in page_images.values())
        logger.info(f"✅ PyMuPDF extraiu {total_images} imagens de {len(page_images)} páginas")

    except Exception as e:
        logger.error(f"Erro ao extrair imagens com PyMuPDF: {str(e)}")

    return page_images


def process_figure_element(element, doc_id: str, page: int, db: Session) -> Dict:
    """
    Processa um elemento do tipo 'figure' do Docling.

    Args:
        element: Elemento Docling do tipo figure/picture
        doc_id: ID do documento
        page: Número da página
        db: Sessão do banco de dados

    Returns:
        Chunk com metadados da figura ou None se não conseguir extrair imagem
    """
    # Extrair caption
    caption = element.text if hasattr(element, 'text') else None

    # Tentar extrair imagem PIL com verificação robusta
    image_pil = None
    if hasattr(element, 'image') and element.image is not None:
        if hasattr(element.image, 'pil_image') and element.image.pil_image is not None:
            image_pil = element.image.pil_image

    if image_pil is None:
        # Elemento picture sem PIL Image extraível - não é erro, apenas skip silencioso
        return None

    # Salvar imagem no SQLite
    image_id = save_image_to_db(image_pil, doc_id, page, caption, db)

    # Criar chunk de referência
    chunk_content = caption if caption else f"[Figura na página {page}]"

    return {
        "content": chunk_content,
        "metadata": {
            "document_id": doc_id,
            "page": page,
            "chunk_type": "figure",
            "has_images": True,
            "image_ids": image_id
        }
    }


def process_table_element(element, doc_id: str, page: int, file_path: str) -> List[Dict]:
    """
    Processa um elemento do tipo 'table' do Docling.

    Estratégia: 1 chunk por linha da tabela, com header repetido.

    Args:
        element: Elemento Docling do tipo table
        doc_id: ID do documento
        page: Número da página
        file_path: Caminho do arquivo fonte

    Returns:
        Lista de chunks (um por linha da tabela)
    """
    chunks = []

    # Extrair dados da tabela
    table_data = element.data if hasattr(element, 'data') else None
    title = element.text if hasattr(element, 'text') else "Tabela"

    if table_data is None or not hasattr(table_data, 'to_dict'):
        logger.warning(f"Tabela na página {page} sem dados estruturados")
        return chunks

    # Converter para dicionários
    rows = table_data.to_dict('records')
    headers = list(table_data.columns)

    # Criar um chunk por linha (com header repetido)
    for row_idx, row in enumerate(rows):
        row_text = f"{title}\n"
        row_text += " | ".join(headers) + "\n"
        row_text += " | ".join(str(v) for v in row.values())

        chunk = {
            "content": row_text,
            "metadata": {
                "document_id": doc_id,
                "page": page,
                "chunk_type": "table",
                "table_title": title,
                "row_index": row_idx,
                "has_images": False,
                "source_file": file_path
            }
        }
        chunks.append(chunk)

    logger.info(f"Tabela '{title}' processada: {len(rows)} linhas")

    return chunks


def process_text_element(element, doc_id: str, page: int, file_path: str, image_ids: str = None) -> List[Dict]:
    """
    Processa elementos de texto (paragraph, heading, list_item) com adaptive chunking.

    Args:
        element: Elemento Docling de texto
        doc_id: ID do documento
        page: Número da página
        file_path: Caminho do arquivo fonte
        image_ids: CSV string de IDs de imagens da página (opcional)

    Returns:
        Lista de chunks gerados pelo adaptive chunker
    """
    text = element.text if hasattr(element, 'text') else ""

    if not text or not text.strip():
        return []

    # Usar adaptive chunker com image_ids se houver
    chunks = split_text_with_metadata(
        text=text,
        base_metadata={
            "document_id": doc_id,
            "page": page,
            "source_file": file_path
        },
        image_ids=image_ids
    )

    return chunks


async def process_document_with_docling(
    file_path: str,
    doc_id: str,
    user_id: int,
    db: Session,
    use_semantic_chunking: bool = True
) -> int:
    """
    Pipeline completo de processamento de documento com Docling.

    Fluxo:
    1. Carregar PDF com Docling
    2. Extrair elementos e aplicar pré-processamento:
       - Agrupar elementos minúsculos em blocos semânticos
       - Reconstruir fórmulas quebradas
       - Manter contexto entre elementos relacionados
    3. Processar cada tipo de elemento:
       - figure → salvar imagem no SQLite
       - table → chunking especializado (1 chunk por linha)
       - text → adaptive chunking com agrupamento semântico
    4. Gerar embeddings com CLIP multilíngue
    5. Salvar chunks e embeddings no ChromaDB

    Args:
        file_path: Caminho do arquivo PDF
        doc_id: ID do documento
        user_id: ID do usuário
        db: Sessão do banco de dados
        use_semantic_chunking: Se True, usa estratégia de chunking semântico inteligente

    Returns:
        Número de chunks criados
    """
    logger.info(f"Iniciando processamento do documento {doc_id}: {file_path}")
    logger.info(f"Semantic chunking: {'ATIVADO' if use_semantic_chunking else 'DESATIVADO'}")

    # 1. Carregar documento com Docling
    converter = DocumentConverter()
    result = converter.convert(file_path)
    docling_doc = result.document

    total_elements = len(list(docling_doc.iterate_items()))
    logger.info(f"Documento carregado: {total_elements} elementos")

    # 1.5. Extrair TODAS as imagens com PyMuPDF (complementar ao Docling)
    logger.info("Extraindo imagens com PyMuPDF...")
    page_images_map = extract_images_from_pdf_with_pymupdf(file_path, doc_id, db)

    # 2. PRÉ-PROCESSAMENTO SEMÂNTICO (SE ATIVADO)
    if use_semantic_chunking:
        logger.info("🔧 Aplicando pré-processamento semântico...")

        # Extrair todos os elementos em lista estruturada
        # Usando estratégia "último header visto" para propagar section_header
        raw_elements = []
        last_section_header = ""

        for idx, (element, level) in enumerate(docling_doc.iterate_items()):
            page = element.prov[0].page_no if element.prov else 0
            el_type = element.label
            text = element.text if hasattr(element, 'text') else ""

            # Atualizar o header corrente quando encontrar um section_header "de verdade"
            if el_type == "section_header":
                last_section_header = text.strip()

            raw_elements.append({
                'text': text,
                'page': page,
                'type': el_type,
                'element_id': idx,
                'section_header': last_section_header,
                'element_obj': element  # Manter referência ao elemento original
            })

        # Aplicar semantic chunker
        semantic_chunker = SemanticChunker(
            min_chunk_size=300,
            max_chunk_size=1000,
            overlap_size=100
        )

        semantic_chunks = semantic_chunker.group_elements(raw_elements)
        logger.info(f"✅ Agrupamento semântico: {len(raw_elements)} elementos → {len(semantic_chunks)} chunks")

    all_chunks = []

    # Contadores para debug
    element_types = {}
    elements_with_images = []

    # 3. Processar chunks semânticos ou elementos individuais
    if use_semantic_chunking and semantic_chunks:
        logger.info("Processando chunks semânticos agrupados...")

        for idx, sem_chunk in enumerate(semantic_chunks):
            try:
                # Associar imagens da página aos chunks de texto
                page_image_ids_list = page_images_map.get(sem_chunk.page, [])
                page_image_ids_csv = ",".join(page_image_ids_list) if page_image_ids_list else None

                # Gerar chunk_id consistente
                chunk_id = f"{doc_id}_chunk_{idx}"

                # Criar chunk com texto agrupado e metadados enriquecidos
                chunk = {
                    "content": sem_chunk.text,
                    "metadata": {
                        "chunk_id": chunk_id,
                        "document_id": doc_id,
                        "page": sem_chunk.page,
                        "chunk_type": sem_chunk.chunk_type,
                        "source_file": file_path,
                        "has_images": bool(page_image_ids_csv),
                        "image_ids": page_image_ids_csv or "",
                        "num_elements": sem_chunk.metadata.get('num_elements', 1),
                        "has_formula": sem_chunk.metadata.get('has_formula', False),
                        "section": sem_chunk.metadata.get('section', '')
                    }
                }
                all_chunks.append(chunk)

            except Exception as e:
                logger.error(f"Erro ao processar chunk semântico na página {sem_chunk.page}: {str(e)}")
                continue

        logger.info(f"✅ {len(all_chunks)} chunks semânticos processados")

    else:
        # Processamento tradicional elemento por elemento
        logger.info("Processando elementos individualmente (modo tradicional)...")

        for idx, (element, level) in enumerate(docling_doc.iterate_items()):
            page = element.prov[0].page_no if element.prov else 0

            # Extrair tipo de elemento
            el_type = element.label

            # Contar tipos de elementos
            element_types[el_type] = element_types.get(el_type, 0) + 1

            # Detectar elementos com imagens (independente do tipo)
            has_image = hasattr(element, 'image') and element.image is not None
            if has_image:
                elements_with_images.append({
                    'type': el_type,
                    'page': page,
                    'has_pil': hasattr(element.image, 'pil_image') if hasattr(element, 'image') else False
                })

            try:
                # --- FIGURAS E PICTURES ---
                if el_type in ["figure", "picture"]:
                    chunk = process_figure_element(element, doc_id, page, db)
                    if chunk:
                        all_chunks.append(chunk)
                    continue

                # --- TABELAS ---
                if el_type == "table":
                    chunks = process_table_element(element, doc_id, page, file_path)
                    all_chunks.extend(chunks)
                    continue

                # --- TEXTO (paragraph, heading, list_item, section_header) ---
                if el_type in ["text", "paragraph", "list_item", "section_header", "title"]:
                    # Associar imagens da mesma página aos chunks de texto
                    page_image_ids_list = page_images_map.get(page, [])
                    page_image_ids_csv = ",".join(page_image_ids_list) if page_image_ids_list else None

                    chunks = process_text_element(element, doc_id, page, file_path, page_image_ids_csv)
                    all_chunks.extend(chunks)
                    continue

            except Exception as e:
                logger.error(f"Erro ao processar elemento {el_type} na página {page}: {str(e)}")
                continue

    # Log de estatísticas de elementos
    logger.info(f"Tipos de elementos encontrados: {element_types}")
    logger.info(f"Elementos com imagens detectados: {len(elements_with_images)}")
    if elements_with_images:
        logger.info(f"Detalhes dos elementos com imagens: {elements_with_images[:10]}")  # Mostrar primeiros 10
    logger.info(f"Total de chunks gerados: {len(all_chunks)}")

    if len(all_chunks) == 0:
        logger.warning("Nenhum chunk foi gerado do documento")
        return 0

    # 2.5. Garantir que todos os chunks tenham chunk_id no metadata
    # (no modo semântico já foi adicionado, no modo tradicional adicionamos aqui)
    for idx, chunk in enumerate(all_chunks):
        if "chunk_id" not in chunk["metadata"]:
            chunk["metadata"]["chunk_id"] = f"{doc_id}_chunk_{idx}"

    # 3. Converter chunks para LangChain Documents
    logger.info("Preparando documentos para o vector store...")
    langchain_docs = []

    for chunk in all_chunks:
        doc = LangChainDocument(
            page_content=chunk["content"],
            metadata=chunk["metadata"]
        )
        langchain_docs.append(doc)

    # 4. Obter vector store e adicionar documentos (embeddings gerados automaticamente)
    logger.info("Salvando no ChromaDB com embeddings CLIP multilíngue...")
    vectorstore = get_chroma_vectorstore(user_id)

    # Extrair chunk_ids dos metadados (já foram definidos anteriormente)
    chunk_ids = [chunk["metadata"]["chunk_id"] for chunk in all_chunks]

    # Adicionar documentos ao vector store (LangChain gera embeddings automaticamente)
    vectorstore.add_documents(
        documents=langchain_docs,
        ids=chunk_ids
    )

    # 5. Salvar registro do documento no SQLite (upsert para evitar duplicatas)
    existing_doc = db.query(Document).filter(Document.id == doc_id).first()
    if existing_doc:
        # Atualizar documento existente
        existing_doc.status = "completed"
        existing_doc.chunks_count = len(all_chunks)
        existing_doc.processed_at = datetime.now(timezone.utc)
    else:
        # Criar novo documento
        doc_record = Document(
            id=doc_id,
            user_id=user_id,
            filename=os.path.basename(file_path),
            file_path=file_path,
            status="completed",
            chunks_count=len(all_chunks),
            created_at=datetime.now(timezone.utc)
        )
        db.add(doc_record)
    db.commit()

    logger.info(f"✅ Documento processado com sucesso: {len(all_chunks)} chunks salvos")

    return len(all_chunks)
