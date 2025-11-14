"""
Serviço RAG - Retrieval-Augmented Generation

Este módulo implementa o sistema de busca e recuperação:
1. Query no ChromaDB por similaridade semântica
2. Recuperação de imagens associadas do SQLite
3. Montagem de contexto multimodal (texto + imagens)
4. Preparação para envio ao LLM
"""

import logging
import os
from typing import Dict, List
from sqlalchemy.orm import Session

# Configurar modo offline para modelos já baixados
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from sentence_transformers import SentenceTransformer

from src.auth.database import DocumentImage

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
    Obtém o vector store Chroma do usuário via LangChain.

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


def get_images_by_ids(image_ids: List[str], db: Session) -> List[Dict]:
    """
    Recupera imagens do SQLite por lista de IDs.

    Args:
        image_ids: Lista de IDs de imagens
        db: Sessão do banco de dados

    Returns:
        Lista de dicionários com dados das imagens:
        [
            {
                "id": "...",
                "data": "base64_string",
                "format": "png",
                "caption": "...",
                "page": 5
            }
        ]
    """
    if not image_ids:
        return []

    images = db.query(DocumentImage).filter(
        DocumentImage.id.in_(image_ids)
    ).all()

    result = [
        {
            "id": img.id,
            "data": img.image_data,  # Base64 string
            "format": img.image_format,
            "caption": img.caption,
            "page": img.page_number
        }
        for img in images
    ]

    logger.info(f"Recuperadas {len(result)} imagens do banco de dados")

    return result


async def query_documents(
    question: str,
    user_id: int,
    db: Session,
    top_k: int = 5,
    expand_neighbors: bool = True,
    n_before: int = 1,
    n_after: int = 1
) -> Dict:
    """
    Busca documentos relevantes no ChromaDB e recupera imagens associadas.

    Fluxo:
    1. Buscar top_k chunks mais similares usando LangChain
    2. (OPCIONAL) Expandir contexto incluindo chunks vizinhos
    3. Para cada chunk:
       - Verificar se tem imagens (metadata.has_images)
       - Se sim, recuperar imagens do SQLite
    4. Retornar contexto estruturado (texto + imagens)

    Args:
        question: Pergunta do usuário
        user_id: ID do usuário
        db: Sessão do banco de dados
        top_k: Número de chunks a recuperar
        expand_neighbors: Se True, inclui chunks vizinhos para contexto adicional
        n_before: Número de chunks anteriores a incluir
        n_after: Número de chunks posteriores a incluir

    Returns:
        {
            "question": "...",
            "chunks": [
                {
                    "id": "...",
                    "text": "...",
                    "metadata": {...},
                    "score": 0.85,
                    "images": [
                        {"id": "...", "data": "base64...", ...}
                    ],
                    "is_neighbor": False  # True se chunk foi adicionado por expansão
                }
            ],
            "total_chunks": 5,
            "expanded_chunks": 8  # Se expansão ativada
        }
    """
    logger.info(f"Processando query: '{question}' (user_id={user_id}, expand_neighbors={expand_neighbors})")

    # 1. Obter vector store
    try:
        vectorstore = get_chroma_vectorstore(user_id)
    except Exception as e:
        logger.error(f"Erro ao acessar vector store do usuário {user_id}: {str(e)}")
        return {
            "question": question,
            "chunks": [],
            "total_chunks": 0,
            "error": "Nenhum documento indexado encontrado"
        }

    # 2. Buscar documentos similares (LangChain gera embedding automaticamente)
    results = vectorstore.similarity_search_with_score(
        query=question,
        k=top_k
    )

    logger.info(f"Vector store retornou {len(results)} chunks")

    # 3. Expandir contexto com vizinhos (SE ATIVADO)
    expanded_results = results[:]
    original_indices = set(range(len(results)))

    if expand_neighbors and len(results) > 0:
        logger.info(f"🔧 Expandindo contexto com vizinhos (n_before={n_before}, n_after={n_after})...")

        # Para cada chunk retornado, buscar vizinhos
        neighbor_docs = []
        for doc, score in results:
            metadata = doc.metadata
            page = metadata.get("page", 0)
            doc_id = metadata.get("document_id", "")

            # Buscar chunks vizinhos da mesma página/documento
            # Usar filtro de metadados para encontrar vizinhos
            for offset in range(-n_before, n_after + 1):
                if offset == 0:
                    continue  # Pular o chunk atual

                try:
                    # Buscar chunks próximos (mesma página ou páginas vizinhas)
                    neighbor_page = page + (offset // 3)  # Aproximação: 3 chunks por página

                    neighbor_results = vectorstore.similarity_search(
                        query=question,
                        k=1,
                        filter={"page": neighbor_page, "document_id": doc_id}
                    )

                    if neighbor_results:
                        neighbor_docs.append((neighbor_results[0], score * 0.8, True))  # Score reduzido, is_neighbor=True

                except Exception as e:
                    logger.debug(f"Erro ao buscar vizinho offset={offset}: {str(e)}")
                    continue

        # Adicionar vizinhos únicos
        existing_ids = {metadata.get("chunk_id", "") for doc, _ in results}
        for neighbor_doc, neighbor_score, is_neighbor in neighbor_docs:
            neighbor_id = neighbor_doc.metadata.get("chunk_id", "")
            if neighbor_id and neighbor_id not in existing_ids:
                expanded_results.append((neighbor_doc, neighbor_score))
                existing_ids.add(neighbor_id)

        logger.info(f"✅ Expandido: {len(results)} → {len(expanded_results)} chunks")

    # 4. Processar resultados e carregar imagens
    context_chunks = []

    for idx, (doc, score) in enumerate(expanded_results):
        metadata = doc.metadata
        is_neighbor = idx not in original_indices

        chunk_data = {
            "id": metadata.get("chunk_id", "unknown"),
            "text": doc.page_content,
            "metadata": metadata,
            "score": score,
            "images": [],
            "is_neighbor": is_neighbor
        }

        # Carregar imagens se houver
        if metadata.get("has_images"):
            image_ids_str = metadata.get("image_ids", "")
            if image_ids_str:
                # Converter CSV string para lista
                image_ids = image_ids_str.split(",")
                images = get_images_by_ids(image_ids, db)
                chunk_data["images"] = images

                logger.info(f"Chunk tem {len(images)} imagem(ns)")

        context_chunks.append(chunk_data)

    logger.info(f"✅ Query processada: {len(context_chunks)} chunks retornados")

    result = {
        "question": question,
        "chunks": context_chunks,
        "total_chunks": len(context_chunks)
    }

    if expand_neighbors:
        result["original_chunks"] = len(results)
        result["expanded_chunks"] = len(context_chunks)

    return result


def format_context_for_llm(query_result: Dict) -> Dict:
    """
    Formata o resultado da query para envio ao LLM multimodal.

    Agrupa texto e imagens em um formato estruturado para o prompt.

    Args:
        query_result: Resultado da função query_documents()

    Returns:
        {
            "question": "...",
            "context_text": "Contexto combinado de todos os chunks",
            "images": [{"id": "...", "data": "base64...", "page": 5}],
            "sources": ["manual.pdf - página 5", ...]
        }
    """
    context_text = ""
    all_images = []
    sources = []

    for chunk in query_result["chunks"]:
        # Adicionar texto
        context_text += f"\n\n{chunk['text']}"

        # Coletar imagens
        if chunk["images"]:
            all_images.extend(chunk["images"])

        # Coletar fontes
        source = f"{chunk['metadata'].get('source_file', 'Unknown')} - página {chunk['metadata'].get('page', '?')}"
        if source not in sources:
            sources.append(source)

    return {
        "question": query_result["question"],
        "context_text": context_text.strip(),
        "images": all_images,
        "sources": sources
    }
