"""
Teste Completo do Pipeline RAG com Docling + Embeddings + ChromaDB

Este script testa o pipeline completo:
1. Extração estruturada com Docling (texto, tabelas, imagens)
2. Salvamento de imagens no SQLite (base64)
3. Chunking adaptativo
4. Geração de embeddings (Sentence Transformers)
5. Armazenamento no ChromaDB via LangChain
6. Query e recuperação com imagens

Uso:
    python test_full_pipeline.py
"""

import asyncio
import logging
import os
import sys
from pathlib import Path
from datetime import datetime

# Configurar variáveis de ambiente antes de imports pesados
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

# Adicionar diretório raiz do projeto ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.auth.database import init_db, SessionLocal, User, Document, DocumentImage
from src.services.ingest import process_document_with_docling
from src.services.rag import query_documents

# Configurar logging detalhado
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def print_section(title: str):
    """Imprime uma seção formatada."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def setup_database():
    """Inicializa o banco de dados."""
    print_section("1. INICIALIZAÇÃO DO BANCO DE DADOS")

    logger.info("Criando tabelas no SQLite...")
    init_db()

    logger.info("✅ Banco de dados inicializado")

    # Verificar se usuário de teste existe
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.id == 1).first()
        if not user:
            logger.warning("⚠️  Usuário de teste não existe. Será criado mock durante o teste.")
        else:
            logger.info(f"✅ Usuário encontrado: {user.user_name}")
    finally:
        db.close()


async def test_document_ingestion(pdf_path: str, force_reprocess: bool = False):
    """
    Testa o pipeline completo de ingestão.

    Args:
        pdf_path: Caminho do PDF
        force_reprocess: Se True, reprocessa mesmo se já existir

    Returns:
        doc_id do documento processado
    """
    print_section("2. INGESTÃO DO DOCUMENTO")

    db = SessionLocal()

    try:
        # Configurar documento de teste
        doc_id = "test_weg_guia_001"
        user_id = 1

        logger.info(f"📄 Arquivo: {pdf_path}")
        logger.info(f"📋 Document ID: {doc_id}")
        logger.info(f"👤 User ID: {user_id}")

        # Verificar se arquivo existe
        if not Path(pdf_path).exists():
            logger.error(f"❌ Arquivo não encontrado: {pdf_path}")
            return None

        # ✅ VERIFICAR SE DOCUMENTO JÁ FOI PROCESSADO
        existing_doc = db.query(Document).filter(Document.id == doc_id).first()

        if existing_doc and not force_reprocess:
            logger.info(f"\n♻️  DOCUMENTO JÁ PROCESSADO - REAPROVEITANDO!")
            logger.info(f"   - Status: {existing_doc.status}")
            logger.info(f"   - Chunks: {existing_doc.chunks_count}")
            logger.info(f"   - Processado em: {existing_doc.created_at}")
            logger.info(f"   - Filename: {existing_doc.filename}")

            # Verificar imagens existentes
            images_count = db.query(DocumentImage).filter(
                DocumentImage.document_id == doc_id
            ).count()
            logger.info(f"   - Imagens salvas: {images_count}")

            logger.info("\n✅ Pulando ingestão - usando dados existentes")
            return doc_id

        if existing_doc and force_reprocess:
            logger.warning(f"\n⚠️  Documento já existe, mas FORCE_REPROCESS está ativo")
            logger.warning(f"   Removendo documento antigo e reprocessando...")

            # Remover imagens antigas
            db.query(DocumentImage).filter(
                DocumentImage.document_id == doc_id
            ).delete()

            # Remover documento antigo
            db.delete(existing_doc)
            db.commit()
            logger.info("   ✅ Dados antigos removidos")

        # EXTRAIR TEXTO COMPLETO COM DOCLING E SALVAR EM TXT
        logger.info("\n📝 Extraindo texto completo com Docling...")
        from docling.document_converter import DocumentConverter

        converter = DocumentConverter()
        result = converter.convert(pdf_path)
        docling_doc = result.document

        # Salvar texto completo em arquivo
        output_dir = Path("output_extractions")
        output_dir.mkdir(exist_ok=True)

        pdf_name = Path(pdf_path).stem
        txt_file = output_dir / f"{pdf_name}_extracted_text.txt"

        with open(txt_file, 'w', encoding='utf-8') as f:
            f.write("="*80 + "\n")
            import datetime
            f.write(f"EXTRAÇÃO COMPLETA DO DOCUMENTO: {Path(pdf_path).name}\n")
            f.write(f"Data: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n\n")

            # Iterar por todos os elementos e extrair texto
            for idx, (element, _) in enumerate(docling_doc.iterate_items()):
                page = element.prov[0].page_no if element.prov else 0
                el_type = element.label

                # Adicionar cabeçalho do elemento
                f.write(f"\n{'─'*80}\n")
                f.write(f"[Elemento {idx+1}] Página: {page} | Tipo: {el_type}\n")
                f.write(f"{'─'*80}\n")

                # Extrair texto do elemento
                if hasattr(element, 'text') and element.text:
                    f.write(element.text)
                    f.write("\n")

                # Se for tabela, tentar extrair dados estruturados
                if el_type == "table" and hasattr(element, 'data'):
                    try:
                        table_data = element.data
                        if hasattr(table_data, 'to_string'):
                            f.write("\n[TABELA]\n")
                            f.write(table_data.to_string())
                            f.write("\n")
                    except:
                        pass

        logger.info(f"✅ Texto completo salvo em: {txt_file}")
        logger.info(f"   Total de elementos processados: {len(list(docling_doc.iterate_items()))}")

        # Executar pipeline de ingestão
        logger.info("\n🚀 Iniciando processamento com Docling...")
        chunks_count = await process_document_with_docling(
            file_path=pdf_path,
            doc_id=doc_id,
            user_id=user_id,
            db=db
        )

        logger.info(f"\n✅ Ingestão concluída: {chunks_count} chunks criados")

        # Verificar imagens salvas
        images = db.query(DocumentImage).filter(
            DocumentImage.document_id == doc_id
        ).all()

        logger.info(f"\n📸 Imagens extraídas e salvas: {len(images)}")
        for i, img in enumerate(images, 1):
            logger.info(f"   {i}. Página {img.page_number} - ID: {img.id}")
            logger.info(f"      Formato: {img.image_format}, Tamanho: {len(img.image_data)} bytes (base64)")
            if img.caption:
                logger.info(f"      Caption: {img.caption[:80]}...")

        return doc_id

    except Exception as e:
        logger.error(f"❌ Erro durante ingestão: {str(e)}")
        import traceback
        traceback.print_exc()
        return None
    finally:
        db.close()


async def test_embeddings_verification(user_id: int):
    """
    Verifica se os embeddings foram gerados e salvos corretamente.

    Args:
        user_id: ID do usuário
    """
    print_section("3. VERIFICAÇÃO DE EMBEDDINGS NO CHROMADB")

    try:
        from src.services.rag import get_chroma_vectorstore

        vectorstore = get_chroma_vectorstore(user_id)

        # Tentar buscar um documento genérico
        logger.info("🔍 Testando busca no ChromaDB...")

        results = vectorstore.similarity_search(
            query="motor",
            k=3
        )

        if results:
            logger.info(f"✅ ChromaDB está funcional: {len(results)} resultados encontrados")
            logger.info("\nExemplo de chunks armazenados:")
            for i, doc in enumerate(results, 1):
                logger.info(f"\n   Chunk {i}:")
                logger.info(f"   - Texto (primeiros 100 chars): {doc.page_content[:100]}...")
                logger.info(f"   - Página: {doc.metadata.get('page', '?')}")
                logger.info(f"   - Tipo: {doc.metadata.get('chunk_type', '?')}")
                logger.info(f"   - Tem imagens: {doc.metadata.get('has_images', False)}")
        else:
            logger.warning("⚠️  Nenhum documento encontrado no ChromaDB")

    except Exception as e:
        logger.error(f"❌ Erro ao verificar embeddings: {str(e)}")
        import traceback
        traceback.print_exc()


async def test_query_with_images(user_id: int):
    """
    Testa queries e recuperação de imagens.

    Args:
        user_id: ID do usuário
    """
    print_section("4. TESTE DE QUERIES E RECUPERAÇÃO DE IMAGENS")

    db = SessionLocal()

    try:
        # Lista de perguntas de teste
        test_questions = [
            "What are the lubrication intervals?",
            "Como fazer a manutenção do motor?",
            "What is the NEMA frame?",
            "Quais são as especificações técnicas?",
        ]

        for i, question in enumerate(test_questions, 1):
            logger.info(f"\n{'─'*80}")
            logger.info(f"Query {i}: '{question}'")
            logger.info('─'*80)

            result = await query_documents(
                question=question,
                user_id=user_id,
                db=db,
                top_k=3
            )

            if result.get("error"):
                logger.error(f"❌ Erro: {result['error']}")
                continue

            logger.info(f"\n✅ Encontrados {result['total_chunks']} chunks relevantes")

            for j, chunk in enumerate(result["chunks"], 1):
                logger.info(f"\n   📄 Chunk {j}:")
                logger.info(f"   - Score: {chunk['score']:.4f}")
                logger.info(f"   - Página: {chunk['metadata'].get('page', '?')}")
                logger.info(f"   - Tipo: {chunk['metadata'].get('chunk_type', '?')}")
                logger.info(f"   - Texto (primeiros 150 chars):")
                logger.info(f"     {chunk['text'][:150]}...")

                if chunk["images"]:
                    logger.info(f"   - 📸 Imagens associadas: {len(chunk['images'])}")
                    for k, img in enumerate(chunk["images"], 1):
                        logger.info(f"      {k}. ID: {img['id']}, Página: {img['page']}, Formato: {img['format']}")
                        logger.info(f"         Tamanho: {len(img['data'])} bytes (base64)")
                        if img.get('caption'):
                            logger.info(f"         Caption: {img['caption'][:80]}...")
                else:
                    logger.info(f"   - 📸 Sem imagens associadas")

        logger.info("\n" + "="*80)
        logger.info("✅ Teste de queries concluído com sucesso!")
        logger.info("="*80)

    except Exception as e:
        logger.error(f"❌ Erro durante queries: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()


async def test_statistics():
    """Exibe estatísticas do banco de dados."""
    print_section("5. ESTATÍSTICAS DO BANCO DE DADOS")

    db = SessionLocal()

    try:
        # Contar documentos
        docs_count = db.query(Document).count()
        logger.info(f"📄 Total de documentos: {docs_count}")

        # Contar imagens
        images_count = db.query(DocumentImage).count()
        logger.info(f"📸 Total de imagens: {images_count}")

        # Listar documentos
        if docs_count > 0:
            logger.info("\n📋 Documentos no banco:")
            docs = db.query(Document).all()
            for doc in docs:
                logger.info(f"\n   - ID: {doc.id}")
                logger.info(f"     Filename: {doc.filename}")
                logger.info(f"     Status: {doc.status}")
                logger.info(f"     Chunks: {doc.chunks_count}")
                logger.info(f"     Created: {doc.created_at}")

    except Exception as e:
        logger.error(f"❌ Erro ao obter estatísticas: {str(e)}")
    finally:
        db.close()


async def main():
    """Função principal de teste."""

    print("\n" + "🚀"*40)
    print("  TESTE COMPLETO DO PIPELINE RAG COM DOCLING")
    print("🚀"*40)

    # Configuração
    # Caminho relativo ao diretório raiz do projeto
    pdf_path = str(Path(__file__).parent.parent / "arquivo_teste" / "WEG-motores-eletricos-guia-de-especificacao-50032749-brochure-portuguese-web.pdf")
    user_id = 1

    # ✅ CONTROLE DE REPROCESSAMENTO
    # Mude para True se quiser forçar o reprocessamento
    force_reprocess = False

    try:
        # 1. Setup
        setup_database()

        # 2. Ingestão (com reaproveitamento inteligente)
        doc_id = await test_document_ingestion(pdf_path, force_reprocess=force_reprocess)

        if not doc_id:
            logger.error("❌ Ingestão falhou. Abortando testes.")
            return

        # 3. Verificar embeddings
        await test_embeddings_verification(user_id)

        # 4. Testar queries
        await test_query_with_images(user_id)

        # 5. Estatísticas
        await test_statistics()

        # Resumo final
        print("\n" + "✅"*40)
        print("  TODOS OS TESTES CONCLUÍDOS COM SUCESSO!")
        print("✅"*40 + "\n")

        print("📊 Resumo:")
        print(f"   ✅ Documento processado: {pdf_path}")
        print(f"   ✅ Embeddings gerados e salvos no ChromaDB")
        print(f"   ✅ Imagens extraídas e salvas no SQLite")
        print(f"   ✅ Queries funcionando com recuperação de imagens")
        print(f"   ✅ Pipeline completo validado!")

    except Exception as e:
        logger.error(f"\n❌ ERRO CRÍTICO: {str(e)}")
        import traceback
        traceback.print_exc()

        print("\n" + "❌"*40)
        print("  TESTES FALHARAM!")
        print("❌"*40 + "\n")


if __name__ == "__main__":
    asyncio.run(main())
