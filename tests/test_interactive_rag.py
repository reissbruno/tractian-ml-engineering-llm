"""
Teste Interativo do Sistema RAG

Script interativo para fazer perguntas ao sistema RAG e ver:
- Chunks de texto recuperados
- Imagens associadas
- Scores de similaridade
- Metadados completos

Uso:
    python tests/test_interactive_rag.py
"""

import asyncio
import base64
import logging
import os
import sys
from pathlib import Path

# Configurar variáveis de ambiente
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"  # Modo offline para não tentar baixar modelos
os.environ["TRANSFORMERS_OFFLINE"] = "1"

# Adicionar diretório raiz do projeto ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.auth.database import SessionLocal
from src.services.rag import query_documents

# Configurar logging
logging.basicConfig(
    level=logging.WARNING,  # Menos verboso para interação
    format='%(levelname)s:%(name)s:%(message)s'
)
logger = logging.getLogger(__name__)


def print_separator(char="─", length=80):
    """Imprime uma linha separadora."""
    print(char * length)


def print_header(text):
    """Imprime um cabeçalho destacado."""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80)


def display_results(result):
    """
    Exibe os resultados da query de forma formatada.

    Args:
        result: Dicionário com question e chunks
    """
    print_header(f"PERGUNTA: {result['question']}")

    chunks = result.get('chunks', [])

    if not chunks:
        print("\n❌ Nenhum resultado encontrado.")
        return

    print(f"\n✅ Encontrados {len(chunks)} chunks relevantes\n")

    for idx, chunk_data in enumerate(chunks, 1):
        print(f"\n{'─'*80}")
        print(f"📄 Chunk {idx}")
        print(f"{'─'*80}")

        # Metadados básicos
        metadata = chunk_data.get('metadata', {})
        print(f"📊 Score de similaridade: {chunk_data.get('score', 'N/A'):.4f}")
        print(f"📖 Página: {metadata.get('page', 'N/A')}")
        print(f"🏷️  Tipo: {metadata.get('chunk_type', 'N/A')}")

        # Texto do chunk COMPLETO
        content = chunk_data.get('content', chunk_data.get('text', ''))
        print(f"\n📝 Texto completo:")
        print(f"{'─'*80}")
        if content:
            print(content)
        else:
            print("⚠️  Sem conteúdo de texto")
        print(f"{'─'*80}")

        # Imagens associadas com BASE64
        images = chunk_data.get('images', [])
        if images:
            print(f"\n📸 Imagens associadas: {len(images)}")
            for img_idx, img in enumerate(images, 1):
                # Tentar diferentes chaves para image_data
                img_data = img.get('image_data', img.get('data', ''))
                img_size_kb = len(img_data) / 1024 if img_data else 0

                print(f"\n   {img_idx}. ID: {img.get('id', 'N/A')}")
                print(f"      Página: {img.get('page_number', img.get('page', 'N/A'))}")
                print(f"      Formato: {img.get('image_format', img.get('format', 'N/A'))}")
                print(f"      Tamanho: {img_size_kb:.2f} KB")

                if img.get('caption'):
                    print(f"      Caption: {img.get('caption', '')}")

                # Mostrar BASE64 (primeiros e últimos caracteres)
                if img_data:
                    print(f"\n      🔐 Base64 da imagem:")
                    print(f"      {img_data[:100]}...")
                    print(f"      ...{img_data[-100:]}")
                    print(f"      (Total: {len(img_data)} caracteres)")
                else:
                    print(f"      ⚠️  Sem dados de imagem")
        else:
            print("\n📸 Sem imagens associadas")

    print("\n" + "="*80)


async def interactive_query_loop():
    """Loop interativo de perguntas."""

    print("\n" + "🚀"*40)
    print("  SISTEMA RAG INTERATIVO - TRACTIAN ML")
    print("🚀"*40)

    print("\n📚 Sistema pronto para responder perguntas!")
    print("💡 Dicas:")
    print("   - Faça perguntas em português ou inglês")
    print("   - Digite 'sair' ou 'exit' para encerrar")
    print("   - Digite 'config' para ajustar parâmetros")
    print("   - Digite 'stats' para ver estatísticas do banco")
    print("   - Digite 'save' para salvar imagens do último resultado")

    # Configurações padrão
    user_id = 1
    top_k = 3
    last_result = None  # Guardar último resultado para salvar imagens

    # Criar sessão do banco
    db = SessionLocal()

    try:
        while True:
            print_separator("─")
            question = input("\n❓ Sua pergunta: ").strip()

            if not question:
                continue

            # Comandos especiais
            if question.lower() in ['sair', 'exit', 'quit', 'q']:
                print("\n👋 Encerrando sistema RAG. Até logo!")
                break

            elif question.lower() == 'config':
                print("\n⚙️  CONFIGURAÇÕES ATUAIS:")
                print(f"   User ID: {user_id}")
                print(f"   Top-K resultados: {top_k}")

                try:
                    new_top_k = input(f"\nNovo Top-K [{top_k}]: ").strip()
                    if new_top_k:
                        top_k = int(new_top_k)
                        print(f"✅ Top-K atualizado para {top_k}")
                except ValueError:
                    print("❌ Valor inválido, mantendo configuração anterior")
                continue

            elif question.lower() == 'stats':
                print("\n📊 ESTATÍSTICAS DO BANCO DE DADOS:")

                # Contar documentos
                from src.auth.database import Document, DocumentImage
                doc_count = db.query(Document).count()
                img_count = db.query(DocumentImage).count()

                print(f"   📄 Documentos processados: {doc_count}")
                print(f"   📸 Imagens armazenadas: {img_count}")

                # Listar documentos
                if doc_count > 0:
                    print(f"\n   📋 Documentos:")
                    docs = db.query(Document).limit(10).all()
                    for doc in docs:
                        print(f"      - {doc.filename} ({doc.chunks_count} chunks)")

                continue

            elif question.lower() == 'save':
                if not last_result:
                    print("\n⚠️  Nenhum resultado para salvar. Faça uma pergunta primeiro!")
                    continue

                print("\n💾 Salvando imagens do último resultado...")
                saved_count = 0

                for chunk_idx, chunk_data in enumerate(last_result.get('chunks', []), 1):
                    images = chunk_data.get('images', [])
                    for img_idx, img in enumerate(images, 1):
                        img_data = img.get('image_data', '')
                        if img_data:
                            # Criar diretório se não existir
                            output_dir = Path("output_images")
                            output_dir.mkdir(exist_ok=True)

                            # Nome do arquivo
                            img_format = img.get('image_format', 'png')
                            filename = f"chunk{chunk_idx}_img{img_idx}_{img.get('id', 'unknown')[:8]}.{img_format}"
                            filepath = output_dir / filename

                            # Decodificar e salvar
                            image_bytes = base64.b64decode(img_data)
                            filepath.write_bytes(image_bytes)

                            saved_count += 1
                            print(f"   ✅ Salva: {filepath}")

                print(f"\n✅ Total de {saved_count} imagem(ns) salva(s) em './output_images/'")
                continue

            # Processar pergunta normal
            print("\n🔍 Buscando informações relevantes...")

            try:
                result = await query_documents(
                    question=question,
                    user_id=user_id,
                    db=db,
                    top_k=top_k
                )

                last_result = result  # Guardar para comando 'save'
                display_results(result)

            except Exception as e:
                print(f"\n❌ Erro ao processar pergunta: {str(e)}")
                logger.error(f"Erro na query: {str(e)}", exc_info=True)

    finally:
        db.close()


async def main():
    """Função principal."""
    try:
        await interactive_query_loop()
    except KeyboardInterrupt:
        print("\n\n👋 Sistema interrompido pelo usuário. Até logo!")
    except Exception as e:
        print(f"\n❌ Erro fatal: {str(e)}")
        logger.error(f"Erro fatal: {str(e)}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
