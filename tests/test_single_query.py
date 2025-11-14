"""
Teste de Query Única do Sistema RAG

Script para fazer uma única pergunta ao sistema RAG.
Útil para testes rápidos ou integração em outros scripts.

Uso:
    python tests/test_single_query.py "Qual é o procedimento de manutenção?"
    python tests/test_single_query.py "What are the motor specifications?" --top-k 5
"""

import asyncio
import argparse
import logging
import os
import sys
from pathlib import Path

# Configurar variáveis de ambiente
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_DISABLE_SYMLINKS"] = "1"

# Adicionar diretório raiz do projeto ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.auth.database import SessionLocal
from src.services.rag import query_documents

# Configurar logging
logging.basicConfig(
    level=logging.WARNING,
    format='%(levelname)s:%(name)s:%(message)s'
)


def display_compact_results(result):
    """
    Exibe os resultados de forma compacta.

    Args:
        result: Dicionário com question e chunks
    """
    print("\n" + "="*80)
    print(f"❓ Pergunta: {result['question']}")
    print("="*80)

    chunks = result.get('chunks', [])

    if not chunks:
        print("\n❌ Nenhum resultado encontrado.")
        return

    print(f"\n✅ {len(chunks)} resultados encontrados\n")

    for idx, chunk_data in enumerate(chunks, 1):
        metadata = chunk_data.get('metadata', {})
        score = chunk_data.get('score', 0)
        content = chunk_data.get('content', '')
        images = chunk_data.get('images', [])

        print(f"[{idx}] Score: {score:.4f} | Página: {metadata.get('page', 'N/A')} | Tipo: {metadata.get('chunk_type', 'N/A')}")
        print(f"    {content[:200]}...")

        if images:
            print(f"    📸 {len(images)} imagem(ns) associada(s)")

        print()


async def single_query(question: str, user_id: int = 1, top_k: int = 3, verbose: bool = False):
    """
    Executa uma única query no sistema RAG.

    Args:
        question: Pergunta a ser feita
        user_id: ID do usuário (padrão: 1)
        top_k: Número de resultados (padrão: 3)
        verbose: Mostrar detalhes completos (padrão: False)

    Returns:
        Dicionário com os resultados
    """
    db = SessionLocal()

    try:
        print(f"\n🔍 Buscando: '{question}'")
        print(f"📊 Parâmetros: user_id={user_id}, top_k={top_k}")

        result = await query_documents(
            question=question,
            user_id=user_id,
            db=db,
            top_k=top_k
        )

        if verbose:
            # Mostrar detalhes completos
            import json
            print("\n📋 Resultado completo (JSON):")
            # Remover image_data para não poluir a saída
            for chunk in result.get('chunks', []):
                for img in chunk.get('images', []):
                    if 'image_data' in img:
                        img['image_data'] = f"<{len(img['image_data'])} bytes>"
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            # Mostrar resumo
            display_compact_results(result)

        return result

    except Exception as e:
        print(f"\n❌ Erro ao processar pergunta: {str(e)}")
        raise

    finally:
        db.close()


def main():
    """Função principal com argumentos de linha de comando."""

    parser = argparse.ArgumentParser(
        description='Faz uma pergunta ao sistema RAG e exibe os resultados.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python tests/test_single_query.py "Como fazer manutenção do motor?"
  python tests/test_single_query.py "What is the NEMA frame?" --top-k 5
  python tests/test_single_query.py "Procedimentos de segurança" --verbose
        """
    )

    parser.add_argument(
        'question',
        type=str,
        nargs='?',
        default=None,
        help='Pergunta a ser feita ao sistema RAG'
    )

    parser.add_argument(
        '--user-id',
        type=int,
        default=1,
        help='ID do usuário (padrão: 1)'
    )

    parser.add_argument(
        '--top-k',
        type=int,
        default=3,
        help='Número de resultados a retornar (padrão: 3)'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Mostrar saída completa em JSON'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Ativar logging debug'
    )

    args = parser.parse_args()

    # Verificar se a pergunta foi fornecida
    if not args.question:
        parser.print_help()
        print("\n❌ Erro: A pergunta é obrigatória.")
        sys.exit(2)

    # Configurar logging se debug ativado
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    # Executar query
    try:
        asyncio.run(single_query(
            question=args.question,
            user_id=args.user_id,
            top_k=args.top_k,
            verbose=args.verbose
        ))
    except KeyboardInterrupt:
        print("\n\n👋 Interrompido pelo usuário.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Erro: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
