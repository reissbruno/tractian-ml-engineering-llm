"""
Teste do Adaptive Chunker com PDF Real

Este teste processa o PDF LB5001.pdf usando o sistema de chunking adaptativo
e mostra estatísticas sobre os tipos de blocos detectados.

Requisitos:
    pip install langchain-community pdfminer.six langchain-text-splitters

Execução:
    python tests/test_adaptive_chunker.py
"""

import sys
from pathlib import Path

# Adicionar raiz do projeto ao path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.services.adaptive_chunker import (
    split_text_dynamic,
)

# Importar loader de PDF do LangChain
try:
    from langchain_community.document_loaders import PDFMinerLoader
    PDFMINER_AVAILABLE = True
except ImportError:
    print("ERRO: langchain-community não instalado.")
    print("Instale com: pip install langchain-community pdfminer.six")
    sys.exit(1)


def test_pdf_with_adaptive_chunker():
    """Processa o PDF LB5001.pdf com chunking adaptativo."""

    pdf_path = Path(__file__).parent.parent / "arquivo_teste" / "LB5001.pdf"

    if not pdf_path.exists():
        print(f"ERRO: PDF não encontrado em {pdf_path}")
        sys.exit(1)

    print("\n" + "=" * 80)
    print(f"PROCESSANDO PDF: {pdf_path.name}")
    print("=" * 80)

    try:
        # Carregar PDF com PDFMinerLoader
        print("\n[1/3] Carregando PDF...")
        loader = PDFMinerLoader(str(pdf_path))
        docs = loader.load()

        num_pages = len(docs)
        total_pages = docs[0].metadata.get('total_pages', num_pages)
        print(f"      Paginas carregadas: {num_pages}")
        print(f"      Total de paginas: {total_pages}")

        # Processar todas as páginas
        print("\n[2/3] Processando paginas com chunking adaptativo...")

        all_chunks = []
        page_stats = {}
        block_type_global = {}

        for page_idx, doc in enumerate(docs):
            page_num = page_idx + 1
            text = doc.page_content

            if not text.strip():
                continue

            # Dividir em blocos (por parágrafos duplos)
            blocks = [b.strip() for b in text.split("\n\n") if b.strip()]

            for block in blocks:
                if len(block) < 20:  # Ignorar blocos muito curtos
                    continue

                # Aplicar chunking adaptativo
                block_type, chunks = split_text_dynamic(block)

                # Estatísticas
                block_type_global[block_type] = block_type_global.get(block_type, 0) + 1

                for chunk in chunks:
                    chunk_data = {
                        "content": chunk,
                        "metadata": {
                            "source_pdf": pdf_path.name,
                            "page": page_num,
                            "chunk_type": block_type,
                            **doc.metadata
                        }
                    }
                    all_chunks.append(chunk_data)

        # Mostrar resultados
        print(f"\n[3/3] Resultados:")
        print(f"\n{'=' * 80}")
        print("ESTATISTICAS DO PROCESSAMENTO")
        print("=" * 80)

        print(f"\nTotal de chunks gerados: {len(all_chunks)}")
        print(f"\nDistribuicao por tipo de bloco:")
        print("-" * 50)

        for block_type, count in sorted(block_type_global.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(all_chunks) * 100) if all_chunks else 0
            bar = "█" * int(percentage / 2)
            print(f"{block_type:15s}: {count:4d} ({percentage:5.1f}%) {bar}")

        # Mostrar exemplos de cada tipo
        print(f"\n{'=' * 80}")
        print("EXEMPLOS DE CHUNKS POR TIPO")
        print("=" * 80)

        types_shown = set()
        for chunk_data in all_chunks:
            chunk_type = chunk_data["metadata"]["chunk_type"]
            if chunk_type not in types_shown:
                types_shown.add(chunk_type)
                print(f"\n[{chunk_type.upper()}] - Pagina {chunk_data['metadata']['page']}")
                print("-" * 80)
                preview = chunk_data["content"][:200].replace("\n", " ")
                print(f"{preview}...")
                print()

        print("=" * 80)
        print("PROCESSAMENTO CONCLUIDO COM SUCESSO!")
        print("=" * 80)

        return True

    except Exception as e:
        print(f"\nERRO ao processar PDF: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = test_pdf_with_adaptive_chunker()
    sys.exit(0 if success else 1)
