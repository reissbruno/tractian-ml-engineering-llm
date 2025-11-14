"""
Exemplo de Uso da API RAG com Docling

Este script demonstra como usar a API via requests Python.
"""

import requests
import json
import base64
from pathlib import Path


BASE_URL = "http://localhost:8000"


def upload_pdf(pdf_path: str):
    """
    Faz upload de um PDF para processamento.

    Args:
        pdf_path: Caminho do arquivo PDF

    Returns:
        Response JSON com informações do upload
    """
    url = f"{BASE_URL}/documents"

    with open(pdf_path, "rb") as f:
        files = {"files": (Path(pdf_path).name, f, "application/pdf")}
        response = requests.post(url, files=files)

    if response.status_code == 200:
        data = response.json()
        print(f"✅ Upload bem-sucedido!")
        print(f"   Documentos indexados: {data['documents_indexed']}")
        print(f"   Total de chunks: {data['total_chunks']}")
        return data
    else:
        print(f"❌ Erro no upload: {response.status_code}")
        print(response.text)
        return None


def list_documents():
    """
    Lista todos os documentos processados.

    Returns:
        Lista de documentos
    """
    url = f"{BASE_URL}/documents"
    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        print(f"\n📚 Documentos disponíveis ({data['total']}):")
        for doc in data["documents"]:
            print(f"\n   • {doc['filename']}")
            print(f"     ID: {doc['id']}")
            print(f"     Status: {doc['status']}")
            print(f"     Chunks: {doc['chunks']}")
            print(f"     Data: {doc['created_at']}")
        return data["documents"]
    else:
        print(f"❌ Erro ao listar: {response.status_code}")
        return []


def ask_question(question: str):
    """
    Faz uma pergunta sobre os documentos indexados.

    Args:
        question: Pergunta em linguagem natural

    Returns:
        Resposta da API
    """
    url = f"{BASE_URL}/question"
    payload = {"question": question}

    response = requests.post(
        url,
        headers={"Content-Type": "application/json"},
        data=json.dumps(payload)
    )

    if response.status_code == 200:
        data = response.json()
        print(f"\n🤖 RESPOSTA:")
        print(f"{data['answer']}\n")

        print(f"📖 REFERÊNCIAS ({len(data['references'])}):")
        for i, ref in enumerate(data["references"], 1):
            print(f"\n   [{i}] {ref[:150]}...")

        return data
    else:
        print(f"❌ Erro na query: {response.status_code}")
        print(response.text)
        return None


def save_images_from_response(response_data: dict, output_dir: str = "output_images"):
    """
    Salva as imagens retornadas pela API em disco.

    Args:
        response_data: Dados da resposta da API
        output_dir: Diretório de saída
    """
    import os

    # Criar diretório se não existir
    os.makedirs(output_dir, exist_ok=True)

    # Verificar se há contexto com imagens (isso será implementado quando integrar com LLM)
    # Por enquanto, este é um exemplo de como processar as imagens quando disponíveis

    print(f"\n💾 Imagens seriam salvas em: {output_dir}/")
    print("   (Funcionalidade será ativada após integração com LLM multimodal)")


def main():
    """Exemplo de uso completo da API."""

    print("="*60)
    print("🚀 EXEMPLO DE USO DA API RAG COM DOCLING")
    print("="*60)

    # 1. Upload de documento
    print("\n📤 PASSO 1: Upload de PDF")
    print("-" * 60)

    pdf_path = "arquivo_teste/sample.pdf"  # ← Ajuste o caminho
    result = upload_pdf(pdf_path)

    if not result:
        print("\n⚠️  Não foi possível fazer upload. Verifique se:")
        print("   1. O servidor está rodando (uvicorn server:app)")
        print("   2. O arquivo PDF existe no caminho especificado")
        return

    # 2. Listar documentos
    print("\n\n📋 PASSO 2: Listar Documentos")
    print("-" * 60)
    list_documents()

    # 3. Fazer perguntas
    print("\n\n❓ PASSO 3: Fazer Perguntas")
    print("-" * 60)

    questions = [
        "Qual é o principal assunto deste documento?",
        "Existem tabelas ou figuras? Descreva-as.",
        "Quais são as principais recomendações?"
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n[Pergunta {i}] {question}")
        print("-" * 60)
        ask_question(question)

    print("\n" + "="*60)
    print("✅ EXEMPLO CONCLUÍDO")
    print("="*60)


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("\n❌ Erro: Não foi possível conectar ao servidor")
        print("💡 Inicie o servidor com: uvicorn server:app --reload")
    except FileNotFoundError as e:
        print(f"\n❌ Erro: Arquivo não encontrado - {e}")
        print("💡 Ajuste o caminho do PDF em 'pdf_path'")
    except Exception as e:
        print(f"\n❌ Erro inesperado: {e}")
