import logging
import os
import uuid
from datetime import datetime
from fastapi import FastAPI, File, UploadFile, Depends, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from typing import List
from contextlib import asynccontextmanager
from src.models import QuestionRequest, DocumentsResponse, QuestionResponse, RegisterRequest, LoginRequest, TokenResponse
from src.auth.database import get_db, init_db, User, Document
from src.auth.auth import hash_password, verify_password, create_access_token
from src.services.ingest import process_document_with_docling
from src.services.rag import query_documents, format_context_for_llm

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gerencia o ciclo de vida da aplicação (startup e shutdown).
    """
    # Startup
    logger.info("Starting Tractian RAG application...")
    init_db()
    logger.info("Application startup complete.")
    yield
    # Shutdown
    logger.info("Shutting down Tractian RAG application...")


app = FastAPI(lifespan=lifespan)

# Servir arquivos estáticos
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
async def root():
    """
    Redireciona para a tela de login.
    """
    return RedirectResponse(url="/static/login.html")


@app.post("/register")
async def register(request: RegisterRequest, db: Session = Depends(get_db)):
    """
    Registra um novo usuário.
    """
    existing_user = db.query(User).filter(User.user_name == request.user_name).first()
    if existing_user:
        raise HTTPException(status_code=400, detail="Usuário já existe")

    hashed_pwd = hash_password(request.senha)
    new_user = User(user_name=request.user_name, hashed_password=hashed_pwd)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    return {"message": "Usuário criado com sucesso"}


@app.post("/login", response_model=TokenResponse)
async def login(request: LoginRequest, db: Session = Depends(get_db)):
    """
    Faz login e retorna um token JWT.
    """
    user = db.query(User).filter(User.user_name == request.user_name).first()
    if not user or not verify_password(request.senha, user.hashed_password):
        raise HTTPException(status_code=401, detail="Credenciais inválidas")

    access_token = create_access_token(data={"sub": user.user_name})
    return TokenResponse(access_token=access_token, token_type="bearer")


@app.get("/documents")
async def list_documents(
    db: Session = Depends(get_db),
    current_user: User = Depends(lambda: User(id=1, user_name="test"))  # TODO: Implementar autenticação real
):
    """
    Lista todos os documentos enviados pelo usuário.
    """
    docs = db.query(Document).filter(Document.user_id == current_user.id).all()

    return {
        "documents": [
            {
                "id": doc.id,
                "filename": doc.filename,
                "status": doc.status,
                "chunks": doc.chunks_count,
                "created_at": doc.created_at.isoformat() if doc.created_at else None
            }
            for doc in docs
        ],
        "total": len(docs)
    }


@app.post("/documents", response_model=DocumentsResponse)
async def upload_documents(
    files: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
    current_user: User = Depends(lambda: User(id=1, user_name="test"))  # TODO: Implementar autenticação real
):
    """
    Faz upload de um ou mais documentos PDF e processa com Docling.
    """
    uploaded_docs = []

    for file in files:
        # Validação
        if file.content_type != 'application/pdf':
            raise HTTPException(status_code=400, detail=f"Arquivo {file.filename} não é PDF")

        # Gerar ID e caminho
        doc_id = str(uuid.uuid4())
        upload_dir = f"uploads/user_{current_user.id}"
        os.makedirs(upload_dir, exist_ok=True)
        file_path = os.path.join(upload_dir, f"{doc_id}_{file.filename}")

        # Salvar arquivo
        logger.info(f"Salvando arquivo: {file_path}")
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Registrar no banco
        doc = Document(
            id=doc_id,
            user_id=current_user.id,
            filename=file.filename,
            file_path=file_path,
            file_size=len(content),
            status="processing",
            created_at=datetime.utcnow()
        )
        db.add(doc)
        db.commit()

        try:
            # Processar com Docling (com semantic chunking ativado)
            logger.info(f"Processando documento {doc_id} com Docling...")
            chunks_count = await process_document_with_docling(
                file_path=file_path,
                doc_id=doc_id,
                user_id=current_user.id,
                db=db,
                use_semantic_chunking=True
            )

            # Atualizar status
            doc.status = "completed"
            doc.chunks_count = chunks_count
            doc.processed_at = datetime.utcnow()
            db.commit()

            logger.info(f"✅ Documento {doc_id} processado: {chunks_count} chunks")

            uploaded_docs.append({
                "id": doc_id,
                "filename": file.filename,
                "chunks": chunks_count
            })

        except Exception as e:
            logger.error(f"❌ Erro ao processar documento {doc_id}: {str(e)}")
            doc.status = "error"
            doc.error_message = str(e)
            db.commit()
            raise HTTPException(status_code=500, detail=f"Erro ao processar {file.filename}: {str(e)}")

    total_chunks = sum(d["chunks"] for d in uploaded_docs)

    return DocumentsResponse(
        message="Documents processed successfully",
        documents_indexed=len(uploaded_docs),
        total_chunks=total_chunks
    )


@app.post("/question", response_model=QuestionResponse)
async def ask_question(
    request: QuestionRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(lambda: User(id=1, user_name="test"))  # TODO: Implementar autenticação real
):
    """
    Faz uma pergunta sobre os documentos indexados e recupera chunks com imagens.
    """
    logger.info(f"Query recebida: {request.question}")

    try:
        # Buscar contexto relevante
        result = await query_documents(
            question=request.question,
            user_id=current_user.id,
            db=db,
            top_k=5
        )

        if result["total_chunks"] == 0:
            return QuestionResponse(
                answer="Não encontrei informações relevantes nos documentos indexados.",
                references=[]
            )

        # Formatar contexto para LLM (futuro)
        formatted = format_context_for_llm(result)

        # TODO: Enviar para LLM multimodal (GPT-4o, Claude 3.5 Sonnet, etc)
        # Por enquanto, retornar contexto recuperado

        answer = f"[Contexto recuperado com sucesso]\n\nEncontrei {result['total_chunks']} trechos relevantes.\n\n"
        answer += f"Imagens associadas: {len(formatted['images'])}\n\n"
        answer += f"Fontes: {', '.join(formatted['sources'])}\n\n"

        # Adicionar scores de similaridade
        answer += "Scores de similaridade:\n"
        for i, chunk in enumerate(result["chunks"], 1):
            score = chunk.get('score', 0)
            answer += f"  {i}. Score: {score:.4f}\n"

        references = [chunk["text"][:200] + "..." for chunk in result["chunks"]]

        return QuestionResponse(
            answer=answer,
            references=references
        )

    except Exception as e:
        logger.error(f"Erro ao processar query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
