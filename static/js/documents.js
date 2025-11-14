// Verificar autenticação
const token = localStorage.getItem('access_token');
const userName = localStorage.getItem('user_name');

if (!token || !userName) {
    window.location.href = '/static/login.html';
}

// Configurar perfil do usuário
document.addEventListener('DOMContentLoaded', () => {
    const userNameElement = document.getElementById('userName');
    const userAvatarElement = document.getElementById('userAvatar');

    if (userNameElement && userName) {
        userNameElement.textContent = userName;
    }

    if (userAvatarElement && userName) {
        userAvatarElement.textContent = userName.charAt(0).toUpperCase();
    }

    // Carregar documentos salvos
    loadDocuments();
});

// Logout
document.getElementById('logoutBtn').addEventListener('click', () => {
    localStorage.removeItem('access_token');
    localStorage.removeItem('user_name');
    window.location.href = '/static/login.html';
});

// Botão de upload (ainda não implementado)
document.getElementById('uploadBtn').addEventListener('click', () => {
    document.getElementById('fileInput').click();
});

// Input de arquivo (ainda não implementado)
document.getElementById('fileInput').addEventListener('change', (e) => {
    const files = e.target.files;
    if (files.length > 0) {
        console.log('Arquivos selecionados:', files);
        // TODO: Implementar funcionalidade de upload
        alert('Funcionalidade de upload será implementada em breve!');
    }
});

// Carregar lista de documentos
function loadDocuments() {
    const documentsList = document.getElementById('documentsList');
    const emptyState = document.getElementById('emptyState');
    const documentsCount = document.getElementById('documentsCount');

    // TODO: Buscar documentos da API
    // Por enquanto, usar lista vazia
    const documents = [];

    if (documents.length === 0) {
        emptyState.style.display = 'flex';
        documentsList.style.display = 'none';
        documentsCount.textContent = '0 documentos';
    } else {
        emptyState.style.display = 'none';
        documentsList.style.display = 'grid';
        documentsCount.textContent = `${documents.length} documento${documents.length !== 1 ? 's' : ''}`;

        // Renderizar documentos
        documentsList.innerHTML = documents.map(doc => createDocumentCard(doc)).join('');
    }
}

// Criar card de documento
function createDocumentCard(doc) {
    return `
        <div class="document-card">
            <div class="document-icon">
                <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
            </div>
            <div class="document-info">
                <h4 class="document-name">${doc.name}</h4>
                <p class="document-meta">${doc.size} • ${doc.date}</p>
                <span class="document-status ${doc.status}">${doc.statusText}</span>
            </div>
            <div class="document-actions">
                <button class="action-btn" onclick="viewDocument('${doc.id}')">
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                    </svg>
                </button>
                <button class="action-btn" onclick="deleteDocument('${doc.id}')">
                    <svg xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                    </svg>
                </button>
            </div>
        </div>
    `;
}

// Visualizar documento (ainda não implementado)
function viewDocument(docId) {
    console.log('Visualizar documento:', docId);
    // TODO: Implementar visualização
    alert('Visualização de documento será implementada em breve!');
}

// Deletar documento (ainda não implementado)
function deleteDocument(docId) {
    console.log('Deletar documento:', docId);
    // TODO: Implementar deleção
    if (confirm('Tem certeza que deseja excluir este documento?')) {
        alert('Funcionalidade de exclusão será implementada em breve!');
    }
}

// Drag and drop (ainda não implementado)
const uploadCard = document.querySelector('.upload-card');

uploadCard.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadCard.classList.add('dragover');
});

uploadCard.addEventListener('dragleave', () => {
    uploadCard.classList.remove('dragover');
});

uploadCard.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadCard.classList.remove('dragover');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        console.log('Arquivos arrastados:', files);
        // TODO: Implementar funcionalidade de upload
        alert('Funcionalidade de upload será implementada em breve!');
    }
});
