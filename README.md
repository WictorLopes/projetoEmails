# Classificador de Emails com IA

## Contexto do Desafio

Estamos criando uma **solução digital para uma grande empresa** do setor financeiro que lida com um **alto volume de emails diariamente**. Esses emails podem ser mensagens solicitando um status atual sobre uma requisição em andamento, compartilhando algum arquivo ou até mesmo mensagens improdutivas, como desejo de feliz natal ou perguntas não relevantes.

Nosso **objetivo é automatizar a leitura e classificação desses emails** e sugerir classificações e respostas automáticas de acordo com o teor de cada email recebido, **liberando tempo da equipe** para que não seja mais necessário ter uma pessoa fazendo esse trabalho manualmente.

---

## Objetivo do Projeto

Desenvolver uma aplicação web simples que utilize inteligência artificial para:

1. **Classificar** emails em categorias predefinidas.
2. **Sugerir respostas automáticas** baseadas na classificação realizada.

### Categorias de Classificação

- **Produtivo:** Emails que requerem uma ação ou resposta específica (ex.: solicitações de suporte técnico, atualização sobre casos em aberto, dúvidas sobre o sistema).
- **Improdutivo:** Emails que não necessitam de uma ação imediata (ex.: mensagens de felicitações, agradecimentos).

---

## Funcionalidades

### Interface Web

- Formulário para inserção direta do texto do email ou upload de arquivos `.txt` e `.pdf`.
- Exibição da categoria atribuída ao email (Produtivo ou Improdutivo).
- Exibição da resposta automática sugerida pelo sistema.
- Estatísticas básicas do texto (número de palavras, caracteres e tempo de processamento).
- Botão para copiar a resposta gerada para a área de transferência.

### Backend em Python

- Extração de texto de arquivos PDF e TXT.
- Pré-processamento do texto com técnicas de NLP (remoção de stop words, lematização).
- Classificação do email usando a API Gemini da Google (modelo Gemini 1.5 Flash).
- Fallback para classificação baseada em palavras-chave caso a API falhe.
- Geração de respostas automáticas adequadas à categoria do email.
- Endpoints REST para integração via API.
- Health check para monitoramento da conexão com a API Gemini.

---

## Tecnologias Utilizadas

- Python 3
- Flask (framework web)
- PyPDF2 (extração de texto de PDFs)
- NLTK (processamento de linguagem natural)
- Google Gemini API (classificação e geração de texto)
- HTML, CSS e JavaScript para frontend
- Dotenv para gerenciamento de variáveis de ambiente

---

## Como Executar Localmente

### Pré-requisitos

- Python 3.8+
- Conta e chave de API para Google Gemini (definida na variável de ambiente `GEMINI_API_KEY`)

### Passos

1. Clone o repositório:

```bash
git clone https://github.com/WictorLopes/projetoEmails.git
cd projetoEmails
```

2. Crie e ative um ambiente virtual:

```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3. Instale as dependências:

```bash
pip install -r requirements.txt
```

4. Configure a variável de ambiente no arquivo `.env`:

```
GEMINI_API_KEY=your_api_key_here
FLASK_DEBUG=True
```

5. Execute a aplicação:

```bash
python app.py
```

6. Acesse no navegador:

```
http://localhost:5000
```

---

## Deploy na Nuvem

A aplicação está hospedada na plataforma gratuita Vercel.

**Link para a aplicação hospedada:**  
[https://projetoemails.vercel.app/](https://projetoemails.vercel.app/)

---

## Estrutura do Projeto

```
├── api/
│ └── index.py                     # Aplicação Flask principal
├── requirements.txt               # Dependências Python
├── vercel.json                    # Configuração de deploy para Vercel
├── static/
│ ├── style.css                    # Arquivo CSS externo
│ └── favicon.ico                  # Favicon do site
├── templates/
│ └── index.html                   # Template HTML principal (para desenvolvimento local)
├── .env                           # Variáveis de ambiente local (não versionar)
├── .gitignore                     # Arquivos ignorados pelo Git
└── README.md                      # Este arquivo
```

---

## Considerações Finais

- A aplicação prioriza o upload de arquivos para extração do texto, mas também aceita texto colado diretamente.
- A classificação e geração de respostas são feitas via API Gemini, com fallback local para garantir robustez.
- A interface foi desenvolvida para ser simples, responsiva e amigável ao usuário final.
---
