import os
import nltk
nltk_data_path = os.path.join(os.path.dirname(__file__), '..', 'nltk_data')
nltk.data.path.append(nltk_data_path)

from flask import Flask, render_template, request, jsonify
import os
import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import string
import re
import time
import google.generativeai as genai
from dotenv import load_dotenv

# Serverless adapter para o vercel
# from vercel_python import create_vercel_handler

load_dotenv()

app = Flask(__name__, template_folder="../templates", static_folder="../static")

# Configurações
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['GEMINI_API_KEY'] = os.getenv('GEMINI_API_KEY')

# Verificar se a chave API está configurada
if not app.config['GEMINI_API_KEY']:
    app.config['GEMINI_API_KEY'] = os.environ.get('GEMINI_API_KEY')

if not app.config['GEMINI_API_KEY']:
    raise ValueError("GEMINI_API_KEY não encontrada. Configure as variáveis de ambiente no Vercel")

# Configurar a API do Gemini
genai.configure(api_key=app.config['GEMINI_API_KEY'])

GEMINI_MODEL = "models/gemini-1.5-flash-latest"

stop_words = set(stopwords.words('portuguese'))
lemmatizer = WordNetLemmatizer()

# Palavras-chave expandidas
produtivo_keywords = [
    'suporte', 'atualização', 'dúvida', 'problema', 'erro', 'reclamação', 
    'ajuda', 'solicitação', 'urgente', 'contrato', 'fatura', 'pagamento',
    'cobrança', 'suporte técnico', 'defeito', 'bug', 'sistema', 'login',
    'senha', 'acesso', 'proposta', 'orçamento', 'projeto', 'prazo'
]

improdutivo_keywords = [
    'feliz natal', 'obrigado', 'parabéns', 'bom dia', 'boa tarde', 
    'agradecimento', 'saudações', 'cumprimentos', 'feliz ano novo',
    'boas festas', 'saudação', 'contato futuro', 'mantenha contato'
]

def extract_text_from_pdf(pdf_file):
    # Extrair texto de um arquivo PDF
    try:
        reader = PyPDF2.PdfReader(pdf_file)
        texto_pdf = []
        for page in reader.pages:
            texto = page.extract_text()
            if texto:
                texto_pdf.append(texto)
        return "\n".join(texto_pdf)
    except Exception as e:
        raise Exception("Não foi possível extrair texto do arquivo PDF")

def extract_text_from_txt(txt_file):
    # Extrair texto de um arquivo TXT
    try:
        return txt_file.read().decode('utf-8')
    except Exception as e:
        raise Exception("Não foi possível ler o arquivo de texto")

def classify_email_gemini(text):
    # Classificação usando a API
    try:
        # Preparar o prompt para classificação
        prompt = f"""
        Classifique o seguinte e-mail em português como "Produtivo" ou "Improdutivo":
        
        E-mail: {text[:1000]}
        
        Um e-mail "Produtivo" geralmente contém solicitações, problemas, dúvidas, 
        questões comerciais ou requer alguma ação. Um e-mail "Improdutivo" é geralmente 
        composto por saudações, agradecimentos, mensagens sociais ou não requer ação.
        
        Responda APENAS com "Produtivo" ou "Improdutivo", nada mais.
        """
        
        # Chamar a API do Gemini com o modelo econômico
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        
        # Processar a resposta
        category = response.text.strip()
        
        if "Produtivo" in category:
            return 'Produtivo'
        elif "Improdutivo" in category:
            return 'Improdutivo'
        else:
            return classify_email_fallback(text)
            
    except Exception as e:
        return classify_email_fallback(text)

def classify_email_fallback(text):
    # Classificação fallback baseada em keywords
    text_lower = text.lower()
    produtivo_count = sum(1 for kw in produtivo_keywords if kw in text_lower)
    improdutivo_count = sum(1 for kw in improdutivo_keywords if kw in text_lower)
    
    if produtivo_count > improdutivo_count:
        return 'Produtivo'
    elif improdutivo_count > produtivo_count:
        return 'Improdutivo'
    else:
        return 'Improdutivo'

def generate_gemini_response(category, email_text):
    # Geração de resposta usando Gemini API
    try:
        # Preparar o prompt baseado na categoria
        if category == 'Produtivo':
            prompt = f"""
            Escreva uma resposta profissional e útil em português para o seguinte e-mail, 
            demonstrando empatia e oferecendo suporte. Seja conciso e direto (máximo 3 frases).
            
            E-mail: {email_text[:1000]}
            
            Resposta:
            """
        else:
            prompt = f"""
            Escreva uma resposta educada em português para o seguinte e-mail, 
            agradecendo o contato e mantendo um tom cordial. Seja breve (máximo 2 frases).
            
            E-mail: {email_text[:1000]}
            
            Resposta:
            """
        
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content(prompt)
        
        return response.text.strip()
        
    except Exception as e:
        return generate_fallback_response(category)

def generate_fallback_response(category):
    # Resposta de fallback
    if category == 'Produtivo':
        return "Agradecemos seu contato. Nossa equipe está analisando sua solicitação e retornaremos em breve."
    else:
        return "Obrigado pela sua mensagem. Ficamos felizes em saber de você!"

@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        email_text = request.form.get('email_text', '').strip()
        if not email_text:
            return render_template('index.html', error="Por favor, insira o texto do email.")
        start_time = time.time()
        category = classify_email_gemini(email_text)
        response = generate_gemini_response(category, email_text)
        processing_time = round(time.time() - start_time, 2)
        return render_template('index.html',
                               email_text=email_text,
                               category=category,
                               response=response,
                               processing_time=processing_time)
    else:
        return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    # Endpoint principal para classificação
    email_text = None
    category = None
    response = None
    error = None
    processing_time = None
    filename = None
    
    try:
        start_time = time.time()
        
        # Verificar se é um upload de arquivo
        if 'file' in request.files:
            email_file = request.files['file']
            if email_file and email_file.filename != '':
                filename = email_file.filename
                file_extension = filename.lower().split('.')[-1]
                
                if file_extension == 'pdf':
                    email_text = extract_text_from_pdf(email_file)
                elif file_extension == 'txt':
                    email_text = extract_text_from_txt(email_file)
                else:
                    return jsonify({
                        'error': 'Formato de arquivo não suportado. Use PDF ou TXT.',
                        'success': False
                    }), 400
        else:
            # Verificar se é texto JSON
            data = request.get_json()
            if data:
                email_text = data.get('text', '')
            else:
                # Verificar se é form data
                email_text = request.form.get('email_text', '')
        
        # Verificar se tem texto para processar
        if not email_text or len(email_text.strip()) <= 10:
            return jsonify({
                'error': 'Texto do email é muito curto para análise. Mínimo 10 caracteres.',
                'success': False
            }), 400
        
        # Classificar email
        category = classify_email_gemini(email_text)
        
        # Gerar resposta
        response = generate_gemini_response(category, email_text)
        
        processing_time = round(time.time() - start_time, 2)
        
        return jsonify({
            'category': category,
            'response': response,
            'processing_time': processing_time,
            'text_length': len(email_text),
            'filename': filename,
            'success': True
        })
        
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/health', methods=['GET'])
def health_check():
    # Endpoint de health check
    try:
        # Testar conexão com a API do Gemini
        model = genai.GenerativeModel(GEMINI_MODEL)
        response = model.generate_content("Teste de conexão")
        return jsonify({
            'status': 'healthy', 
            'gemini_connection': 'success',
            'model': GEMINI_MODEL
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy', 
            'gemini_connection': 'failed', 
            'error': str(e),
            'model': GEMINI_MODEL
        }), 500

# Handler para o Vercel
# handler = create_vercel_handler(app)

# Para desenvolvimento local
if __name__ == '__main__':
    app.run(debug=os.getenv('FLASK_DEBUG', 'False').lower() == 'true', 
            host='0.0.0.0', 
            port=5000)