"""
app.py

Descrição:
Este arquivo define o aplicativo Flask que serve como backend para o projeto. Ele contém rotas para renderizar páginas HTML e processar requisições enviadas pelo usuário. Inclui lógica para manipulação de dados e comunicação com utilitários externos.

Rotas disponíveis:
- "/" (GET): Renderiza a página inicial.
- "/submit" (POST, GET): Processa dados enviados por um formulário, realiza manipulações e retorna sugestões.
- "/social" (GET): Renderiza a página social.
- "/members" (GET): Renderiza a página de membros.

Dependências:
- Flask: Framework web usado para construir o servidor e gerenciar rotas.
- utils: Módulo externo contendo funções utilitárias.
- test: Módulo contendo funções para manipulação de tokens e consulta de modelos.

Como executar:
1. Certifique-se de ter instalado as dependências necessárias (`Flask`).
2. Execute este script para iniciar o servidor: `python app.py`.
3. Acesse o aplicativo no navegador em `http://127.0.0.1:5000/`.
"""

from flask import Flask, render_template, request
import utils
from test import *

app = Flask(__name__, template_folder='templates')

@app.route("/")
def hello():
    return render_template('index.html')

@app.route("/submit", methods = ["POST", "GET"])
def submit():

    data = request.form


    print(data['text[]'])

    example = check_token(data['text[]'])

    input_len = len(example.split())
    size = data['length[]']# DEFINIDO PELO USUÁRIO DEFAULT 50
    
    output = query({"inputs": example,
                "parameters": {'repetition_penalty': float(1.2), 'num_beams':5,
                               'no_repeat_ngram_size':3, 'max_length':input_len + int(size)}})
    print(output[0].get('generated_text'))
    

    
    return render_template('index.html', suggestion_text=remove_token(output[0].get('generated_text')))

@app.route("/social")
def social():
    return render_template('social.html')

@app.route("/members")
def members():
    return render_template('members.html')




if __name__ == "__main__":
    app.run()