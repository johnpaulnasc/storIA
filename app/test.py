"""
test.py

Descrição:
Este módulo contém funções para interação com o modelo de inferência hospedado no Hugging Face e manipulação de texto. Ele inclui funcionalidades para verificar e adicionar tokens, remover tokens desnecessários, e enviar requisições para gerar texto baseado em entradas fornecidas.

Funções:
- remove_token(text): Remove o token inicial do texto gerado.
- check_token(input): Adiciona um token inicial ao texto de entrada, caso não esteja presente.
- query(payload): Envia uma requisição POST para a API de inferência do Hugging Face e retorna o JSON de resposta.

Dependências:
- requests: Biblioteca para enviar requisições HTTP.

Configuração:
- API_URL: URL do modelo de inferência hospedado no Hugging Face.
- headers: Cabeçalho contendo o token de autenticação necessário para acessar a API.

Exemplo de Uso:
1. Insira o texto do usuário na variável `input`.
2. Ajuste o tamanho máximo de saída usando a variável `size`.
3. Chame `check_token` para preparar o texto.
4. Use a função `query` para gerar texto.
5. Manipule o texto gerado com `remove_token`.

Nota:
Substitua o token de autenticação no cabeçalho por uma chave válida para acesso à API.
"""

import requests

API_URL = "https://api-inference.huggingface.co/models/Felipehonorato/storIA"
headers = {"Authorization": "Bearer api_CwzaLVoNBMVQhviuBtnxxdVoXvQgjuTEmW"}

def remove_token(text):
  return " ".join(text.split()[1:])

def check_token(input):
  token = '<|startoftext|> '
  if input.split()[0] != token:
    return token + input
  else:
    return input

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()

input = 'it was a dark night' # TEXTO DO USUÁRIO
input_len = len(input.split())
size = 50 # DEFINIDO PELO USUÁRIO DEFAULT 50
input = check_token(input)
output = query({"inputs": input,
                "parameters": {"max_length": 50, 'repetition_penalty': float(1.2), 'num_beams':5,
                               'no_repeat_ngram_size':3, 'max_length':input_len + size}})