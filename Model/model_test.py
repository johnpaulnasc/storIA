"""
model_test.py

Descrição:
Este script utiliza a biblioteca Transformers para carregar e testar um modelo GPT-2 ajustado, permitindo a geração de texto por diferentes métodos e configurações. Ele implementa:
- Geração de texto com `pipeline`.
- Métodos avançados como Beam Search e amostragem com Top-K e Top-P.
- Variação nos parâmetros de temperatura, penalidade de repetição, e número de feixes.

Principais Funcionalidades:
- Carrega um modelo ajustado a partir de um diretório local.
- Tokeniza e processa o texto de entrada.
- Realiza a geração de texto com diversos métodos para comparar resultados:
  - Pipeline simples.
  - Beam Search.
  - Amostragem com Top-K e Top-P.
- Exibe as histórias geradas no console para avaliação.

Parâmetros de Geração:
- `top_k`: Limita as escolhas do modelo às palavras mais prováveis dentro do K.
- `top_p`: Probabilidade acumulada para filtrar palavras, usando amostragem Nucleus.
- `num_beams`: Número de feixes para Beam Search.
- `max_length` e `min_length`: Controlam o tamanho do texto gerado.
- `temperature`: Controla a aleatoriedade do texto gerado.
- `repetition_penalty`: Penaliza palavras repetidas para maior diversidade.
- `do_sample`: Ativa amostragem ao invés de buscas determinísticas.

Dependências:
- transformers: Biblioteca para manipulação de modelos de linguagem.
- random: Gera seeds aleatórias para controlar a repetibilidade.

Exemplo de Uso:
1. Configure o caminho do modelo ajustado em `model_path`.
2. Insira o texto inicial em `inputs`.
3. Execute o script para gerar histórias com diferentes configurações.
4. Analise os resultados no console.

Nota:
Certifique-se de que o modelo ajustado está presente no caminho especificado e que as dependências estão instaladas.
"""


from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import pipeline, set_seed
import random

model_path = "Model\Models\V-2.0\checkpoint-53705"

set_seed(random.randint(0, 999))

tokenizer = GPT2Tokenizer.from_pretrained(
            'gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>')

model = GPT2LMHeadModel.from_pretrained(
            model_path, eos_token_id=tokenizer.eos_token_id, 
            bos_token_id=tokenizer.bos_token_id)
        
model.resize_token_embeddings(len(tokenizer))

model = model.to('cpu')

inputs = ""
input_ids = tokenizer.encode(inputs, return_tensors='pt')

input_length = len(inputs.split())

escritor = pipeline('text-generation', tokenizer= tokenizer, model = model)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

storia_um = (escritor(inputs, top_k = 30, max_length = input_length + 500, min_length = input_length + 100, return_full_text = True, temperature = float(1), repetition_penalty = float(2.9) )[0].get('generated_text'))

print("\n" *10)
print("STORY WITH PIPELINE / TEMP: 1 / Rep_Pen: 2.9")

print("-"*300)
print(storia_um)
print("-"*300)

print("\n" *2)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

storia_dois = (escritor(inputs, top_k = 30, max_length = input_length + 500, min_length = input_length + 100, return_full_text = True, temperature = float(5), repetition_penalty = float(1) )[0].get('generated_text'))

print("\n" *2)
print("STORY WITH PIPELINE / TEMP: 5 / Rep_Pen: 1")

print("-"*300)
print(storia_dois)
print("-"*300)
print("\n" *2)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

beam_outputs = model.generate(
    input_ids, 
    num_beams=5, 
    max_length = input_length + 500,
    min_length = input_length + 100,
    no_repeat_ngram_size=2, 
    num_return_sequences=1,
    reptition_penalty = float(1.5),
    temperature = float(1),
    early_stopping=True
)

storia_tres = tokenizer.decode(beam_outputs[0], skip_special_tokens=True)

print("\n" *2)
print("STORY WITH GENERATE (Beam_search) / TEMP: 1.5 / Rep_Pen: 1.5")

print("-"*300)
print(storia_tres)
print("-"*300)
print("\n" *2)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

sample_outputs = model.generate(
    input_ids,
    num_beams=10,
    do_sample=True, 
    max_length = input_length + 500,
    min_length = input_length + 100,
    top_k=250, 
    top_p=0.95,
    reptition_penalty = float(3),
    temperature = float(1.5),
    num_return_sequences=1
)

storia_quatro = tokenizer.decode(sample_outputs[0], skip_special_tokens=True)

print("\n" *2)
print("STORY WITH GENERATE(TOP_K/TOP_P) / TEMP: 1.5 / Rep_Pen: 2")

print("-"*300)
print(storia_quatro)
print("-"*300)
print("\n" *2)
#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------