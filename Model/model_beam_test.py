"""
model_beam_test.py

Descrição:
Este script utiliza a biblioteca Transformers para carregar e gerar texto com um modelo de linguagem GPT-2 ajustado. Ele inclui funcionalidades para:
- Tokenizar o texto de entrada.
- Gerar texto utilizando o método de Beam Search.
- Configurar parâmetros como penalidade por repetição, temperatura, e tamanho do texto gerado.

Principais Funcionalidades:
- Carrega um modelo GPT-2 ajustado a partir de um diretório local.
- Realiza a tokenização com tokens de início e fim personalizados.
- Utiliza o método de geração `generate` com Beam Search para produzir texto.
- Exibe o texto gerado com parâmetros específicos.

Parâmetros de Geração:
- `num_beams`: Número de feixes para Beam Search.
- `max_length`: Tamanho máximo do texto gerado.
- `min_length`: Tamanho mínimo do texto gerado.
- `no_repeat_ngram_size`: Restringe a repetição de n-gramas.
- `repetition_penalty`: Penalidade para palavras repetidas.
- `temperature`: Controla a aleatoriedade na geração de texto.
- `early_stopping`: Para a geração assim que todos os feixes chegarem ao token final.

Dependências:
- transformers: Biblioteca para modelos de linguagem.
- random: Para geração de seeds aleatórias.

Exemplo de Uso:
1. Configure o caminho do modelo ajustado em `model_path`.
2. Insira o texto inicial em `inputs`.
3. Ajuste os parâmetros do método `generate` para personalizar o texto gerado.
4. Execute o script para gerar texto com Beam Search.

Nota:
Certifique-se de que o modelo ajustado está presente no caminho especificado e que as dependências necessárias estão instaladas.
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

inputs = "<|startoftext|> I was a always alone, i had no friends"

input_ids = tokenizer.encode(inputs, return_tensors='pt')


input_length = len(inputs.split())

escritor = pipeline('text-generation', tokenizer= tokenizer, model = model)

#-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

beam_outputs = model.generate(
    input_ids, 
    num_beams=5, 
    max_length = input_length + 100,
    min_length = input_length + 50,
    no_repeat_ngram_size=2, 
    num_return_sequences=1,
    reptition_penalty = float(1),
    temperature = float(1),
    early_stopping=True
)

storia_tres = tokenizer.decode(beam_outputs[0], skip_special_tokens=True)

print("\n" *2)
print("STORY WITH GENERATE (Beam_search) / TEMP: 1.5 / Rep_Pen: 1.5")

print("-"*100)
print(storia_tres)
print("-"*100)
print("\n" *2)