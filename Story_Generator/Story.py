"""
Story.py

Descrição:
Este script define a classe `Story_generator` que fornece uma interface para carregar um modelo GPT-2 ajustado e gerar histórias com base em um texto de entrada. A classe encapsula toda a lógica de pré-processamento do texto, configuração do modelo, e geração de texto com opções personalizáveis.

Principais Funcionalidades:
- Carregamento de um modelo GPT-2 ajustado a partir de um caminho especificado.
- Pré-processamento do texto de entrada para garantir consistência e adequação ao modelo.
- Geração de histórias com parâmetros ajustáveis, como comprimento máximo, temperatura, penalidade de repetição, e Beam Search.

Classes e Métodos:
- `Story_generator`: Classe principal para geração de histórias.
  - `__init__`: Inicializa o modelo e o tokenizer.
  - `clean_text(text)`: Remove espaços no final, converte o texto para minúsculas, e adiciona o token inicial.
  - `generate_story(text, size, temperature)`: Gera uma história com base no texto fornecido e nos parâmetros de configuração.

Parâmetros de Geração:
- `size`: Número máximo de palavras adicionais a serem geradas.
- `temperature`: Controla a aleatoriedade na escolha das próximas palavras.
- `repetition_penalty`: Penaliza repetições para aumentar a diversidade do texto.
- `num_beams`: Número de feixes usados no Beam Search.
- `no_repeat_ngram_size`: Evita a repetição de n-gramas específicos.

Dependências:
- transformers: Biblioteca para manipulação de modelos de linguagem.
- random: Gera seeds aleatórias para experimentação.

Exemplo de Uso:
1. Crie uma instância da classe `Story_generator`.
2. Use o método `generate_story` para criar uma história com base em texto inicial.
3. Personalize os parâmetros de geração para obter diferentes estilos de texto.

Nota:
Certifique-se de que o modelo ajustado está no caminho especificado e que as dependências necessárias estão instaladas corretamente.
"""


from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import set_seed
import random
from transformers import pipeline

class Story_generator:

    def __init__(self):
        self.model_path = "V-3.0\checkpoint-53705"

        set_seed(random.randint(0, 999))

        self.tokenizer = GPT2Tokenizer.from_pretrained(
            'gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>', add_prefix_space = False)

        self.model = GPT2LMHeadModel.from_pretrained(
                    self.model_path, eos_token_id=self.tokenizer.eos_token_id, 
                    bos_token_id=self.tokenizer.bos_token_id)
                
        self.model.resize_token_embeddings(len(self.tokenizer))

        self.model = self.model.to('cpu')


    def clean_text(self, text) -> str:
        """
        Params: Input text
        Returns: treated input text (whithout spaces on the end, lower cased and with start token)
        """
        restart = True

        while(restart):
            if text[-1] != " ":
                restart = False
            else:
                text = text[:-1]

        text = text.lower()
        
        return "<|startoftext|> " + text


    def generate_story(self, text, size, temperature) -> str:
        """
        Params: Input text, max size and temperature
        Returns: generated story
        """

        if text != "":
            text = self.clean_text(text)

        input_length = len(text.split())

        writer = pipeline('text-generation', model = self.model, tokenizer = self.tokenizer)

        story = writer(
                text, max_length = input_length + size, 
                temperature = float(temperature), 
                repetition_penalty = float(1.2),
                num_beams = 5,
                no_repeat_ngram_size = 3)
            
        story = story[0].get('generated_text')

        story = story.replace('<|startoftext|>', "")
        story = story.replace('<|endoftext|>', "")

        return story

story = Story_generator()

example = story.generate_story("", 50, 1.2 )

print("\n" +example+ "\n")