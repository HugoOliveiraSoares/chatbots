# Chatbot

Trabalho da disciplina de PLN(Processamento de Linguagem natural)

## Arquivos

train.py -> Script que treina o bot e gera o arquivo .csv e os arquivos de modelo treinados

bot.py -> Script do chatbot

breast_cancer_wiki.csv -> dataset com o texto original dividido pelos titulos

intents.json -> Lista de possiveis entradas do usuario e respostas

requirements.txt -> Lista com as dependencias do projeto

Os outros arquivos fazer parte do modelo treinado e logs do tensorflow

## Instalar dependencias

pip install -r requirements.txt

## Treinamento do modelo

python train.py

## Executando o bot

python bot.py