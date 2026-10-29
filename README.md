# Desafio Sprint 03

## Introdução

Este desafio consiste em treinar um modelo de acordo com o probelma escolhido. O problema escolhido é prever um paciente pode ter ou não diabetes de acordo com os dados passados.

## Tecnologias Usadas

- Python
- Pandas
- Sklearn
- Joblib
- Matplotlib
- Seaborn

## Instruções

1. Primeiro clone o repositório:

```bash
# sem ssh configurado e cadastrado no github
git clone https://github.com/ricardo-henrique/desafio-sprint03-ml.git

# com ssh configurado e cadastrado no github
git clone git@github.com:ricardo-henrique/desafio-sprint03-ml.git
```

2. Crie um ambiente virtual python

```bash
# crie com o nome ".venv" é uma convenção
python -m venv .venv
```

3. Entre no ambiente virtual

Caso tenha a extensão `python enviroment` instalado no vscode, pule esta etapa, pois, assim que abre o terminal dentro vscode irá executar o mesmo comando ditado abaixo.

```bash
# Powershell
.venv\Scripts\Activate.ps1

# bash/zsh/fish
.venv\Scripts\activate
```

Isso é importante para que as bibliotecas sejam instaladas localmente e tenha controle das versões das depêndencias.

4. Instale as dependência:

```bash
pip install requirements.txt
```

## Preparação para treino de modelo

Para treinar este modelo, primeiro é necessário baixar o dataset em formato .csv na plateforma do `kaggle` [diabetes dataset](https://www.kaggle.com/datasets/datascikhan/diabetes).

> É necessário que se cadastre na plataforma para conseguir baixar o dataset.

Após baixar o dataset, crie a pasta `data/` e armazene o arquivo .csv nesta pasta

Crie também a pasta `models/`, pois é onde serão armazenar os modelos treinados.

## Execução

execute o script para treinar o modelo:

```bash
python main.py
```

Obs: É normal que demore um pouco, dependendo da configuração do seu computador pode levar alguns minutos.

# Conclusão

Será exibido no seu terminal o resultado do treinamento do modelo e será gerado os modelos na pasta `models/`

para mais informações clique neste link [docs](/docs/model_testing.md)

Antes do treinamento final, o script também avalia os modelos com validação cruzada
Stratified K-Fold de 5 folds sobre os dados de treino. São exibidas as médias e os
desvios padrão de acurácia, precisão, recall e F1-score ponderados. O conjunto de
teste permanece separado para a avaliação final.

Para a Random Forest, o script também executa uma busca aleatória de 30 combinações
de hiperparâmetros usando os mesmos 5 folds e seleciona a combinação com melhor
F1-score ponderado. O estimador selecionado é então avaliado no conjunto de teste
e salvo como `models/model1_random_forest.pkl`.
