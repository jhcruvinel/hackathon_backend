# Importacoes
import os
import ast
import nltk
import pandas as pd
import pandas.io.sql as psql
import pandas as pd
import string
from nltk.corpus import stopwords
import re
import math
from collections import Counter
from unicodedata import normalize
from nltk import ngrams
from flask import Flask
from flask import request
from flask import jsonify
from flask_cors import CORS
from nltk.stem import SnowballStemmer
import json
from pandas.io.json import json_normalize
from datetime import datetime


# Constantes
portuguese_stops = set(stopwords.words('portuguese'))
ignore = ['@', '.', '!','?',',','$','-','\'s','g','(',')','[',']','``',':','http','html','//members']
stemmer = SnowballStemmer('english')
# Regex para encontrar tokens
REGEX_WORD = re.compile(r'\w+')
# Numero de tokens em sequencia
N_GRAM_TOKEN = 3
DATABASE = 'banco.csv'

#-----------------------------------------------------------------------------------
# FUNÇOES DE PRE-PROCESSAMENTO DE TEXTO

# Funcao de pre-processamento do texto
def preprocess_text(content):
    # Remove pontuacoes
    content.translate(str.maketrans('', '', string.punctuation))
    word_tokens = nltk.word_tokenize(content)
    # Remove stopwords e coloca em minusculo
    word_tokens2 = [w.lower() for w in word_tokens if w not in portuguese_stops]
    # remove lista de ignore
    word_tokens3 = [w for w in word_tokens2 if w not in ignore]
    # aplica o stemming
    word_tokens4 = [stemmer.stem(w) for w in word_tokens3]
    #return word_tokens4
    return ' '.join(word_tokens4)

# Faz a normalizacao do texto removendo espacos a mais e tirando acentos
def text_normalizer(src):
    return re.sub('\s+', ' ',
                normalize('NFKD', src)
                   .encode('ASCII','ignore')
                   .decode('ASCII')
           ).lower().strip()

#-----------------------------------------------------------------------------------
# FUNÇOES DE CALCULO DE SIMILARIDADE

# Calculo de similaridade do coseno
def cosine_similarity(vec1, vec2):
    intersection = set(vec1.keys()) & set(vec2.keys())
    numerator = sum([vec1[x] * vec2[x] for x in intersection])
    sum1 = sum([vec1[x]**2 for x in vec1.keys()])
    sum2 = sum([vec2[x]**2 for x in vec2.keys()])
    denominator = math.sqrt(sum1) * math.sqrt(sum2)
    if not denominator:
        return 0.0
    else:
        coef = float(numerator) / denominator
        if coef > 1:
            coef = 1
        return coef

# Monta o vetor de frequencia da sentenca
def sentence_to_vector(text, use_text_bigram):
    words = REGEX_WORD.findall(text)
    accumulator = []
    for n in range(1,N_GRAM_TOKEN):
        gramas = ngrams(words, n)
        for grama in gramas:
            accumulator.append(str(grama))
    if use_text_bigram:
        pairs = get_text_bigrams(text)
        for pair in pairs:
            accumulator.append(pair)
    return Counter(accumulator)

# Calcula similaridade entre duas sentencas
def get_sentence_similarity(sentence1, sentence2, use_text_bigram=False):
    vector1 = sentence_to_vector(text_normalizer(sentence1), use_text_bigram)
    vector2 = sentence_to_vector(text_normalizer(sentence2), use_text_bigram)
    return cosine_similarity(vector1, vector2)

# Metodo de gerar bigramas de uma string
def get_text_bigrams(src):
    s = src.lower()
    return [s[i:i+2] for i in range(len(s) - 1)]

# Simulando o banco
df = pd.read_csv('dados_teste.csv',delimiter='|')
df['nome'] = df['nome'].apply(preprocess_text)
texts = df['nome']

def most_similar(sentences,sentence):
    max_sim = 0
    most_similar = ''
    for s in sentences:
        sim = get_sentence_similarity(sentence, s, use_text_bigram=True)
        if sim > max_sim:
            max_sim = sim
            most_similar = s
            print(str(sim)+' '+s)
    print ('Mais similar a ' + sentence + ' = ' + most_similar + ' with distance = ' + str(max_sim))
    return {'doc':most_similar}


#-----------------------------------------------------------------------------------
# FUNÇOES PARA SIMULACAO DE BANCO DE DADOS COM PANDAS

# Salva dados no banco
def write_to_csv_file_by_pandas(data_frame):
    data_frame.to_csv(DATABASE, index=False)
    print(DATABASE + ' has been created.')

# Leitura 
def read_csv_file_by_pandas():
    data_frame = None
    if(os.path.exists(DATABASE)):
        data_frame = pd.read_csv(DATABASE, index_col=False)
    else:
        print(DATABASE + " do not exist.")    
    return data_frame

#-----------------------------------------------------------------------------------
# DEFINICAO DOS SERVICOS REST DO FLASK 

# Iniciando app Flask  
app = Flask(__name__)
cors = CORS(app, resources={r"/*": {"origins": "*"}})
@app.route('/')
def index():
  return 'Acesse o servico em /similaridade passando o JSON com POST'

# Mensagem de erro caso seja acessado via GET
@app.route('/api/v1/similaridade', methods=['GET'])
def erro():
  return 'Este servico so pode ser acessado via POST'

# Calculo de similaridade de um texto apenas, para teste
@app.route('/api/v1/similaridade', methods=['POST'])
def similaridade():
  content = request.json
  print ("Request: "+str(content))
  print ("Nome: "+content['nome'])
  text = preprocess_text(content['nome'])
  response = jsonify(most_similar(texts,text))
  response.headers.add('Access-Control-Allow-Origin', '*')
  return response

# Chamada de inclusao de processo
@app.route('/api/v1/processos/incluirProcesso', methods=['POST'])
def incluirProcesso():
  content = request.json
  print ("Request: "+str(content))
  df_existente = read_csv_file_by_pandas()
  content['id'] = 1
  content['acordo'] = 'S'
  content['link'] = 'http://www.trt12.jus.br/busca/sentencas/browse?q=aviso+pr%C3%A9vio&from=&to=&fq=&fq=ds_orgao_julgador%3A%221%C2%AA+VARA+DO+TRABALHO+DE+BLUMENAU%22'
  if df_existente is None:
    df = pd.io.json.json_normalize(content)
    write_to_csv_file_by_pandas(df)
  else:
    content['id'] = int(pd.read_csv(DATABASE)['id'].max())+1
    df = pd.io.json.json_normalize(content)
    df_existente = df_existente.append(df)
    write_to_csv_file_by_pandas(df_existente)
  print('Processo salvo no banco local')
  response = jsonify(content)
  response.headers.add('Access-Control-Allow-Origin', '*')
  return response


# Chamada pra consulta de processo
@app.route('/api/v1/processos/processo/<id>', methods=['GET'])
def buscaProcesso(id):
  print ("Buscando Processo ID: "+id)
  df_existente = pd.read_csv(DATABASE,header=0)
  df_processo = df_existente.loc[df_existente['id'] == id]
  json = df_processo.to_json()[1:-1].replace('},{', '} {')
  print(json)
  return json


# Chamada de insights de processo
@app.route('/api/v1/processos/insightsProcesso/<id>', methods=['GET'])
def insightsProcesso(id):
  print ("Buscando Insight para Processo ID: "+id)
  df_existente = pd.read_csv(DATABASE,header=0)
  print('recuperou')
  df_processo = df_existente.loc[df_existente['id'] == id]
  calcula_probabilidade_acordo(df_processo, df_existente)
  return 'teste'

def calcula_probabilidade_acordo(df_processo, df_existente):
    for i, p in df_existente.iterrows():
       print(p['acordo'])
       print(df_processo.iloc[0]['acordo'])


def calcula_similaridade_processos(processo1,processo2):
    sim_nomeParteReclamante = get_sentence_similarity(processo1['nomeParteReclamante'],processo2['nomeParteReclamante'])
    print("Similaridade Partes: "+sim_nomeParteReclamante)
    date1 = datetime.strptime(processo1['dataAjuizamentoInicial'], '%T/%m/%d').date()
    date2 = datetime.strptime(processo2['dataAjuizamentoInicial'], '%T/%m/%d').date()
    dif_dataAjuizamentoInicial = abs((d2 - d1).days)
    print("Diferença Datas: "+dif_dataAjuizamentoInicial)
    return 
