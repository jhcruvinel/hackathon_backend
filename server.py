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

def recupera_processo(id):
    df_existentes = pd.read_csv(DATABASE,header=0)
    df_processo = df_existentes.loc[df_existentes['id'] == int(id)]
    processo = df_processo.to_dict(orient='records')[0]
    processo['pedidos'] = ast.literal_eval(processo['pedidos'])
    return processo

# Chamada pra consulta de processo
@app.route('/api/v1/processos/processo/<id>', methods=['GET'])
def buscaProcesso(id):
  print ("Buscando Processo ID: "+id)
  return jsonify(recupera_processo(id))

# Chamada de insights de processo
@app.route('/api/v1/processos/insightsProcesso/<id>', methods=['GET'])
def insightsProcesso(id):
  print ("Buscando Insight para Processo ID: "+id)
  df_existente = pd.read_csv(DATABASE,header=0)
  df_processo = df_existente.loc[df_existente['id'] == int(id)]
  print('recuperou')
  return calcula_insights(df_processo, df_existente)
  
# Metodo para concatenar em uma string mais de um pedido para fins de comparacao
def concatenar_pedidos(pedidos):
    pedido_concatenado = ''
    for pedido in pedidos:
        pedido_concatenado += pedido['tipo']+"_"
    return pedido_concatenado

# Metodo para o calculo ponderado da diferenca entre processos
def calcula_diferenca_processos(p1, p2):
    dif = {}
    # So compara se nao for com si mesmo
    if (p1['id'] != p2['id'].iloc[0]):
        # Comparacao dos Nomes das Partes Reclamadas
        nomes_reclamante_parecidos = get_sentence_similarity(p1['nomeParteReclamante'] , p2['nomeParteReclamante'].iloc[0], use_text_bigram=True)
        dif['nomes_reclamante_parecidos'] = nomes_reclamante_parecidos
        # Comparacao dos Nomes das Partes Reclamantes
        nomes_reclamada_parecidos = get_sentence_similarity(p1['nomeParteReclamada'] , p2['nomeParteReclamada'].iloc[0], use_text_bigram=True)
        dif['nomes_reclamada_parecidos'] = nomes_reclamada_parecidos
        # Compracao do prazo de Ajuizamento
        d1 = datetime.strptime(p1['dataAjuizamentoInicial'], '%Y-%m-%d').date()
        d2 = datetime.strptime(p2['dataAjuizamentoInicial'].iloc[0], '%Y-%m-%d').date()
        dif_dataAjuizamentoInicial = abs((d2 - d1).days)
        dif['dif_dataAjuizamentoInicial'] = dif_dataAjuizamentoInicial
        # Comparacao dos meses de contratacao
        d1 = datetime.strptime(p1['dataInicioContrato'], '%Y-%m-%d').date()
        d2 = datetime.strptime(p2['dataTerminoContrato'].iloc[0], '%Y-%m-%d').date()
        dif_meses = abs((d2 - d1).days)/30
        dif['dif_meses'] = dif_meses
        # Comparacao dos meses de salario proporcional de 13
        propMeses13P1 = float(p1['meses13SalarioProporcional'])
        propMeses13P2 = float(p2['meses13SalarioProporcional'].iloc[0])
        dif_propMeses13P = propMeses13P1-propMeses13P2
        dif['dif_propMeses13P'] = dif_propMeses13P
        # Comparacao dos salarios
        salario1 = float(p1['salario'])
        salario2 = float(p2['salario'].iloc[0])
        dif_salario = salario1-salario2
        dif['dif_salario'] = dif_salario
        # Comparacao das jornadas
        jornadaSemanal1 = int(p1['jornadaSemanal'])
        jornadaSemanal2 = int(p2['jornadaSemanal'].iloc[0])
        dif_jornadaSemanal = jornadaSemanal1-jornadaSemanal2
        dif['dif_jornadaSemanal'] = dif_jornadaSemanal
        # Comparacao dos pedidos
        pedido_concatenado1 = concatenar_pedidos(ast.literal_eval(p1['pedidos']))
        pedido_concatenado2 = concatenar_pedidos(ast.literal_eval(p2['pedidos'].iloc[0]))
        pedidos_iguais = pedido_concatenado1 == pedido_concatenado2
        dif['pedidos_iguais'] = pedidos_iguais
        # Calculo da distancia total entre pedidos com uso de pesos diferenciados
        dif_total = nomes_reclamante_parecidos * 1.5
        dif_total += nomes_reclamada_parecidos * 1.5
        dif_total += dif_dataAjuizamentoInicial / 10
        dif_total += dif_meses / 48
        dif_total += dif_propMeses13P / 6
        dif_total += dif_salario / 2000
        dif_total += dif_jornadaSemanal / 44
        dif['dif_total'] = dif_total
    return dif

# Metodos para calculo de Insights
def calcula_insights(df_processo, df_existente):
    print('************** CALCULA INSIGHTS **************')
    insights = {}
    difs = []
    difs_totais = 0.0
    menor_dif = 100000.0
    maior_dif = 0.0
    processo_mais_semelhante = None
    diff_semelhante = {}
    for i, p in df_existente.iterrows():
        dif = calcula_diferenca_processos(p,df_processo)
        if dif != {}:
            if dif['pedidos_iguais']:
                dif_total = dif['dif_total']
                print('Diferenca ponderada total com processo %s: %.2f' % (p['id'],dif_total))
                difs.append(dif_total)
                difs_totais += dif_total
                if menor_dif > dif_total:
                    menor_dif = dif_total;
                    processo_mais_semelhante = p['id']
                    diff_semelhante = dif
                if maior_dif < dif_total:
                    maior_dif = dif_total;
    print('Menor diferenca: %.2f' % menor_dif)
    print('Maior diferenca: %.2f' % maior_dif)
    print('Processo mais semelhante: ' + str(processo_mais_semelhante))
    probabilidade_acordos = 0.0
    count = 0
    acordos = 0
    for i, p in df_existente.iterrows():
        dif = calcula_diferenca_processos(p,df_processo)
        if dif != {}:
            if dif['pedidos_iguais']:
                count += 1
                if p['acordo'] == 'S':
                    fator = (dif['dif_total']-menor_dif)/(maior_dif-menor_dif)
                    if fator > 0.5:
                        fator = 0.5
                    print(fator)
                    acordos += (1 - fator)
    print('Quantidade = '+str(count)+' , Acordos = '+str(acordos))
    probabilidade_acordos = acordos/count
    print('Probabilidade ponderada de acordo: %.2f' % probabilidade_acordos)
    insights['probabilidade_acordos'] = probabilidade_acordos
    insights['nomes_reclamante_parecidos'] = diff_semelhante['nomes_reclamante_parecidos']
    insights['nomes_reclamada_parecidos'] = diff_semelhante['nomes_reclamada_parecidos']
    insights['dif_dataAjuizamentoInicial'] = diff_semelhante['dif_dataAjuizamentoInicial']
    insights['dif_meses'] = diff_semelhante['dif_meses']
    insights['dif_propMeses13P'] = diff_semelhante['dif_propMeses13P']
    insights['dif_jornadaSemanal'] = diff_semelhante['dif_jornadaSemanal']
    insights['dif_salario'] = diff_semelhante['dif_salario']
    df_existente = pd.read_csv(DATABASE,header=0)
    df_processo = df_existente.loc[df_existente['id'] == int(processo_mais_semelhante)]
    insights['processo_mais_semelhante'] = recupera_processo(processo_mais_semelhante)
    print(insights)
    return jsonify(insights)

# Chamada de insights de processo
@app.route('/api/v1/processos/ata/<id>', methods=['GET'])
def ata(id):
  print ("Buscando Insight para Processo ID: "+id)
  df_existente = pd.read_csv(DATABASE,header=0)
  df_processo = df_existente.loc[df_existente['id'] == int(id)]
  print('recuperou')
  return calcula_insights(df_processo, df_existente)
  