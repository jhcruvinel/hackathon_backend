from flask import Flask
import os
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
portuguese_stops = set(stopwords.words('portuguese'))
ignore = ['@', '.', '!','?',',','$','-','\'s','g','(',')','[',']','``',':','http','html','//members']
from nltk.stem import SnowballStemmer
# Cria o Stemmer
stemmer = SnowballStemmer('english')
def preprocess_text(content):
    content.translate(str.maketrans('', '', string.punctuation))
    word_tokens = nltk.word_tokenize(content)
    #print(len(word_tokens),word_tokens)
    word_tokens2 = [w.lower() for w in word_tokens if w not in portuguese_stops]
    #print(len(word_tokens2),word_tokens2)
    word_tokens3 = [w for w in word_tokens2 if w not in ignore]
    #print(len(word_tokens3),word_tokens3)
    word_tokens4 = [stemmer.stem(w) for w in word_tokens3]
    #print(len(word_tokens4),word_tokens4)
    #return word_tokens4
    return ' '.join(word_tokens4)

df = pd.read_csv('dados_teste.csv',delimiter='|')
df['nome'] = df['nome'].apply(preprocess_text)
texts = df['nome']

app = Flask(__name__)

@app.route('/')
def index():
  return 'Server Works!'
  
@app.route('/similaridade', methods=['POST'])
def similaridade():
  content = request.json
  print ("Request: "+content)
  print ("Nome: "+content['nome'])
  text = preprocess_text(content['nome'])

  return 'Hello from Server'

print(nomes)


#Regex para encontrar tokens
REGEX_WORD = re.compile(r'\w+')
#Numero de tokens em sequencia
N_GRAM_TOKEN = 3

#Faz a normalizacao do texto removendo espacos a mais e tirando acentos
def text_normalizer(src):
    return re.sub('\s+', ' ',
                normalize('NFKD', src)
                   .encode('ASCII','ignore')
                   .decode('ASCII')
           ).lower().strip()

#Faz o calculo de similaridade baseada no coseno
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

#Monta o vetor de frequencia da sentenca
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

#Obtem a similaridade entre duas sentencas
def get_sentence_similarity(sentence1, sentence2, use_text_bigram=False):
    vector1 = sentence_to_vector(text_normalizer(sentence1), use_text_bigram)
    vector2 = sentence_to_vector(text_normalizer(sentence2), use_text_bigram)
    return cosine_similarity(vector1, vector2)

#Metodo de gerar bigramas de uma string
def get_text_bigrams(src):
    s = src.lower()
    return [s[i:i+2] for i in range(len(s) - 1)]

if __name__ == "__main__":
    w1 = 'COMERCIAL CASA DOS FRIOS - USAR LICINIO DIAS'
    words = [
        'ARES DOS ANDES - EXPORTACAO & IMPORTACAO LTDA', 
        'ADEGA DOS TRES IMPORTADORA', 
        'BODEGAS DE LOS ANDES COMERCIO DE VINHOS LTDA', 
        'ALL WINE IMPORTADORA'
    ]

    print('Busca: ' + w1)

    #Nivel de aceite (40%)
    cutoff = 0.40
    #Sentenças similares
    result = []

    for w2 in words:
        print('\nCosine Sentence --- ' + w2)

        #Calculo usando similaridade do coseno com apenas tokens
        similarity_sentence = get_sentence_similarity(w1, w2)
        print('\tSimilarity sentence: ' + str(similarity_sentence))

        #Calculo usando similaridade do coseno com tokens e com ngramas do texto
        similarity_sentence_text_bigram = get_sentence_similarity(w1, w2, use_text_bigram=True)
        print('\tSimilarity sentence: ' + str(similarity_sentence_text_bigram))

        if similarity_sentence >= cutoff:
            result.append((w2, similarity_sentence))

    print('\nResultado:')
    #Exibe resultados
    for data in result:
        print(data)


# In[153]:


texts.count()


# In[155]:


max_sim = 0
most_similar = ''
test = 'John da silva'
for w in texts:
    sim = get_sentence_similarity(test, w, use_text_bigram=True)
    if sim > max_sim:
        max_sim = sim
        most_similar = w
        print(str(sim)+' '+w)
print ('Mais similar a ' + test + ' = ' + most_similar)


# In[159]:


get_ipython().system('pip install spacy')


# In[ ]:


import spacy
nlp = spacy.load('pt')


# In[ ]:





# In[155]:


import spacy
nlp = spacy.load('pt')
 
doc1 = nlp(u'Hello this is document similarity calculation')
doc2 = nlp(u'Hello this is python similarity calculation')
doc3 = nlp(u'Hi there')
 
print (doc1.similarity(doc2)) 
print (doc2.similarity(doc3)) 
print (doc1.similarity(doc3))  
 
max_sim = 0
most_similar = ''
test = 'John da silva'
for w in texts:
    sim = get_sentence_similarity(test, w, use_text_bigram=True)
    if sim > max_sim:
        max_sim = sim
        most_similar = w
        print(str(sim)+' '+w)
print ('Mais similar a ' + test + ' = ' + most_similar)


# In[156]:


get_ipython().system('pip install flask')


# In[ ]:




