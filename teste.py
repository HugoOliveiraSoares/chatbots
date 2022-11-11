import wikipedia as wiki
import re 
import pandas as pd

import nltk
from nltk import tokenize  
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('portuguese')
print(stopwords)

wiki.set_lang("pt")
breast_cancer_content = wiki.page("Breast cancer").content

classes = re.findall('=\s(.*)\s=',breast_cancer_content)
raw = re.split(r'=+\s(.*)\s=+',breast_cancer_content)

blocos = []
for r in range(0, len(raw)-1, 2):
    blocos.append(raw[r])

classes.insert(0, 'Cancer de mama')

d = {'classes': [], 'sentencas':[]}
df = pd.DataFrame(d)

for b in range(0, len(blocos)-1):
    sentences = tokenize.sent_tokenize(blocos[b], language='portuguese')
    for s in sentences:
        s = re.sub(r'\n', '', s)
        w = tokenize.word_tokenize(s)
        df = pd.concat([df, pd.Series({'classes': classes[b], 'sentencas': w}).to_frame().T], ignore_index=True) 

# for i, line in df.iterrows():
    # print(line['classes'], line['sentencas'])
print(df.head())