
import pandas as pd
import numpy as np
import json
import math
import glob
from IPython.display import Image
from IPython.display import display, HTML
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from nltk.stem.snowball import SnowballStemmer
from scipy.spatial import distance
from matplotlib import pyplot as plt
from wordcloud import WordCloud
import ipywidgets as widgets

#set the root directory
root_dir = 'C:\\Users\\Josh\\OneDrive - Microsoft\\CORD-19\\archive'
#print(root_dir)

#read in the metadata csv file
df_meta = pd.read_csv(f'{root_dir}//metadata.csv')
#print(df)

#print(df_meta.head())
##print(df_meta.describe())
#print(df_meta.info())
print ('Number of instances = %d' % df_meta.shape[0])
print ('Number of attributes = %d' % df_meta.shape[1])


#Get a count of all json docs in folder
#all_json = glob.glob('C:\\Users\\Josh\OneDrive - Microsoft\\CORD-19\\archive\\document_parses\\pdf_json\\*.json', recursive=True)
doc_paths = glob.glob(f'{root_dir}/*/temp/*.json', recursive=True)

print(doc_paths)

import os
os.listdir("C:\\Users\\Josh\\OneDrive - Microsoft\\CORD-19\\archive")

df_meta.sha.fillna("", inplace=True)

#d.sha.fillna("", inplace=True)

#find article text where available
def get_text(sha):
    if sha == "":
        return ""
    document_path = [x for x in doc_paths if sha in x]
    if not document_path:
        return ""
    with open(document_path[0]) as f:
        file = json.load(f)
        full_text = []
        #body text and abstract iteration
        for part in ['abstract', 'body_text']:
            # paragraph
            for text_part in file[part]:
                text = text_part['text']
                # citation removal
                for citation in text_part['cite_spans']:
                    text = text.replace(citation['text'], "")
                full_text.append(text)
            
        return str.join(' ', full_text)

df_meta['text'] = df_meta.apply(lambda x: get_text(x.sha), axis=1)
print(df_meta['text'])
