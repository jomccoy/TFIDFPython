
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

root_dir = 'C:\\Users\\Josh\\OneDrive - Microsoft\\CORD-19\\archive'
print(root_dir)

df = pd.read_csv(f'{root_dir}//metadata.csv')
print(df)