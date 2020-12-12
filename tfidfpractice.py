#Tutorial Reference:  https://towardsdatascience.com/natural-language-processing-feature-engineering-using-tf-idf-e8b9d00e7e76

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

#document is a string
documentA = 'the man went out for a walk'
documentB = 'the children sat around the fire children'

print (type(documentA))
#Create a list of words out of the sentences
bagOfWordsA = documentA.split(' ')
bagOfWordsB = documentB.split(' ')
print('split a')
print(type(bagOfWordsA))
print(bagOfWordsA)
#print (type(bagOfWordsA))
#print (bagOfWordsA)

#Cast words to a set, this will automatically remove any duplicate words
uniqueWords = set(bagOfWordsA).union(set(bagOfWordsB))
print('uniquewords')
print(uniqueWords)

#Next, we’ll create a dictionary of words and their occurence for each document in the corpus (collection of documents).
#Create a dictionary of all the words in all the documents
numOfWordsA = dict.fromkeys(uniqueWords, 0)
print('numofWordsA')
print(numOfWordsA)

for word in bagOfWordsA:
    numOfWordsA[word] += 1

print('numOfWordsA')
print(numOfWordsA)

numOfWordsB = dict.fromkeys(uniqueWords, 0)
for word in bagOfWordsB:
    numOfWordsB[word] += 1

print('numOfWordsB')
print (numOfWordsB)


#In natural language processing, useless words are referred to as stop words. The python natural language toolkit library provides a list of english stop words.
from nltk.corpus import stopwords
stopwords.words('english')
#print(stopwords.words('english'))


#TF Term Frequency
#The number of times a word appears in a document divded by the total number of words in the document. Every document has its own term frequency.
def computeTF(wordDict, bagOfWords):
    tfDict = {}
    bagOfWordsCount = len(bagOfWords)
    print(bagOfWordsCount)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bagOfWordsCount)
    return tfDict


print (numOfWordsA)
print (bagOfWordsA)

tfA = computeTF(numOfWordsA, bagOfWordsA)
tfB = computeTF(numOfWordsB, bagOfWordsB)
print(tfA)

#IDF Inverse Data Frequency
#The log of the number of documents divided by the number of documents that contain the word w. Inverse data frequency determines the weight of rare words across all documents in the corpus.
def computeIDF(documents):
    import math
    N = len(documents)
    
    idfDict = dict.fromkeys(documents[0].keys(), 0)
    for document in documents:
        for word, val in document.items():
            if val > 0:
                idfDict[word] += 1
    
    for word, val in idfDict.items():
        idfDict[word] = math.log(N / float(val))
    return idfDict

#The IDF is computed once for all documents.
idfs = computeIDF([numOfWordsA, numOfWordsB])
print (idfs)

#Lastly, the TF-IDF is simply the TF multiplied by IDF.
def computeTFIDF(tfBagOfWords, idfs):
    tfidf = {}
    for word, val in tfBagOfWords.items():
        tfidf[word] = val * idfs[word]
    return tfidf

#Finally, we can compute the TF-IDF scores for all the words in the corpus.
tfidfA = computeTFIDF(tfA, idfs)
tfidfB = computeTFIDF(tfB, idfs)
df = pd.DataFrame([tfidfA, tfidfB])
#print(df)

vectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform([documentA, documentB])
feature_names = vectorizer.get_feature_names()
dense = vectors.todense()
denselist = dense.tolist()
df = pd.DataFrame(denselist, columns=feature_names)
print(df)