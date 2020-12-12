#Tutorial Reference:  https://towardsdatascience.com/natural-language-processing-feature-engineering-using-tf-idf-e8b9d00e7e76

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

documentA = 'the man went out for a walk'
documentB = 'the children sat around the fire children'

#Create a list of words out of the sentences
bagOfWordsA = documentA.split(' ')
bagOfWordsB = documentB.split(' ')

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