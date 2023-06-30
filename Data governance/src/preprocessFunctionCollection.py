import string
from collections import Counter
import re
from nltk.corpus import stopwords
from nltk import PorterStemmer, WordNetLemmatizer

def removeHtmlTagsFromtext(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', text)

def removePunctuationsFromText(text):
   return ("".join([ch for ch in text if ch not in string.punctuation]))

def removeStopWordsFromText(text, stopWordLanguage):
  stopWords = stopwords.words(stopWordLanguage)
  words = text.split(' ')
  return (" ".join([word for word in words if word not in stopWords]))

def getNMostFrequentWords(dataframe, columnName,n):
  cnt = Counter()
  for text in dataframe[columnName].values:
      for word in text.split():
          cnt[word] += 1
  return cnt.most_common(n)

def removeFrequentWords(text, frequentWords):
    setOfFreqWords = set([w for (w, wc) in frequentWords])
    return " ".join([word for word in str(text).split() if word not in setOfFreqWords])

def removeNumbersFromText(text):
  return re.sub(r'\d+', '', text)

def lemmatizeText(text):
    word_net = WordNetLemmatizer()
    return [word_net.lemmatize(word) for word in text]

def lemmatizeAndStemtext(text):
  lemmatized = lemmatizeText(text)
  stemmer = PorterStemmer()
  return (" ".join([stemmer.stem(word) for word in text]))