import numpy as np
import pandas as pd

import preprocessFunctionCollection as pfc

import nltk
from nltk.tokenize import word_tokenize
from nltk import PorterStemmer, WordNetLemmatizer

review_df = pd.read_csv("data/MovieDataset.csv")
print("------------------- Running the preprocessing script -------------------")

# Declaring the numeric class labels for the target feature
target_dict = {"positive":0, "negative":1}

# Replacing the positive/negative words with 0/1 labels
review_df["target"] = review_df["sentiment"].replace(target_dict)

#Dropping the sentiment feature because we already have an encoded version of it
review_df.drop(labels="sentiment", axis=1, inplace=True)

# Lowercasing text in the whole dataframe
review_df["review"] = review_df["review"].str.lower()

# Replacing the html breakline tags
review_df["review"] = review_df["review"].apply(lambda row: pfc.removeHtmlTagsFromtext(row))

# Removing all of the punctuations
review_df["review"] = review_df["review"].apply(lambda row: pfc.removePunctuationsFromText(row))

# Removing leading and tailing whitespaces
review_df["review"] = review_df["review"].apply(lambda row: row.strip())

# Removing numbers 
review_df["review"] = review_df["review"].apply(lambda row: pfc.removeNumbersFromText(row))

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Removing English stopwords
review_df["review"] = review_df["review"].apply(lambda row: pfc.removeStopWordsFromText(row, "english"))

# Getting the 3 most frequent words from the reviews
nMostFrequentWords = pfc.getNMostFrequentWords(review_df,"review", 3)

review_df["review"] = review_df["review"].apply(lambda row: pfc.removeFrequentWords(row, nMostFrequentWords))

nltk.download('punkt')

review_df["review_tokenized"] = review_df["review"].apply(lambda row: word_tokenize(row))

review_df['review_stemmed_lemmatized'] = review_df['review_tokenized'].apply(lambda row: pfc.lemmatizeAndStemtext(row))

final_export_df = review_df[["review_stemmed_lemmatized", "target"]]

final_export_df.to_csv("data/processedData.csv", index=False)
print("------------------- Preprocessing script finished -------------------")
