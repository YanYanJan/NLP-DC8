import numpy as np # linear algebra
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import MultinomialNB
import nltk.classify.util, nltk.metrics
import collections
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier

init_data = pd.read_csv("winemag-data-130k-v2.csv")
print("Length of dataframe before duplicates are removed:", len(init_data))

parsed_data = init_data[init_data.duplicated('description', keep=False)]
print("Length of dataframe after duplicates are removed:", len(parsed_data))

parsed_data.dropna(subset=['description', 'points'])
print("Length of dataframe after NaNs are removed:", len(parsed_data))

parsed_data.head()

dp = parsed_data[['description','points']]
#dp.info()
#dp.head()

dp = dp.assign(description_length = dp['description'].apply(len))
#dp.info()
#dp.head()

#Transform method taking points as param
def transform_points_simplified(points):
    if points < 84:
        return 1
    elif points >= 84 and points < 88:
        return 2
    elif points >= 88 and points < 92:
        return 3
    elif points >= 92 and points < 96:
        return 4
    else:
        return 5

#Applying transform method and assigning result to new column "points_simplified"
dp = dp.assign(points_simplified = dp['points'].apply(transform_points_simplified))
#dp.head()

X = dp['description']
y = dp['points_simplified']

# NaiveBayesClassifier


from io import StringIO
col = ['points_simplified', 'description']
df = dp[col]
df = df[pd.notnull(df['description'])]

df['category_id'] = df['points_simplified'].factorize()[0]
category_id_df = df[['points_simplified', 'category_id']].drop_duplicates().sort_values('category_id')
category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'points_simplified']].values)

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer(sublinear_tf=True, min_df=5, norm='l2', encoding='latin-1', ngram_range=(1, 2), stop_words='english')
features = tfidf.fit_transform(df.description).toarray()
labels = df.category_id
print(features.shape)

from sklearn.feature_selection import chi2
import numpy as np
N = 5
for point, category_id in sorted(category_to_id.items()):
  features_chi2 = chi2(features, labels == category_id)
  indices = np.argsort(features_chi2[0])
  feature_names = np.array(tfidf.get_feature_names())[indices]
  unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
  bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
  print("# '{}':".format(point))
  print("  . Most correlated unigrams:\n. {}".format('\n. '.join(unigrams[-N:])))
  print("  . Most correlated bigrams:\n. {}".format('\n. '.join(bigrams[-N:])))

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
clf = MultinomialNB().fit(X_train_tfidf, y_train)
predict = clf.predict(count_vect.transform(X_test))
print(classification_report(y_test, predict))


## comparison with randomforestclassifier
vectorizer = CountVectorizer()
vectorizer.fit(X)
#print(vectorizer.vocabulary_)

X = vectorizer.transform(X)
print('Shape of Sparse Matrix: ', X.shape)
print('Amount of Non-Zero occurrences: ', X.nnz)
# Percentage of non-zero values
density = (100.0 * X.nnz / (X.shape[0] * X.shape[1]))
print('Density: {}'.format((density)))

# Training the model
X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size=0.1, random_state=101)
rfc = RandomForestClassifier()
rfc.fit(X_train1, y_train1)

# Testing the model
predictions = rfc.predict(X_test1)
print(classification_report(y_test1, predictions))