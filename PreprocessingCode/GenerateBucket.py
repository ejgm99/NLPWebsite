import preprocessing
import vectorization
import pandas as pd;
import twitterAPIKeys as t

# words_df = preprocessing.getDfFromJSON('All_Beauty.json.gz')
# words_df = words_df[:10] #just testing on a small substring of data
words_df = pd.read_csv('All_Beauty10000.csv');
#at this point words df is just a column of review texts and their associated scores
# preprocessing.preprocessForSentimentAnalsis(words_df['reviewText'][4], preprocessing.stopwords,preprocessing.lemmatizer);
# words_df['documents']=words_df['reviewText'].map(preprocessing.preprocess
# words_df = words_df[words_df['documents'] != False]
#documents column now is just the preprocessed words stripped of fluff, ready to be turned into a sparse matrix
#First we just need a list of all of the words

#now we construct our CountVectorizer

from sklearn.feature_extraction.text import CountVectorizer;
from sklearn.feature_extraction.text import TfidfVectorizer;


#generate a sparce array from the things
words_df['documents'] = [" ".join(preprocessing.tokenize(doc)).split(" ") for doc in words_df['documents']];
all_words = vectorization.getAllWordsFromDF(words_df, 'documents');
docList= [" ".join(doc) for doc in words_df['documents']];

# docList = vectorization.ListToString(words_df,'documents')
vectorizer, sparceVector = vectorization.vectorize(CountVectorizer, all_words, docList);
type(vectorizer.vocabulary)
sv_array = sparceVector.toarray();

#now we just need to form our labels in whatever way we want them to
words_df["pos_neg"] = words_df['overall'].map(vectorization.binarizeRating);
import sklearn
import numpy as np
xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(sv_array,list(words_df['pos_neg']),test_size = .3);

ytrain = np.array(yTrain)
ytest = np.array(yTest)
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.preprocessing import normalize


modelAmazon = Sequential();
modelAmazon.add(Dense(50, activation = "relu", input_shape=(8250, )));
# Hidden - Layers
modelAmazon.add(Dropout(0.3, noise_shape=None, seed=None));
modelAmazon.add(Dense(50, activation = "relu"));
modelAmazon.add(Dropout(0.3, noise_shape=None, seed=None));
modelAmazon.add(Dense(50, activation = "relu"));
# Output- Layer
modelAmazon.add(Dense(1, activation = "sigmoid"));
modelAmazon.summary();

modelAmazon.compile( optimizer = "adam",loss = "binary_crossentropy", metrics = ["accuracy"]);

results = modelAmazon.fit(
 xTrain, ytrain,
 epochs= 30,
 batch_size = 700,
 validation_data = (xTest, ytest)
);

print(np.mean(results.history["accuracy"]))

from keras.datasets import imdb
index = imdb.get_word_index()

def transformToIMDB(doc,numWords):
    sparce_array = np.zeros(10000);
    for word in doc:
        if index[word]<numWords:
            print(index[word])



(training_data, training_targets), (testing_data, testing_targets) = imdb.load_data(num_words = 10000)
len(training_data[6])

len(training_data[6])
data = np.concatenate((training_data, testing_data), axis=0)
targets = np.concatenate((training_targets, testing_targets), axis=0)

def vectorize(sequences, dimension = 10000):
  results = np.zeros((len(sequences), dimension))
  for i, sequence in enumerate(sequences):
    results[i, sequence] = 1
  return results

data = vectorize(data)
training_targets[8]
targets = np.array(targets).astype("float32")

test_x = data[:10000]
type(test_x[0][0])
test_y = targets[:10000]
train_x = data[10000:]
train_y = targets[10000:]


def runOnSample(input):
    prepped = preprocessing.preprocessForSentimentAnalsis(input,preprocessing.stopwords,preprocessing.lemmatizer)
    prepped= " ".join(prepped)
    sparce_inputs = vectorizer.transform([prepped]).toarray()
    return modelAmazon.predict(sparce_inputs)[0][0]




def getSentimentOfTopic(topic, nTweets):
    tweets = t.getTopic(topic, nTweets)
    print(tweets)
    prepped = [ preprocessing.preprocessForSentimentAnalsis(tweet,preprocessing.stopwords,preprocessing.lemmatizer) for tweet in tweets]
    actual_words= [getActuallyUsedWords(doc) for doc in prepped]
    prepped= [" ".join(doc) for doc in prepped]
    sparce_inputs = v.transform(prepped).toarray()
    output =  modelAmazon.predict(sparce_inputs)
    output = list(np.squeeze(output))
    print(output)
    return  pd.DataFrame(data= {"Tweets" : tweets,"Trained Words":actual_words,"Output" : output})


def sentimentCalculation(norm):
    if norm<.2:
        return "Very Negative"
    if norm<.4:
        return "Negative"
    if norm<.5:
        return "Somewhat Negative"
    if norm<.75:
        return "Neutral"
    if norm<1:
        return "Positive"

def getSemanticsAnalysis():
    text = entry1.get()
    score = runOnSample(text)
    label1 = tk.Label(root, text=str(score)+" ("+sentimentCalculation(score)+")         " )
    canvas1.create_window(200, 230, window=label1)


def getActuallyUsedWords(phrase):
    out = []
    for word in phrase:
        if max(v.transform([word]).toarray()[0]) !=0:
            out = out+ [word]
    return out;


mylist = ["a", "b", "a", "c", "c"]
mylist = list(dict.fromkeys(all_words))
print(len(mylist))

all_words = mylist
all_words_score = [runOnSample(word) for word in all_words[:1000]]
all_words_score
df = pd.DataFrame({"Word": all_words[:1000], "Score": all_words_score})
df.to_csv("Bucket.csv")


all_words_score = [runOnSample(word) for word in all_words[1000:2000]]
all_words_score
df = pd.DataFrame({"Word": all_words[1000:2000], "Score": all_words_score})
df.to_csv("Bucket1.csv")


all_words_score = [runOnSample(word) for word in all_words[2000:3000]]
all_words_score
df = pd.DataFrame({"Word": all_words[2000:3000], "Score": all_words_score})
df.to_csv("Bucket2.csv")


all_words_score = [runOnSample(word) for word in all_words[3000:4000]]
all_words_score
df = pd.DataFrame({"Word": all_words[3000:4000], "Score": all_words_score})
df.to_csv("Bucket3.csv")


all_words_score = [runOnSample(word) for word in all_words[4000:5000]]
all_words_score
df = pd.DataFrame({"Word": all_words[4000:5000], "Score": all_words_score})
df.to_csv("Bucket4.csv")


all_words_score = [runOnSample(word) for word in all_words[5000:6000]]
all_words_score
df = pd.DataFrame({"Word": all_words[5000:6000], "Score": all_words_score})
df.to_csv("Bucket5.csv")


all_words_score = [runOnSample(word) for word in all_words[6000:7000]]
all_words_score
df = pd.DataFrame({"Word": all_words[6000:7000], "Score": all_words_score})
df.to_csv("Bucket6.csv")

all_words_score = [runOnSample(word) for word in all_words[7000:8000]]
all_words_score
df = pd.DataFrame({"Word": all_words[7000:8000], "Score": all_words_score})
df.to_csv("Bucket7.csv")


all_words_score = [runOnSample(word) for word in all_words[8000:]]
all_words_score
df = pd.DataFrame({"Word": all_words[8000:9000], "Score": all_words_score})
df.to_csv("Bucket8.csv")
