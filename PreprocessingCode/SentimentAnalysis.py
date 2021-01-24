import preprocessing
import vectorization
import pandas as pd;
import twitterAPIKeys as t

# words_df = preprocessing.getDfFromJSON('All_Beauty.json.gz')
# words_df = words_df[:10] #just testing on a small substring of data
words_df = pd.read_csv('All_Beauty10000.csv')
words_df
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
words_df['documents'] = [" ".join(preprocessing.tokenize(doc)).split(" ") for doc in words_df['documents']]
all_words = vectorization.getAllWordsFromDF(words_df, 'documents')
docList= [" ".join(doc) for doc in words_df['documents']]

# docList = vectorization.ListToString(words_df,'documents')
v,sparceVector = vectorization.vectorize(CountVectorizer, all_words, docList)
sv_array = sparceVector.toarray()

#now we just need to form our labels in whatever way we want them to
words_df["pos_neg"] = words_df['overall'].map(vectorization.binarizeRating)
import sklearn
import numpy as np
xTrain, xTest, yTrain, yTest = sklearn.model_selection.train_test_split(sv_array,list(words_df['pos_neg']),test_size = .3);

xTrain[0].shape

ytrain = np.array(yTrain)
ytest = np.array(yTest)
ytrain.shape
xTrain.shape
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from sklearn.preprocessing import normalize

xTrain
# ytrain=ytrain/5.0
# ytest = ytest/5.0

modelAmazon = Sequential()
modelAmazon.add(Dense(50, activation = "relu", input_shape=(8250, )))
# Hidden - Layers
modelAmazon.add(Dropout(0.3, noise_shape=None, seed=None))
modelAmazon.add(Dense(50, activation = "relu"))
modelAmazon.add(Dropout(0.3, noise_shape=None, seed=None))
modelAmazon.add(Dense(50, activation = "relu"))
# Output- Layer
modelAmazon.add(Dense(1, activation = "sigmoid"))
modelAmazon.summary()

modelAmazon.compile( optimizer = "adam",loss = "binary_crossentropy", metrics = ["accuracy"])

results = modelAmazon.fit(
 xTrain, ytrain,
 epochs= 30,
 batch_size = 700,
 validation_data = (xTest, ytest)
)

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
    print(prepped)
    prepped= " ".join(prepped)
    print(prepped)
    sparce_inputs = v.transform([prepped]).toarray()
    return modelAmazon.predict(sparce_inputs)

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

def getActuallyUsedWords(phrase):
    out = []
    for word in phrase:
        if max(v.transform([word]).toarray()[0]) !=0:
            out = out+ [word]
    return out;

getActuallyUsedWords(["real", "notarealword"])

donald_df = getSentimentOfTopic("donald trump",20);
d = {'col1': [1, 2], 'col2': [3, 4]}
df = pd.DataFrame(data=d)
df

wap, wap_actual_words, wap_analysis = getSentimentOfTopic("wap", 20)
wap_analysis

happy_tweets, used, nnout = getSentimentOfTopic("happy",20)

import tkinter as tk

# root = tk.Tk()
#
# canvas1 = tk.Canvas(root, width=800, height=800)
# canvas1.pack()
#
# entry1 = tk.Entry(root).pack(side="left")
#
#
# class Table:
#     def __init__(self,root,df):
#         # code for creating table
#
#                 canvas1.create_window(300+(j*80),300+(i*10), window=self.e)
#
#
# button1 = tk.Button(text='Analyze Semantics of Entry', command=getSemanticsAnalysis).pack(side="left")
# root.mainloop()

def sentimentCalculation(norm):
    if norm<.2:
        return "Very Negative"
    if norm<.4:
        return "Negative"
    if norm<.5:
        return "Somewhat Negative"
    if norm<.5:
        return "Somewhat Negative"
    if norm<.75:
        return "Neutral"
    if norm<1:
        return "Positive"


class Table:
    def __init__(self,f,df):
        # code for creating table
        for i in range(0,len(df)):
            for j in range(0, len(df.keys())):
                self.e = tk.Label(f,text = df[df.keys()[j]][i],pady=3,
                            font=('Arial',6,'bold'))
                self.e.grid(row=i, column=j)
        score = np.array(list(df['Output'])).mean()
        footer =tk.Label(f,text=  "Results: ")
        footer.grid(row =len(df), column = 0)
        score_label=tk.Label(f,text= str(score))
        score_label.grid(row =len(df), column = 1)
        sentiment=tk.Label(f,text= sentimentCalculation(score))
        sentiment.grid(row =len(df), column = 2)



import tkinter as tk

class MainWindow(tk.Frame):
    counter = 0
    def __init__(self, *args, **kwargs):
        tk.Frame.__init__(self, *args, **kwargs)
        self.entry1 = tk.Entry(self)
        self.entry1.pack(side = "left")
        self.button = tk.Button(self, text="Get sentiment on topic",
                                command=self.create_window)
        self.button.pack(side="right")
    def create_window(self):
        topic = self.entry1.get()
        df = getSentimentOfTopic(topic, 15)
        self.counter += 1
        t = tk.Toplevel(self)
        t.wm_title("Window #%s" % self.counter)
        t = Table(t,df)

root = tk.Tk()
main = MainWindow(root)
main.pack(side="top", fill="both", expand=True)
root.mainloop()
#https://datatofish.com/entry-box-tkinter/
