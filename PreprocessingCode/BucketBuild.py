import preprocessing
import vectorization
import pandas as pd;
import twitterAPIKeys as t
import numpy as np
# words_df = preprocessing.getDfFromJSON('All_Beauty.json.gz')
# words_df = words_df[:10] #just testing on a small substring of data

def sentimentCalculation(norm):
    if norm<.2:
        return "Very Negative"
    if norm<.4:
        return "Negative"
    if norm<.57:
        return8 "Somewhat Negative"
    if norm<.75:
        return "Neutral"
    if norm<1:
        return "Positive"


def fixCounts(c):
    for i in range(len(c),0):
        print(i)


df=pd.read_csv("Bucket.csv")
df1=pd.read_csv("Bucket1.csv")
df2=pd.read_csv("Bucket2.csv")
df3=pd.read_csv("Bucket3.csv")
df4=pd.read_csv("Bucket4.csv")
df5=pd.read_csv("Bucket5.csv")
df6=pd.read_csv("Bucket6.csv")
df7=pd.read_csv("Bucket7.csv")
df8=pd.read_csv("Bucket8.csv")

df
all_scores = list(df['Score'])+list(df1['Score'])+list(df2['Score'])+list(df3['Score'])+list(df4['Score'])+list(df5['Score'])+list(df6['Score'])+list(df7['Score'])+list(df8['Score'])
len(all_scores)

buckets = [.25,.5,.6,.75,1]
counts = [[len(df[(df['Score'] >(bucket-.25)) & (df["Score"]<bucket )]) for bucket in buckets] for df in dfs]
counts = [np.array(count) for count in counts]
sum(counts)

dfs = [df,df1,df2,df3,df4,df5,df6,df7,df8]

relavant_words = df[['Score']['']+df1+df2+df3+df4+df5+df6+df7+df8
buckets = [.25,.5,.6,.75,1]
counts = [[len(df[(df['Score'] >(bucket-.25)) & (df["Score"]<bucket )]) for bucket in buckets] for df in dfs]
counts = [np.array(count) for count in counts]
sum(counts)

import matplotlib.pyplot as plt
plt.axis([0,1,0,100])
h= plt.hist(all_scores,bins=30)[0]

len(h)
h[:29]
h
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
range(0,1,29).toarray()
ax.bar(range(0,1,len(h[:29])),h[:29])
plt.show()
