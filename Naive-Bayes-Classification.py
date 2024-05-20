import nltk 
import re 
import numpy as np 
import heapq
import sys
import sklearn.metrics as sklearn
np.set_printoptions(threshold=sys.maxsize)
np.set_printoptions(suppress=True)

class Classifier:
  def __init__(self, nameDataset):
    numFreqWords = 1000
    with open(nameDataset) as f:
      content = [line.strip().split('\t') for line in f] 
    
    #Convert text to lower case. Remove all non-word characters. Remove all punctuations.
    for item in content:
      dataset = nltk.sent_tokenize(item[0]) 
      for i in range(len(dataset)): 
        dataset[i] = dataset[i].lower() 
        dataset[i] = re.sub(r'\W', ' ', dataset[i]) 
        dataset[i] = re.sub(r'\s+', ' ', dataset[i]) 
      item[0] = dataset
    #print(content)
    stopWord = ['the', 'and', 'a', 'of', 'is', 'it', 'i', 'this', 'to', 'in', 'was', 'that', 's', 'for', 't', 'with', 'you', 
    'one', 'are', 'so', 'there', 'at', 'an', 'be', 'have', 'by', 'from', 'his', 'he', 'or', 'who', 
    'were', 'can', 'has', 'my', 'they','some', 'your', 'also','its', 'me', 'her', 'any', 'had', 'them',
    'their', 've', 'those', 'been', 'such', 'being', 'she']
    word2count = {} 
    for item in content:
      dataset = item[0]
      for data in dataset: 
        words = nltk.word_tokenize(data) 
        for word in words: 
          if word in stopWord:
            continue
          if word not in word2count.keys(): 
            word2count[word] = 1
          else: 
            word2count[word] += 1

    freq_words = heapq.nlargest(numFreqWords, word2count, key=word2count.get)
    # print(freq_words)
    # print(len(word2count))
    X = [] 
    for item in content:
      dataset = item[0]
      for data in dataset: 
        vector = [] 
        for word in freq_words: 
          if word in nltk.word_tokenize(data): 
            vector.append(1) 
          else: 
            vector.append(0) 
      vector.append(item[1])
      X.append(vector) 

    X = np.asarray(X)
    Y = X[:,numFreqWords]
    X = X[:, 0:numFreqWords]
    self.bagWordTrain = X[:800]
    self.labelTrain = Y[:800]
    self.bagWordTest = X[800:]
    self.labelTest = Y[800:]

    y0 = 0
    y1 = 0
    for yi in self.labelTrain:
      if yi == '0':
        y0 += 1
      elif yi == '1':
        y1 += 1

    self.pY1 = np.log(y1/len(self.labelTrain))
    self.pY0 = np.log(y0/len(self.labelTrain))
  
  def XgivenY(self, sentence, label):
    p = 0
    # print("sen", sentence)
    for i in range(len(sentence)):
      if sentence[i] != '0':
        countY = 0
        countX = 0
        # print(i, self.labelTrain)
        for j in range(len(self.labelTrain)):
          # print("labelTrain[j]", i,  self.labelTrain[j])
          if self.labelTrain[j] == label:
            countY += 1
            # print("bag]", i, self.bagWordTrain[j][i] )
            if self.bagWordTrain[j][i] == '1':
              countX += 1
        # print("countX/countY", countX, countY)
        pxy = countX/countY
        # print("pxy", pxy)
        # if pxy != 0.0:
        #   p *= pxy
        # else:
        #   p *= 1/len(self.labelTrain)
        p += np.log((countX + 1) / (countY + 2))
    return p

  def bayes(self, sentence):
    pXY0 = self.XgivenY(sentence, '0')
    pXY1 = self.XgivenY(sentence, '1')
    pY0givenX = self.pY0 + pXY0
    pY1givenX = self.pY1 + pXY1
    # print(pY0givenX, pY1givenX)
    if pY0givenX > pY1givenX:
      return 0
    return 1

  def calAccuracy(self, sentences, y):
    yTrue = y.astype(np.int).tolist()
    yPred = []
    for s in sentences:
      yPred.append(self.bayes(s))
    cm = sklearn.confusion_matrix(yTrue, yPred)
    accuracy = ((cm[0][0] + cm[1][1]) / len(yTrue)) * 100
    return accuracy

  def main(self):

    accuracyTest = self.calAccuracy(self.bagWordTest, self.labelTest)
    accuracyTrain = self.calAccuracy(self.bagWordTrain, self.labelTrain)

    print("accuracyTrain", accuracyTrain)
    print("accuracyTest", accuracyTest)

print("imdb")
c = Classifier('imdb_labelled.txt')
c.main()

print("yelp")
c = Classifier('yelp_labelled.txt')
c.main()

print("amazon")
c = Classifier('amazon_cells_labelled.txt')
c.main()
