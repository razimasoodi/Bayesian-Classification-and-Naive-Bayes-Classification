import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
  
class Bayesian():

  def __init__(self, mean0, cov0, mean1, cov1, mean2, cov2, numData, numTrain):
    
    x0, y0 = np.random.multivariate_normal(mean0, cov0, numData).T
    x1, y1 = np.random.multivariate_normal(mean1, cov1, numData).T
    x2, y2 = np.random.multivariate_normal(mean2, cov2, numData).T

    self.cov0 = cov0
    self.cov1 = cov1
    self.cov2 = cov2
    self.mean0 = mean0
    self.mean1 = mean1
    self.mean2 = mean2

    self.meanClass0 = np.array( [(sum(x0[:numTrain])/numTrain), (sum(y0[:numTrain])/numTrain)] )
    self.meanClass1 = np.array( [(sum(x1[:numTrain])/numTrain), (sum(y1[:numTrain])/numTrain)] )
    self.meanClass2 = np.array( [(sum(x2[:numTrain])/numTrain), (sum(y2[:numTrain])/numTrain)] )
    self.phiClass0 = 1/3
    self.phiClass1 = 1/3
    self.phiClass2 = 1/3
    self.countData = numTrain * 3

    features = []
    label =[]
    demeanX0 = []
    demeanX1 = []
    demeanX2 = []
    testFeature = []
    testLabel = []
    for i in range(numTrain):
      features.append([x0[i], y0[i]]) 
      label.append([0])
      demeanX0.append([x0[i] - self.meanClass0[0], y0[i] - self.meanClass0[1]])
      features.append([x1[i], y1[i]]) 
      label.append([1])
      demeanX1.append([x1[i] - self.meanClass1[0], y1[i] - self.meanClass1[1]])
      features.append([x2[i], y2[i]])  
      label.append([2])
      demeanX2.append([x2[i] - self.meanClass2[0], y2[i] - self.meanClass2[1]])

    for j in range(numTrain, numData):
      testFeature.append([x0[j], y0[j]]) 
      testLabel.append([0])
      testFeature.append([x1[j], y1[j]]) 
      testLabel.append([1])
      testFeature.append([x2[j], y2[j]]) 
      testLabel.append([2])

    self.features = np.array(features)
    self.label = np.array(label)
    self.testFeature = np.array(testFeature)
    self.testLabel = np.array(testLabel)

    demeanX0 = np.array(demeanX0)
    demeanX1 = np.array(demeanX1)
    demeanX2 = np.array(demeanX2)
    demeanX0 = demeanX0/np.linalg.norm(demeanX0, ord=2, axis=1, keepdims=True)
    demeanX1 = demeanX1/np.linalg.norm(demeanX1, ord=2, axis=1, keepdims=True)
    demeanX2 = demeanX2/np.linalg.norm(demeanX2, ord=2, axis=1, keepdims=True)
    self.sigma0 = np.round( (1/self.countData) * (demeanX0.T @ demeanX0) , 5)
    self.sigma1 = np.round( (1/self.countData) * (demeanX1.T @ demeanX1) , 5)
    self.sigma2 = np.round( (1/self.countData) * (demeanX2.T @ demeanX2) , 5)

  def classifier(self, n, dataX, dataY):
    predict = []
    coefficient0 = 1 / ( ((2 * np.pi) ** (n/2)) * np.sqrt(abs(np.linalg.det(self.sigma0))) )
    coefficient1 = 1 / ( ((2 * np.pi) ** (n/2)) * np.sqrt(abs(np.linalg.det(self.sigma1))) )
    coefficient2 = 1 / ( ((2 * np.pi) ** (n/2)) * np.sqrt(abs(np.linalg.det(self.sigma2))) )
    correct = 0
    for i in range(len(dataX)):

      demean0 = dataX[i] - self.meanClass0
      demean1 = dataX[i] - self.meanClass1
      demean2 = dataX[i] - self.meanClass2

      sigmaInvers0 = np.linalg.inv(self.sigma0)
      sigmaInvers1 = np.linalg.inv(self.sigma1)
      sigmaInvers2 = np.linalg.inv(self.sigma2)

      pXY0 = coefficient0 * (np.exp(  -0.5 * ( (demean0.T @ sigmaInvers0) @ demean0 )  ))
      pXY1 = coefficient1 * (np.exp(  -0.5 * ( (demean1.T @ sigmaInvers1) @ demean1 )  ))
      pXY2 = coefficient2 * (np.exp(  -0.5 * ( (demean2.T @ sigmaInvers2) @ demean2 )  ))

      l0 = pXY0 * self.phiClass0
      l1 = pXY1 * self.phiClass1
      l2 = pXY2 * self.phiClass2
      
      res = max(l0,l1,l2)

      if res == l0:
        predict.append(0)
        if dataY[i][0] == 0:
          correct += 1
      elif res == l1:
        predict.append(1)
        if dataY[i][0] == 1:
          correct += 1
      elif res == l2:
        predict.append(2)
        if dataY[i][0] == 2:
          correct += 1

    accuracy = correct / len(dataX) * 100

    print(accuracy, "%")

  def plot1(self):
    class2 = []
    class1 = []
    class0 = []
    for i in range(len(self.label)):
      if self.label[i][0] == 1:
        class1.append(self.features[i])
      elif self.label[i][0] == 0:
        class0.append(self.features[i])
      elif self.label[i][0] == 2:
        class2.append(self.features[i])
    # train
    plt.scatter(*zip(*class0), color = "#ff4d4d", label = 'train class 0') # y = 0, train
    plt.scatter(*zip(*class1), color = "#80dfff", label = 'train class 1') # y = 1, train
    plt.scatter(*zip(*class2), color = "#32CD32", label = 'train class 3') # y = 2, train
    class1 = []
    class0 = []
    class2 = []
    for i in range(len(self.testLabel)):
      if self.testLabel[i][0] == 1:
        class1.append(self.testFeature[i])
      elif self.testLabel[i][0] == 0:
        class0.append(self.testFeature[i])
      elif self.testLabel[i][0] == 2:
        class2.append(self.testFeature[i])
    #test
    plt.scatter(*zip(*class0), color = "#ff8080")  # y = 0, test golbe e
    plt.scatter(*zip(*class1), color = "#00ace6")  # y = 1, test abi 
    plt.scatter(*zip(*class2), color = "#00FF7F") # y = 2, train sabz
    # plt.show()

    #descision boundry
    c = ( np.log(self.phiClass0/self.phiClass1) ) 
    - ( 0.5 * np.log( np.linalg.det(self.sigma0)/ np.linalg.det(self.sigma1) ) ) 
    + ( self.meanClass0.T @ (np.linalg.inv(self.sigma0) @ self.meanClass0) ) 
    - ( self.meanClass1.T @ (np.linalg.inv(self.sigma1) @ self.meanClass1) )
    b = -2 * ( (np.linalg.inv(self.sigma0) @ self.meanClass0) - (np.linalg.inv(self.sigma1) @ self.meanClass1) )
    a = -0.5 * (np.linalg.inv(self.sigma0) - np.linalg.inv(self.sigma1))
    d = a[0][0]
    k = a[1][0]
    f = a[0][1]
    g = a[1][1]
    h = b[0]
    b = b[1]
    x01 = np.linspace(1.0, 9.0, num = 80)
    sq = np.sqrt(x01**2 * k**2 + 2*x01**2*f*k + 2*x01*b*k - 4*g*c - 4*x01*g*h - 4*x01**2*d*g + x01**2 * f**2 + 2*x01*b*f + b**2)
    y01 = (-1 * (0.5 * sq) / g) - ((0.5*x01*k)/g) - ((0.5*x01*f)/g) - ((0.5*b)/g)
    # y01 = -1 * ( (+1 * np.sqrt(x01**2 *(k**2 + 2*f*k - 4*d*g + f**2) + x01*(2*b*k - 4*g*h + 2*b*f) - 4*g*c + b**2) + x01*(k+f) + b) / (2*g) )

    c = ( np.log(self.phiClass2/self.phiClass1) ) - ( 0.5 * np.log( np.linalg.det(self.sigma2)/ np.linalg.det(self.sigma1) ) ) + ( self.meanClass2.T @ (np.linalg.inv(self.sigma2) @ self.meanClass2) ) - ( self.meanClass1.T @ (np.linalg.inv(self.sigma1) @ self.meanClass1) )
    b = -2 * ( (np.linalg.inv(self.sigma2) @ self.meanClass2) - (np.linalg.inv(self.sigma1) @ self.meanClass1) )
    a = -0.5 * (np.linalg.inv(self.sigma2) - np.linalg.inv(self.sigma1))
    d = a[0][0]
    k = a[1][0]
    f = a[0][1]
    g = a[1][1]
    h = b[0]
    b = b[1]
    x21 = np.linspace(1.0, 9.0, num = 80)
    sq = np.sqrt(x21**2 * k**2 + 2*x21**2*f*k + 2*x21*b*k - 4*g*c - 4*x21*g*h - 4*x21**2*d*g + x21**2 * f**2 + 2*x21*b*f + b**2)
    y21 = (-1 * (0.5 * sq) / g) - ((0.5*x21*k)/g) - ((0.5*x21*f)/g) - ((0.5*b)/g)
    # y21 = -1 * ( (+1 * np.sqrt(x21**2 *(k**2 + 2*f*k - 4*d*g + f**2) + x21*(2*b*k - 4*g*h + 2*b*f) - 4*g*c + b**2) + x21*(k+f) + b) / (2*g) )

    c = ( np.log(self.phiClass2/self.phiClass0) ) - ( 0.5 * np.log( np.linalg.det(self.sigma2)/ np.linalg.det(self.sigma0) ) ) + ( self.meanClass2.T @ (np.linalg.inv(self.sigma2) @ self.meanClass2) ) - ( self.meanClass0.T @ (np.linalg.inv(self.sigma0) @ self.meanClass0) )
    b = -2 * ( (np.linalg.inv(self.sigma2) @ self.meanClass2) - (np.linalg.inv(self.sigma0) @ self.meanClass0) )
    a = -0.5 * (np.linalg.inv(self.sigma2) - np.linalg.inv(self.sigma0))
    d = a[0][0]
    k = a[1][0]
    f = a[0][1]
    g = a[1][1]
    h = b[0]
    b = b[1]
    x20 = np.linspace(1.0, 9.0, num = 80)
    sq = np.sqrt(x20**2 * k**2 + 2*x20**2*f*k + 2*x20*b*k - 4*g*c - 4*x20*g*h - 4*x20**2*d*g + x20**2 * f**2 + 2*x20*b*f + b**2)
    y20 = (-1 * (0.5 * sq) / g) - ((0.5*x20*k)/g) - ((0.5*x20*f)/g) - ((0.5*b)/g)
    # y20 = -1 * ( (+1 * np.sqrt(x20**2 *(k**2 + 2*f*k - 4*d*g + f**2) + x20*(2*b*k - 4*g*h + 2*b*f) - 4*g*c + b**2) + x20*(k+f) + b) / (2*g) )


    xSame = -1
    for i in range(80):
      if (round(y01[i]) == round(y21[i]) and round(y21[i]) == round(y20[i])):
        # print(x01[i], y01[i])
        xSame = i
        break

    if xSame == -1:
      plt.plot(x01, y01, color = "#000000")
      plt.plot(x21, y21, color = "#000000")
      plt.plot(x20, y20, color = "#000000")
      plt.show()
      print("please try again for better result")
      
    else:
      plt.plot(x01[:xSame+1], y01[:xSame+1], color = "#000000")
      plt.plot(x21[xSame:], y21[xSame:], color = "#000000")
      plt.plot(x20[20:xSame+1], y20[20:xSame+1], color = "#000000")
      plt.show()

  def plot2(self):
    class2 = []
    class1 = []
    class0 = []
    for i in range(len(self.label)):
      if self.label[i][0] == 1:
        class1.append(self.features[i])
      elif self.label[i][0] == 0:
        class0.append(self.features[i])
      elif self.label[i][0] == 2:
        class2.append(self.features[i])
    # train
    plt.scatter(*zip(*class0), color = "#ff4d4d", label = 'train class 0') # y = 0, train
    plt.scatter(*zip(*class1), color = "#80dfff", label = 'train class 1') # y = 1, train
    plt.scatter(*zip(*class2), color = "#32CD32", label = 'train class 3') # y = 2, train
    class1 = []
    class0 = []
    class2 = []
    for i in range(len(self.testLabel)):
      if self.testLabel[i][0] == 1:
        class1.append(self.testFeature[i])
      elif self.testLabel[i][0] == 0:
        class0.append(self.testFeature[i])
      elif self.testLabel[i][0] == 2:
        class2.append(self.testFeature[i])
    #test
    plt.scatter(*zip(*class0), color = "#ff8080")  # y = 0, test golbe e
    plt.scatter(*zip(*class1), color = "#00ace6")  # y = 1, test abi 
    plt.scatter(*zip(*class2), color = "#00FF7F") # y = 2, train sabz
    # plt.show()

    #descision boundry
    c = ( np.log(self.phiClass0/self.phiClass1) ) - ( 0.5 * np.log( np.linalg.det(self.sigma0)/ np.linalg.det(self.sigma1) ) ) + ( self.meanClass0.T @ (np.linalg.inv(self.sigma0) @ self.meanClass0) ) - ( self.meanClass1.T @ (np.linalg.inv(self.sigma1) @ self.meanClass1) )
    b = -2 * ( (np.linalg.inv(self.sigma0) @ self.meanClass0) - (np.linalg.inv(self.sigma1) @ self.meanClass1) )
    a = -0.5 * (np.linalg.inv(self.sigma0) - np.linalg.inv(self.sigma1))
    d = a[0][0]
    k = a[1][0]
    f = a[0][1]
    g = a[1][1]
    h = b[0]
    b = b[1]
    x01 = np.linspace(1.0, 9.0, num = 80)
    sq = np.sqrt(x01**2 * k**2 + 2*x01**2*f*k + 2*x01*b*k - 4*g*c - 4*x01*g*h - 4*x01**2*d*g + x01**2 * f**2 + 2*x01*b*f + b**2)
    y01 = (-1 * (0.5 * sq) / g) - ((0.5*x01*k)/g) - ((0.5*x01*f)/g) - ((0.5*b)/g)

    c = ( np.log(self.phiClass2/self.phiClass1) ) - ( 0.5 * np.log( np.linalg.det(self.sigma2)/ np.linalg.det(self.sigma1) ) ) + ( self.meanClass2.T @ (np.linalg.inv(self.sigma2) @ self.meanClass2) ) - ( self.meanClass1.T @ (np.linalg.inv(self.sigma1) @ self.meanClass1) )
    b = -2 * ( (np.linalg.inv(self.sigma2) @ self.meanClass2) - (np.linalg.inv(self.sigma1) @ self.meanClass1) )
    a = -0.5 * (np.linalg.inv(self.sigma2) - np.linalg.inv(self.sigma1))
    d = a[0][0]
    k = a[1][0]
    f = a[0][1]
    g = a[1][1]
    h = b[0]
    b = b[1]
    x21 = np.linspace(1.0, 9.0, num = 80)
    sq = np.sqrt(x21**2 * k**2 + 2*x21**2*f*k + 2*x21*b*k - 4*g*c - 4*x21*g*h - 4*x21**2*d*g + x21**2 * f**2 + 2*x21*b*f + b**2)
    y21 = (-1 * (0.5 * sq) / g) - ((0.5*x21*k)/g) - ((0.5*x21*f)/g) - ((0.5*b)/g)

    c = ( np.log(self.phiClass2/self.phiClass0) ) - ( 0.5 * np.log( np.linalg.det(self.sigma2)/ np.linalg.det(self.sigma0) ) ) + ( self.meanClass2.T @ (np.linalg.inv(self.sigma2) @ self.meanClass2) ) - ( self.meanClass0.T @ (np.linalg.inv(self.sigma0) @ self.meanClass0) )
    b = -2 * ( (np.linalg.inv(self.sigma2) @ self.meanClass2) - (np.linalg.inv(self.sigma0) @ self.meanClass0) )
    a = -0.5 * (np.linalg.inv(self.sigma2) - np.linalg.inv(self.sigma0))
    d = a[0][0]
    k = a[1][0]
    f = a[0][1]
    g = a[1][1]
    h = b[0]
    b = b[1]
    x20 = np.linspace(1.0, 9.0, num = 80)
    sq = np.sqrt(x20**2 * k**2 + 2*x20**2*f*k + 2*x20*b*k - 4*g*c - 4*x20*g*h - 4*x20**2*d*g + x20**2 * f**2 + 2*x20*b*f + b**2)
    y20 = (+1 * (0.5 * sq) / g) - ((0.5*x20*k)/g) - ((0.5*x20*f)/g) - ((0.5*b)/g)

    xSame = -1
    for i in range(80):
      if (round(y01[i]) == round(y21[i]) and round(y21[i]) == round(y20[i])):
        # print(x01[i], y01[i])
        xSame = i
        break

    if xSame == -1:
      plt.plot(x01, y01, color = "#000000")
      plt.plot(x21, y21, color = "#000000")
      plt.plot(x20, y20, color = "#000000")
      # plt.show()
      print("please try again for better result")
      
    else:
      plt.plot(x01[:xSame+1], y01[:xSame+1], color = "#000000")
      plt.plot(x21[xSame:], y21[xSame:], color = "#000000")
      plt.plot(x20[xSame:65], y20[xSame:65], color = "#000000")
      plt.show()

print("dataSet1 results")
mean0 = [3, 6]
cov0 = [[1.5, 0], [0, 1.5]]  
mean1 = [5, 4]
cov1 = [[2, 0], [0, 2]]
mean2 = [6, 6]
cov2 = [[1, 0], [0, 1]]
b = Bayesian(mean0, cov0, mean1, cov1, mean2, cov2, numData=500, numTrain=400)
print("accuracy train")
b.classifier(n=2, dataX=b.features, dataY=b.label)
print("accuracy test")
b.classifier(n=2, dataX=b.testFeature, dataY=b.testLabel)
b.plot1()


print("dataSet2 results")
mean0 = [3, 6]
cov0 = [[1.5, 0.1], [0.1, 0.5]]  
mean1 = [5, 4]
cov1 = [[1, -0.2], [-0.2, 2]]
mean2 = [6, 6]
cov2 = [[2, -0.25], [-0.25, 1.5]]  
b = Bayesian(mean0, cov0, mean1, cov1, mean2, cov2, numData=500, numTrain=400)
print("accuracy train")
b.classifier(n=2, dataX=b.features, dataY=b.label)
print("accuracy test")
b.classifier(n=2, dataX=b.testFeature, dataY=b.testLabel)
b.plot2()