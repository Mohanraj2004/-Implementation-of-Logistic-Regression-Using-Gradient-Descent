# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the packages required.
2. Read the dataset.
3. Define X and Y array.
4. Define a function for costFunction,cost and gradient.
5. Define a function to plot the decision boundary and predict the Regression value.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: POOJA A
RegisterNumber:  212222240072

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

data=np.loadtxt("ex2data1.txt",delimiter=',')
X=data[:,[0,1]]
y=data[:,2]

X[:5]

y[:5]

plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

def sigmoid(z):
    return 1/(1+np.exp(-z))

plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot,sigmoid(X_plot))
plt.show()

def costFunction (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    grad=np.dot(X.T,h-y)/X.shape[0]
    return J,grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J,grad=costFunction(theta,X_train,y)
print(J)
print(grad)

def cost (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    J=-(np.dot(y,np.log(h))+np.dot(1-y,np.log(1-h)))/X.shape[0]
    return J

def gradient (theta,X,y):
    h=sigmoid(np.dot(X,theta))
    grad=np.dot(X.T,h-y)/X.shape[0]
    return grad

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
res=optimize.minimize(fun=cost,x0=theta,args=(X_train,y),method='Newton-CG',jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
    x_min,x_max=X[:,0].min()-1,X[:,0].max()+1
    y_min,y_max=X[:,1].min()-1,X[:,1].max()+1
    xx,yy=np.meshgrid(np.arange(x_min,x_max,0.1),np.arange(y_min,y_max,0.1))
    X_plot=np.c_[xx.ravel(),yy.ravel()]
    X_plot=np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
    y_plot=np.dot(X_plot,theta).reshape(xx.shape)
    
    plt.figure()
    plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
    plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not Admitted")
    plt.contour(xx,yy,y_plot,levels=[0])
    plt.xlabel("Exam 1 score")
    plt.ylabel("Exam 2 score")
    plt.legend()
    plt.show()

plotDecisionBoundary(res.x,X,y)

prob=sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
    X_train =np.hstack((np.ones((X.shape[0],1)),X))
    prob=sigmoid(np.dot(X_train,theta))
    return (prob>=0.5).astype(int)
np.mean(predict(res.x,X)==y)
*/
```

## Output:
### 1.Array Value of x
![e5](https://user-images.githubusercontent.com/119390329/236677709-b8811b0d-77cd-4c78-9484-f6fa00c85757.png)

### 2.Array Value of y
![x5](https://user-images.githubusercontent.com/119390329/236677746-4266f3fe-4c57-4849-9103-d42cd6836046.png)

### 3.Exam 1 - score graph
![p5](https://user-images.githubusercontent.com/119390329/236677760-16c92efc-daaf-4d66-9fc6-02bb6cd01835.png)

### 4.Sigmoid function graph
![ex5](https://user-images.githubusercontent.com/119390329/236677781-9caee7a0-d353-4b3e-af81-a3a641ece196.png)

### 5.X_train_grad value
![5](https://user-images.githubusercontent.com/119390329/236677840-80fcaac5-a2a9-4b62-8e0f-fe36648d2ec0.png)

### 6.Y_train_grad value
![ee5](https://user-images.githubusercontent.com/119390329/236677854-f7505b79-c27c-4dfc-9173-7fe437dd036f.png)

### 7.Print res.x
![xx5](https://user-images.githubusercontent.com/119390329/236677864-98ec539a-f768-4272-aac3-5b65471394a8.png)

### 8.Decision boundary - graph for exam score
![xxx5](https://user-images.githubusercontent.com/119390329/236677788-bf0ebcb5-c00b-4890-847e-6ba5ab241d61.png)

### 9.Proability value 
![p5p](https://user-images.githubusercontent.com/119390329/236677944-e2267df6-6700-48de-abc9-621c92548129.png)

### 10.Prediction value of mean
![ppp5](https://user-images.githubusercontent.com/119390329/236677976-e407b486-ea71-43be-a1b7-8bc289855c60.png)

## Result:
Thus the program to implement the Logistic Regression Using Gradient Descent is written and verified using python programming.
