import numpy as np
import random
from collections import Counter
from scipy.sparse import *
from scipy.special import expit
# import matplotlib.pyplot as plt
import plotly.graph_objs as go

def sigmoid(x):
    return expit(x)



def compute_gradient(x,y,y6):   #输入完整的x, y
    # i=random.randint(0,y.shape[0]-1)

    # gra=sparse.lil_matrix(np.transpose(x[i,:])*(np.transpose(y)[i]-y6[i]))          #只抽取i列算gradient
    gra = np.transpose(x) * (np.transpose(y) - y6)              #整个x,y算gradient
    return gra

def tokenize(s):
    res=s.split(' ')
    return res

def log_likelihood(x,y,b):
    bx=np.transpose(b)*np.transpose(x)

    res=np.sum(y*bx-np.log(1+np.exp(bx)))
    return res

def logistic_regression(train_x,train_y,num_step,learning_rate):      #这就是train函数
    b=np.zeros(train_x.shape[1])

    #print(b.shape)
    log_list=[]
    for i in range(num_step):
        print("train step {} complete".format(i))
        y6=sigmoid(train_x*b)
        gra=compute_gradient(train_x,train_y,y6)
        print(' gra complete {}'.format(gra.shape))

        b=b+learning_rate*gra
        print('new b')
        log_list.append(log_likelihood(train_x,train_y,b))


    return b,log_list

def predict(b,usex):
    ye = sigmoid(usex * b)

    for i in range(usex.shape[0]):
        judge = 0
        if ye[i] > 0.5: judge = 1
        ye[i]=judge

    return ye


def input_process_try(filename,column={}):
    llx=[];test_tf=True
    if column!={}:test_tf=False
    filex=open(filename,'r')
    wordcnt=0
    for i in filex:
        line=tokenize(i)
        cnt=Counter(line);llx.append(cnt)
        for j in line:    #j是word
            if(j not in column)and(test_tf):column[j]=wordcnt;wordcnt+=1
    filex.close()
    #llx搞定,column包含nx的所有列
    #print(column)

    #print(len(llx),len(column))
    nx=dok_matrix((len(llx),len(column)+1))
    nx[:,-1]=1
    lineIndex=-1
    for i in llx:    #i是dict

        lineIndex += 1
        for j in i:   #j是word
            if j in column:
                nx[lineIndex,column[j]]=i[j]
    #nx搞定
    #print(nx.shape)
    return nx.tocsr(),column


lis_y=[]
filey=open('y_train.txt','r')
for i in filey:
    lis_y.append(int(i))
ny=np.array(lis_y)
#print(type(ny[0,11]))
#ny搞定

x_train,column=input_process_try('X_train.txt')
op="out"
b,log1=logistic_regression(x_train,ny,100000,5e-5)



if op=="test":
    xaxis=np.array(range(len(log1)))
    # file = open("res0.txt", 'w')
    # for i in range(len(log1)):  file.write('{}\n'.format(log1[i]))
    # file.close()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xaxis,y=log1))

    fig.update_layout(
        title="log likelihood vs. step",
        xaxis_title="step",
        yaxis_title="log likelihood"
    )
    fig.show()

    _,log2=logistic_regression(x_train,ny,1000,5e-4)
    _,log3=logistic_regression(x_train,ny,1000,5e-8)


    fig = go.Figure()
    fig.add_trace(go.Scatter(y=log1,name=5e-5))
    fig.add_trace(go.Scatter(y=log2,name=5e-4))
    fig.add_trace(go.Scatter(y=log3,name=5e-8))
    fig.update_layout(
        title="log likelihood vs. step",
        xaxis_title="step",
        yaxis_title="log likelihood"
    )
    fig.show()


    usex=x_train;usey=ny
    ye=predict(b,usex)
    acc = 0;TP = 0;FN = 0;FP = 0;TN = 0
    for i in range(usey.shape[0]):
        if (ye[i]==1)and(int(usey[i])==1):TP+=1
        elif (ye[i]==1)and(int(usey[i])==0):FP+=1
        elif (ye[i] == 0) and (int(usey[i]) == 1):FN += 1
        elif (ye[i] == 0) and (int(usey[i]) == 0):TN += 1
        if ye[i] == int(usey[i]): acc += 1
    acc /= usey.shape[0]
    print(acc)
    print('TP: {}'.format(TP))
    print('FP: {}'.format(FP))
    print('FN: {}'.format(FN))
    print('TN: {}'.format(TN))
    F1 = 2 * TP / (2 * TP + FN + FP);
    print('F1: {}'.format(F1))
elif op=="out":
    usex,no=input_process_try('X_test.txt',column)
    # print('train {}'.format(x_train.shape))
    # print('test {}'.format(usex.shape))
    ye=predict(b,usex)
    y_File = open('y_test_log.csv', 'w')
    y_File.writelines('Id,Category\n')
    #print(ye.shape)
    for i in range(usex.shape[0]):
        y_File.writelines('{},{}\n'.format(i, int(ye[i])))
    y_File.close()



# print(b)
# print(b.shape)
# b=[[1,2,1],[111,23,1],[1,4,4]]
# b=np.array(b,dtype=np.float32)
# a=sparse.csc_matrix((23331,10000))
# print(a[1799,23])


