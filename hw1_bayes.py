import numpy
import re
import plotly.graph_objs as go
ngram=5

common=['I','me','my','you','your','it','they','we',
        'is','was','are','were','be','being',
        'like','can','do','have','has','make','get','go','want',
        'to','the','a','in','of','on','for','with','so','just','and','or','at','as','some','any']
common2=['','I','me','my','you','your','it','i','to',
        'is','be','being','do','have','the','a','in','of','at']
common3=['','I','the']

def tokenize(s):
    res=s.split(' ')
    return res
    # res=[]
    # step1=s.split(' ')
    # for i in range(len(step1)-2):
    #     res.append((step1[i],step1[i+1],step1[i+2]))
    # res.append((step1[len(step1)-2],step1[len(step1)-1]))
    # return res

def ngram_tokenize(s):
    res = []
    step1 = s.split(' ')
    for i in range(len(step1) - ngram+1):
        res.append(tuple([step1[i+j] for j in range(ngram)]))
    res.append(tuple([step1[len(step1)-j] for j in reversed(range(1,ngram if ngram<len(step1) else len(step1)))]))
    return res
def better_tokenize(s):
    # res=s.split(' ')

    # s=s.lower()
    # lis=s.split(' ')
    # res=lis

#remove common word
    lis=s.split(' ');res=[]
    for i in lis:
        if i not in common3:
            res.append(i)
            try:
                re.search(r'[\!\*\?]', i)
                res.append(i)
            except:
                continue

    # lis=s.split(' ')
    # res=[]
    # for i in lis:
    #     res.append(i)
    #     try:
    #         re.search(r'fuck',i)
    #         res.append('fuck')
    #     except:continue

    # lis=s.split(' ')
    # res=[]
    # for i in range(len(lis)):
    #     try:
    #         a=re.search(r'\w+',lis[i]).group().lower()
    #         #print(a)
    #         res.append(a)
    #     except:continue
    #print(res)
    return res

def train(lis_x,lis_y,alpha=0):
    res=[{},{}]      #word occurence for p/n
    wordC=[0,0]        #word count for p/n
    p=0                 #p case num
    for line in range(len(lis_x)):
        np=int(lis_y[line])
        p+=np
        # for i in lis_x[line]:
        #     if i not in res[0]:res[0][i]=0
        #     if i not in res[1]:res[1][i]=0
        #     res[np][i]+=1
        #     wordC[np]+=1
        for i in lis_x[line]:
            if i not in res[0]:res[0][i]=0
            if i not in res[1]:res[1][i]=0
            res[np][i]+=1
            wordC[np]+=1

    # res0_file = open("res0.txt", 'w')
    # for i in res[0]: res0_file.write('{},{}\n'.format(i,res[0][i]))
    # res0_file.close()
    # res1_file = open("res1.txt", 'w')
    # for i in res[1]: res1_file.write('{},{}\n'.format(i, res[1][i]))
    # res1_file.close()

    px=res[0].copy();pxy0=res[0].copy();pxy1=res[0].copy()
    py1=p/len(lis_y);py0=1-py1


    for i in res[0]:
        #print((res[0][i]+alpha)/(wordC[0]+alpha*(wordC[0]+wordC[1])),(res[1][i]+alpha)/(wordC[1]+alpha*(wordC[0]+wordC[1])))
        px[i]=(res[0][i]+res[1][i])/(wordC[0]+wordC[1])
        pxy0[i]=(res[0][i]+alpha)/(wordC[0]+alpha*(wordC[0]+wordC[1]))
        pxy1[i]=(res[1][i]+alpha)/(wordC[1]+alpha*(wordC[0]+wordC[1]))

    return [px,py0,py1,pxy0,pxy1]



def classify(x,py0,py1,pxy0,pxy1):
    y=[]
    for line in x:

        a = py0
        b=py1
        for i in line:
            if (i not in pxy0):
                continue


            a *= pxy0[i]
            b*=pxy1[i]
        if (a > b)or((a==0)and(b==0)):
            y.append(0)
        else:
            y.append(1)
    return y

def alpha_test(x_use,y_use,st,ed,intv):
    step=(ed-st)/intv
    plotx=[];ploty=[]
    print("alpha","accuracy")
    for i in numpy.arange(st,ed,step):
        px, py0, py1, pxy0, pxy1 = train(lis_x, lis_y, i)
        y = classify(x_use, py0, py1, pxy0, pxy1)
        acc = 0
        for j in range(len(y_use)):

            if y[j] == int(y_use[j]):
                acc += 1

        acc = acc / len(y_use)
        print(i,acc)
        plotx.append(i);ploty.append(acc)


    fig=go.Figure()
    fig.add_trace(go.Scatter(x=plotx,y=ploty))

    fig.update_layout(
        title="Accuracy vs. Smoothing_alpha",
        xaxis_title="smoothing_alpha",
        yaxis_title="accuracy"
        )
    fig.show()
    return

def common_word_remove(dic1,dic2,para):
    while (True) and (len(dic1) > para) and (len(dic2) > para):
        akey = sorted(dic1.items(), key=lambda item: item[1], reverse=True)[:para]
        bkey = sorted(dic2.items(), key=lambda item: item[1], reverse=True)[:para]
        same_tf = False
        for i in range(para):
            for j in range(para):
                if akey[i][0] == bkey[j][0]:
                    dic1.pop(akey[i][0]);
                    dic2.pop(bkey[j][0])
                    same_tf = True
        if (not same_tf): break
    return dic1,dic2

#读取train.txt，训练
x_file=open('X_train.txt','r')
y_file=open('y_train.txt','r')
lis_x=[];lis_y=[]
for i in x_file:lis_x.append(ngram_tokenize(i))
for i in y_file:lis_y.append(i)
px,py0,py1,pxy0,pxy1=train(lis_x,lis_y,0.4258)
x_file.close();y_file.close()

#好坏2dice
#将排序pxy0捞出来，看常用词
# pxy0,pxy1=common_word_remove(pxy0,pxy1,500)

# sortgood=sorted(pxy0.items(), key=lambda kv: (-kv[1], kv[0]))
# print(type(pxy0));print(type(sortgood))
# goodFile=open("pxy0_sort.txt",'w')
# for i in sortgood:goodFile.write(i[0]+','+str(i[1])+'\n')
# goodFile.close()
# sortbad=sorted(pxy1.items(), key=lambda kv: (-kv[1], kv[0]))
# badFile=open("pxy1_sort.txt",'w')
# for i in sortbad:badFile.write(i[0]+','+str(i[1])+'\n')
# badFile.close()

# y0_file=open("pxy0.txt",'w')
# for i in pxy0:y0_file.write(i+','+str(pxy0[i])+'\n')
# y0_file.close()
#
# y1_file=open("pxy1.txt",'w')
# for i in pxy1:y1_file.write(i+','+str(pxy1[i])+'\n')
# y1_file.close()

##读取dev.txt
x_test_file=open('X_dev.txt','r')
y_test_file=open('y_dev.txt','r')
test_x=[];test_y=[]
for i in x_test_file:test_x.append(ngram_tokenize(i))
for i in y_test_file:test_y.append(i)
x_test_file.close();y_test_file.close()

#读取test
x_realTestFile=open('X_test.txt','r')
kaggle_x=[]
for i in x_realTestFile:kaggle_x.append(ngram_tokenize(i))
x_realTestFile.close()




op="output"
if(op=="test"):

#进行classify
    x_use=test_x;y_use=test_y;
    y=classify(x_use,py0,py1,pxy0,pxy1)
    acc=0
    TP=0;FN=0;FP=0;TN=0
    print(len(y_use))
    for i in range(len(y_use)):
        if (y[i]==1)and(int(y_use[i])==1):TP+=1
        elif (y[i]==1)and(int(y_use[i])==0):FP+=1
        elif (y[i] == 0) and (int(y_use[i]) == 1):FN += 1
        elif (y[i] == 0) and (int(y_use[i]) == 0):TN += 1
        if y[i]==int(y_use[i]):
            acc+=1
    alpha_test(x_use, y_use, 0, 1.3, 30)
    acc=acc/len(y_use)
    print("accuracy",acc)
    print('TP: {}'.format(TP))
    print('FP: {}'.format(FP))
    print('FN: {}'.format(FN))
    print('TN: {}'.format(TN))
    F1=2*TP/(2*TP+FN+FP);print('F1: {}'.format(F1))
elif(op=='output'):
    y_realTestFile=open('y_test_Bayes.csv','w')
    x_use = kaggle_x;
    y = classify(x_use, py0, py1, pxy0, pxy1)
    acc = 0
    y_realTestFile.writelines('Id,Category\n')
    for i in range(len(kaggle_x)):
        y_realTestFile.writelines('{},{}\n'.format(i,int(y[i])))
    y_realTestFile.close()
