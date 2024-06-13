import pandas as pd
from sklearn.metrics import confusion_matrix as cm
alldata = pd.read_csv('all_Data_newest_more_features_multilables.csv',encoding='utf')
alldata['new编号'] = alldata['编号'].apply(lambda x: int(x[-1])-1) 
from collections import Counter
import pickle
with open("datas/labels.bin_version2", "rb") as f:
    lables = pickle.load(f) 
lable_dict={}
def jgd(a):
    if a==0:return 0
    else: return 1
for k in lables.keys():
    if lables[k] != []:
        tmp = lables[k]
        for i in range(len(tmp)):
            if tmp[i]!=-1:
                lable_dict[str(k) + "_" + str(i+1)] = str(int(jgd(tmp[i]) ))

df = alldata[alldata['编号'].isin(lable_dict.keys())]

code  = list(df['编号'])
pos1 = dict(zip(alldata['图片序号2比例'], [lable_dict[i] for i in code]))
pos2 = dict(zip(alldata['图片序号3比例'], [lable_dict[i] for i in code]))
pos3 = dict(zip(alldata['图片序号4比例'], [lable_dict[i] for i in code]))
def Guize(i):
    if i<=0 :return 0
    else: return 1
def judgeByGuize(pos):

    d0 = list(pos.keys())
    l1 = list(pos.values())
    l0 = list(map(lambda x: str(Guize(x)),d0))
    return cm(l1,l0)
def fromPosToFig(pos2,words,i):
    C1 = judgeByGuize( pos2)
    import seaborn as sns #导入包
    from matplotlib import pyplot as plt 
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib import cm
    #mpl.rcParams.update({'figure.dpi':150})
    Q=['Greens','jet','rainbow']
    colors=plt.get_cmap(Q[0])

    xtick=['0','1']
    ytick=xtick

    sns.heatmap(C1,fmt='g',cmap=colors,annot=True,cbar=False,xticklabels=xtick, yticklabels=ytick) #画热力图,annot=True 代表 在图上显示 对应的值， fmt 属性 代表输出值的格式，cbar=False, 不显示 热力棒
    plt.title(words)
    plt.xlabel("Predict")
    plt.ylabel("True")
    plt.show()
    #plt.savefig(fname="./"+''.join(words.split(" ")))
    plt.close()
#fromPosToFig(pos1,"Confusion Matrix of Lordotic Position",0)
#fromPosToFig(pos2,"Confusion Matrix of Neutral Position",1)
fromPosToFig(pos3,"Confusion Matrix of Kyphotic Position",2)