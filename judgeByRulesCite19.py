import pandas as pd
alldata = pd.read_csv('all_Data_newest_more_features_pisa.csv',encoding='gbk')
alldata['Lumbar Number'] = alldata['编号'].apply(lambda x: float(int(x[-1])+1)) 

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 加载CSV文件
df = alldata
psdfile=pd.read_csv('all_Data_newest_more_features_pisa_new.csv',encoding='utf8')
psdDataAnter = psdfile['图片序号2PSD']
psdDataPoster = psdfile['图片序号4PSD']

# 参数p0
p0 = 0.1  # 假设p0为0.5，你可以根据需要调整这个值
psdDataAnter= list(psdDataAnter.values)
psdDataPoster = list(psdDataPoster.values)
df['psdData'] = [min(x, y) for x, y in zip(psdDataAnter, psdDataPoster)]
# 首先计算规则1的结果  
rule1_result = (df['Neutral Position_ratio'] > p0).astype(int)  
# 创建一个新的布尔序列来检查 psdData 是否大于 37  
psd_greater = df['psdData'] > 37
# 然后，我们更新 rule1_result，对于不满足规则1的行（即 rule1_result 为 0 的行），  
# 如果 psdData 大于 37，则将 rule1_result 设置为 1  
rule1_result.loc[~rule1_result & psd_greater] = 1

# 规则2: 判断Lordotic Position_ratio和Kyphotic Position_ratio的差的绝对值是否大于0.14
rule2_result = (abs(df['Lordotic Position_ratio'] - df['Kyphotic Position_ratio']) > 0.14).astype(int)

#规则3：判断psd是否大于37
rule3_result = (df['psdData'] > 37).astype(int)  
# 计算评估指标
def getResult(combined_result):
    acc = accuracy_score(df['label'], combined_result)
    precision = precision_score(df['label'], combined_result)
    recall = recall_score(df['label'], combined_result)
    f1 = f1_score(df['label'], combined_result)
    return acc,precision,recall,f1
# 打印结果
print(getResult(rule2_result))
