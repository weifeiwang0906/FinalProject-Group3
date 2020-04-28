# build result for submission and plot roc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df4=pd.read_csv('D:/GWU/first year/6103/project/test.tsv',sep='\t')
df=pd.read_csv('D:/GWU/first year/6103/project/test_results.tsv',sep='\t',header=None)
df2=np.zeros(len(df))

for name,value in df.iterrows():
    if value[0] < value[1]:
        df2[name] = 1
    else:
        df2[name] = 0
df3 = pd.DataFrame(df2,columns=['target'])
df3['id']=df4['User_ID']
df3=df3[['id','target']]
df3['id']=df3['id'].apply(lambda x:str(int(x)))
df3['target']=df3['target'].apply(lambda x:str(int(x)))
df3.to_csv('D:/GWU/first year/6103/project/label.csv', sep=',',index=False)

# plot roc curve
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
import matplotlib as mpl
import matplotlib.pyplot as plt

# since our evaluation set  use random state=100, here we want to get the true label
df=pd.read_csv('D:/GWU/first year/6103/project/RealorNottrain.csv',sep=',')
df_bert = pd.DataFrame({'user_id': df['id'],
                        'label': df['target_relabeled'],
                        'id': df['id'],
                        'text': df['newtext'].replace(r'\n', ' ', regex=True)})  # Here,we build the suitable data for bert input(just the second and four columns. It's ok for  any of the first and third input.)


df_bert_train, df_bert_dev = train_test_split(df_bert, test_size=0.2,random_state=100)
labels=df_bert_dev['label']
dfdev=pd.read_csv('D:/GWU/first year/6103/project/test_resultsfordev.tsv',sep='\t',header=None)


#plot
false_positive_rate,true_positive_rate,thresholds=roc_curve(labels, dfdev.iloc[:,1])
plt.title('ROC')
roc_auc=auc(false_positive_rate, true_positive_rate)
plt.plot(false_positive_rate, true_positive_rate,'b',label='AUC = %0.4f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.show()