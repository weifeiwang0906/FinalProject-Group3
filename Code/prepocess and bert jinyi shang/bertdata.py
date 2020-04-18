# build bert structure data from jinyi
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from pandas import DataFrame

# Creating train dataframe according to BERT
df=pd.read_csv('D:/GWU/first year/6103/project/RealorNottrain.csv',sep=',')
pd.set_option('display.max_columns', None)
print(df.head())
df_bert = pd.DataFrame({'user_id': df['id'],
                        'label': df['target_relabeled'],
                        'id': df['id'],
                        'text': df['newtext'].replace(r'\n', ' ', regex=True)})  # Here,we build the suitable data for bert input(just the second and four columns. It's ok for  any of the first and third input.)



# create evaluation set.
df_bert_train, df_bert_dev = train_test_split(df_bert, test_size=0.2,random_state=100)

# Creating test dataframe according to BERT
df_test = pd.read_csv("D:/GWU/first year/6103/project/RealorNottest.csv")
df_bert_test = pd.DataFrame({'User_ID': df_test['id'],
                             'text': df_test['newtext'].replace(r'\n', ' ', regex=True)})

# Saving dataframes to .tsv format as required by BERT
#df_bert_train.to_csv('D:/GWU/first year/6103/project/train.tsv', sep='\t', index=False, header=False)
#df_bert_dev.to_csv('D:/GWU/first year/6103/project/dev.tsv', sep='\t', index=False, header=False)
#df_bert_test.to_csv('D:/GWU/first year/6103/project/test.tsv', sep='\t', index=False, header=True)

# Since we dont have test target, if we want to plot ROC curve,we should know target. So we use our dev to plot.
# now use dev as test to build model again to get predict probability
del(df_bert_dev['label'])
del(df_bert_dev['id'])
df_bert_dev.to_csv('D:/GWU/first year/6103/project/test2.tsv', sep='\t', index=False, header=True)


