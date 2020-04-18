# Encodding text from jinyi
import pandas as pd
import numpy as np

# read data
df = pd.read_csv('D:/GWU/first year/6103/project/RealorNottrain.csv')
df_test = pd.read_csv('D:/GWU/first year/6103/project/RealorNottest.csv')
print(df.head())
print(df_test.head())

# build data list for running bert as service
datatrain=df['newtext'].copy()
datatest=df_test['newtext'].copy()
descriptiontrain=[x for x in datatrain]
descriptionttest=[x for x in datatest]

# import bert as service. if you want to import this, you need to build bert envirnoment:
# First, you need to install the server and client via pip:  pip install bert-serving-server   # pip install bert-serving-client
# Next, you need to download the Pre-trained BERT Model
# Then, your  Python >= 3.5 and  Tensorflow >= 1.10
# Finally, start your bert as service using: bert-serving-start -model_dir /tmp/english_L-12_H-768_A-12/ -num_worker=1
# when you see your prompt show : ready and request
# Execute the following code
# more information and tutorial is here: https://github.com/hanxiao/bert-as-service
from bert_serving.client import BertClient
bc = BertClient()
t1=bc.encode(descriptiontrain)
t2=bc.encode(descriptionttest)
print(t1.shape)
print(t2.shape)

# combine encodded text and target
x=np.array(df['target_relabeled']).reshape(-1,1)
newtrain=np.hstack((t1,x))

# output csv
df2=pd.DataFrame(newtrain)
df3=pd.DataFrame(t2)
df2.to_csv('realornot_trainencode.csv')
df3.to_csv('realornot_testencode.csv')