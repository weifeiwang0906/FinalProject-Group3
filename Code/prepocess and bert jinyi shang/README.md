# Data prepocess, Bert as service and Bert
****
This part will show the methods of prepocessing the tweets data, encodding the text content to a vector with specific length(768) using bert as service and how to use bert model to solve NLP classification problem.



****
## Content
-----------
* [prepocessNLP](#prepocessNLP)
  * slang.txt(abb part)
* [bert_as_service](#bert_as_service)
* [bertdata](#bertdata)
* [runbert](#runbert)
* [bertresults](#bertresults)

### prepocessNLP
-----------
This is the first part. We need to download the data from kaggle competition:
[NLP Realornot](https://www.kaggle.com/c/nlp-getting-started )  

Data Prepocessing consists of Mislabel, CONTRACTION_MAP, @/#/http, Repetitive Letter, Abbreviation, Lowercase, punctuation, Stopwords and Keywords Variable.

**Mislabel:**  
At the beginning, I grouped the train data by text and found there is a dozen of sentence have a different target, but the content of text are the same. Therefore, I CREATE a new column: target_relabeled and  revise the mislabel manually.

**Contraction map:**   
|Original|Converted|
|--------- | --------|
|ain't, don't, hadn't|is not, do not, had not|   

Here, I use a contraction dictionary from the Internet to achieve this. You can see the dictionary in the code.  

**@/#/http:**  
In the train data, there are lots of website information, @ and # ,this part used for cleaning them.

**Repetitive Letter**:
|Original|Converted|
|--------- | --------|
|loooooook, goooood|look,good| 

Here, I write some loops to convert some words 'loooook' to 'look'

**Abbreviation**:  
This part is different from contraction part. In English, there are some expression like 'lmao' means happy, 'icq' means 'I seek you'  
|Original|Converted|
|--------- | --------|
|lol,thx|happy,thanks|

Here, I download a dictionary and a slang text. I combine these two part to clean my data.

**Lowercase**:  
|Original|Converted|
|--------- | --------|
|HE, THIS|he, this|

Uniform case.

**punctuation**:  
|Delete|
|--------- | 
|!#@$%:;<=>,? |

**Stopwords**:  
In our text analysis, there are many words which didn't have actual meaning, like: 'to','at' etc.  
Here, I use the package nltk.corpus, which contains many stopwords to clean the text.  
```
from nltk.corpus import stopwords
```

**Keywords Variable**:  
Finally, in the training data, there is a column called keywords, it contains the disaster words of the text or the disaster this text described(didn't contain in the text).  
Since these words are important, I add them to the text.


### bert_as_service:
-----------
This is the second part.

After prepocessing the text, we will encode the text content to a vector with specific length to run our traditional machine leaning model.

Our methods is bert as service. Now, what is bert as service, simply speaking, bert as service is the encoder of bert. As we know, bert is a powerful deep learning model, which can solve nearly all of NLP problem. Therefore, we use bert as service to achieve our text to vector process.

Here is a link of bert as service tutorial 
[bert as service](https://github.com/hanxiao/bert-as-service )

Actually, there are only 4 lines of code to encode our text  
```
from bert_serving.client import BertClient
text=[x for x in datatrain]
bc = BertClient()
t1=bc.encode(text)
```
if you want to import this, you need to build bert envirnoment:
First, you need to install the server and client via pip:  
```
pip install bert-serving-server  # server
pip install bert-serving-client  # client, independent of `bert-serving-server`
```
Next, you need to download the Pre-trained BERT Model [you can download from here](https://github.com/google-research/bert#pre-trained-models).
Then, your  Python >= 3.5 and  Tensorflow >= 1.10  
Finally, start your bert as service using: 
```
bert-serving-start -model_dir /tmp/english_L-12_H-768_A-12/ -num_worker=1
```
when you see your prompt show : ready and request, execute the 4 lines of codes  
Then, you can obtain a matrix with  the number of texts by 768.

### bertdata
-----------
This is the third part.  

Bert as service is used for executing our traditional machine learning model. like decision tree, support vector machine, KNN etc.  
This part aims to build the data suitable for bert.  
Here is the bert tutorial: [bert](https://github.com/google-research/bert).

How to use bert to solve a binary classification problem?  
if you download the bert model, in the **run_classifier.py**, you can see many different dataProcessors, these are some examples google provide to solve different NLP problem. One of processor named  **ColaProcessor** is used for solving the binary classification problem.

Therefore, we can biuld our data structure like Cola data. Then, we can run our model directly.

What's more, since our test data is from kaggle and we don't have have the true value to evaluate our prediction. we run bert twice. one is train, evaluation and test data, this is used for obtain the predict values to submit on kaggle. The other is only train and evaluation data, this is used for plot roc curve, evaluation purpose.  

This is our split:  
```
df_bert_train, df_bert_dev = train_test_split(df_bert, test_size=0.2,random_state=100)
```
First run:  

|intput|output|
|--------- | --------|
|train, evalution, test|test prediction|

Second run:
|intput|output|
|--------- | --------|
|train, evalution|evaluation prediction|

### runbert
-----------
This is the fourth part. All of code here is running on the Colaboratory.

Since google provide a free Gpu and it's faster to run deep learning model
Here is the link: [Colaboratory](https://colab.research.google.com/notebooks/gpu.ipynb)  
Notice, if you are the first user of colab, remember to authorize.

As for bert tutorial:
first, you need to download the the pretrain bert model: [Here](https://github.com/google-research/bert#pre-trained-models).  
Then, upload your file to the google drive, and open a new ipynb.

first, set your envirnoment and connect your google drive with your ipynb:
```
from google.colab import drive
drive.mount('/content/drive/')
```
```
cd /content/drive//My\ Drive/bert-master
```
Then,  you can run bert model directly using:
```
!python run_classifier.py \
  --task_name=cola \
  --do_eval=true \
  --do_train=true \
  --do_predict=true \
  --data_dir=6103project \
  --vocab_file=cased_L-12_H-768_A-12/vocab.txt \
  --bert_config_file=cased_L-12_H-768_A-12/bert_config.json \
  --init_checkpoint=cased_L-12_H-768_A-12/bert_model.ckpt \
  --max_seq_length=40 \
  --train_batch_size=8 \
  --learning_rate=2e-5 \
  --num_train_epochs=3.0 \
  --output_dir=output \
  --do_lower_case=False
```
As we can see, our dataprocessor is cola. if your upload dataset have dev, you can do evaluation. Set your vocab_file, bert_config_file and init_checkpoint same as what you download from bert website.

Here's the result on the evaluation set:  
![image](https://github.com/weifeiwang0906/FinalProject-Group3/blob/master/Code/prepocess%20and%20bert%20jinyi%20shang/img/eval.PNG)

### bertresults
-----------
After finishing running, you can obtain a txt file named test result which contains the probability of predicted label

And the txt form like this:
|   |   |
|--------- | --------|
|0.1|0.9|
|0.2|0.8|
|0.3|0.7|
|0.1|0.9|
|...|...|

Load this text and use this part of code covert theprobability to label.

Also, ploting ROC curve is at this part
```
false_positive_rate,true_positive_rate,thresholds=roc_curve(labels, dfdev.iloc[:,1])
```
Use this obtain TPR, FPR and the sorted probability.

Here is the result:  
![image](https://github.com/weifeiwang0906/FinalProject-Group3/blob/master/Code/prepocess%20and%20bert%20jinyi%20shang/img/ROC.PNG)
