# Data prepocess, Bert as service and Bert
****
In this part, I  will show you the methods of prepocessing the tweets data, encodding the text content to a vector with specific length(768) and how to use bert model to deal with NLP classification problem.



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
At the beginning, I grouped the train data by text and found there is a dozen of sentence have a different target, but the content of text are the same. Therefore, I CREATE a new column: target_relabeled and  revsie the mislabel manually.

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
Finally, in the training data, there is a column called keywords, it contains the disaster words of the text or the disaster this text discribed(didn't contain in the text).  
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

![code-past]

--------------
[code-past]:/Code/prepocess and bert jinyi shang/img/bert as service1.PNG
