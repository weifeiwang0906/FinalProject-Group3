# Data prepocess and Bert
****
In this part, I  will show you the methods of prepocessing the tweets data, encodding the text content to a vector with specific length(768) and how to use bert model to deal with NLP classification problem.



****
## Content
-----------
* [prepocessNLP](#prepocessNLP)
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

Here, I download a dictionary and a slang txt. I combine these two part to clean my data.




