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

At the beginning, I grouped the train data by text and found there is a dozen of sentence have a different target, but the content of text are the same. Therefore, I CREATE a new column: target_relabeled and  revsie the mislabel manually.

Contraction map: like 
|Original|Converted|
|'ain't', 'don't', 'hadn't'|'is not','do not','had not'|
