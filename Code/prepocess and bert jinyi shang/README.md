# Data prepocess and Bert
****
In this part, I  will show you the methods of prepocessing the tweets data, encodding the text content to a vector with specific length(768) and how to use bert model to deal with NLP classification problem.

****
## Content
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

