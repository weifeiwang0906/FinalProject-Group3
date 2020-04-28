# Data prepocessing from Jinyi
import pandas as pd
import numpy as np
import string
import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# read data:
# From the print as below, we can see the training set and test set have 5  and 4 columns,separately: id: a unique identifier for each tweet.
# text: the text of the tweet. location: the location the tweet was sent from (may be blank)
# keyword - a particular keyword from the tweet (may be blank)
# target - in train.csv only, this denotes whether a tweet is about a real disaster (1) or not (0)
df_train = pd.read_csv('D:/realornot/nlp/nlp-getting-started/train.csv',encoding='gbk')
df_test = pd.read_csv('D:/realornot/nlp/nlp-getting-started/test.csv',encoding='gbk')
df_train['newtext']=df_train['text']                #create a new column of text for comparing with the original text
df_test['newtext']=df_test['text']
#pd.set_option('display.max_columns',None)
#print('Train dataframe shape:', df_train.shape)
#print('Test dataframe shape:', df_test.shape)
#print(df_train.head(1))
#print(df_test.head(1))


# Mislabel:
# At the beginning, I grouped trainset by text and found there is a dozen of sentence have a different target, but the content of text are the same.
# Therefore, I CREATE a new column: target_relabeled and  revsie the mislabel manually.
df_mislabeled = df_train.groupby(['text']).nunique().sort_values(by='target', ascending=False)
df_mislabeled = df_mislabeled[df_mislabeled['target'] > 1]['target']
# print(df_mislabeled)              #   by print this.you can see the mislabel sentence
df_train['target_relabeled'] = df_train['target'].copy()
df_train.loc[df_train['text'] == "that horrible sinking feeling when you??ave been at home on your phone for a while and you realise its been on 3G this whole time", 'target_relabeled'] = 0
df_train.loc[df_train['text'] == 'I Pledge Allegiance To The P.O.P.E. And The Burning Buildings of Epic City. ??????','target_relabeled']=0
df_train.loc[df_train['text'] == 'like for the music video I want some real action shit like burning buildings and police chases not some weak ben winston shit','target_relabeled']=0
df_train.loc[df_train['text'] == 'Hellfire is surrounded by desires so be careful and don??at let your desires control you! #Afterlife','target_relabeled']=0
df_train.loc[df_train['text'] == 'CLEARED:incident with injury:I-495  inner loop Exit 31 - MD 97/Georgia Ave Silver Spring','target_relabeled']=1
df_train.loc[df_train['text'] == 'Hellfire! We don??at even want to think about it or mention it so let??as not do anything that leads to it #islam!','target_relabeled']=0
df_train.loc[df_train['text'] == 'RT NotExplained: The only known image of infamous hijacker D.B. Cooper. http://t.co/JlzK2HdeTG','target_relabeled']=1
df_train.loc[df_train['text'] == 'wowo--=== 12000 Nigerian refugees repatriated from Cameroon','target_relabeled']=0
df_train.loc[df_train['text'] == "Mmmmmm I'm burning.... I'm burning buildings I'm building.... Oooooohhhh oooh ooh...",'target_relabeled']=0
df_train.loc[df_train['text'] == '.POTUS #StrategicPatience is a strategy for #Genocide; refugees; IDP Internally displaced people; horror; etc. https://t.co/rqWuoy1fm4','target_relabeled']=1
df_train.loc[df_train['text'] == "The Prophet (peace be upon him) said 'Save yourself from Hellfire even if it is by giving half a date in charity.'",'target_relabeled']=0
df_train.loc[df_train['text'] == 'Caution: breathing may be hazardous to your health.','target_relabeled']=1
df_train.loc[df_train['text'] == "Who is bringing the tornadoes and floods. Who is bringing the climate change. God is after America He is plaguing her\n \n#FARRAKHAN #QUOTE",'target_relabeled']=1
df_train.loc[df_train['text'] == 'To fight bioterrorism sir.','target_relabeled']=0
df_train.loc[df_train['text'] == '#foodscare #offers2go #NestleIndia slips into loss after #Magginoodle #ban unsafe and hazardous for #humanconsumption','target_relabeled']=0
df_train.loc[df_train['text'] == 'In #islam saving a person is equal in reward to saving all humans! Islam is the opposite of terrorism!','target_relabeled']=0
df_train.loc[df_train['text'] == 'He came to a land which was engulfed in tribal war and turned it into a land of peace i.e. Madinah. #ProphetMuhammad #islam','target_relabeled']=0
df_train.loc[df_train['text'] == '#Allah describes piling up #wealth thinking it would last #forever as the description of the people of #Hellfire in Surah Humaza. #Reflect','target_relabeled']=0


# CONTRACTION_MAP:
# Here, I found a contraction dictionary to put the contraction of the sentence back to the original form.
# The following function expand contraction is from internet.
CONTRACTION_MAP = {
"ain't": "is not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"I'd": "I would",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have",
"gonna": "going to",
"w/": "with"
}
def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) \
            if contraction_mapping.get(match) \
            else contraction_mapping.get(match.lower())
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

df_train['newtext'] = df_train['newtext'].apply(lambda x: expand_contractions(x))
df_test['newtext'] = df_test['newtext'].apply(lambda x: expand_contractions(x))


# @/#/http:
# In the text, there are lots of website information, @ and #,this part used for cleaning them.
def Find(string):
    # findall() has been used
    # with valid conditions for urls in string
    url = re.findall('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', string)
    # https://urlregex.com
    return url


def clean_text(text):
    text = re.sub(r"\x89", " ", text)
    text = re.sub(r"@[\w]*", "@", text)
    text = re.sub(r"#", "", text)
    if len(Find(text)) > 0:
        text = re.sub(r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+", " ", text)
    return text

df_train['newtext'] = df_train['newtext'].apply(lambda x: clean_text(x))
df_test['newtext'] = df_test['newtext'].apply(lambda x: clean_text(x))



# Repetitive Letter
# Here are some repetitive letter,like: gooooood! etc, change it as good
for ind in range(0, len(df_train["newtext"])):
    line = df_train.loc[ind, "newtext"]
    for num in range(len(line) - 2):
        if num + 3 > len(line):
            break
        counter = 0
        if line[num] == line[num + 1] and line[num + 1] == line[num + 2]:
            while True:
                if num + 3 + counter > len(line):
                    break
                if line[num + 2 + counter] != line[num]:
                    break
                counter = counter + 1
            line = line.replace(line[num:(num + 2 + counter)], line[num:(num + 3)])
            # print(line)
    df_train.loc[ind, "newtext"] = line

for ind in range(0, len(df_test["newtext"])):
    line = df_test.loc[ind, "newtext"]
    for num in range(len(line) - 2):
        if num + 3 > len(line):
            break
        counter = 0
        if line[num] == line[num + 1] and line[num + 1] == line[num + 2]:
            while True:
                if num + 3 + counter > len(line):
                    break
                if line[num + 2 + counter] != line[num]:
                    break
                counter = counter + 1
            line = line.replace(line[num:(num + 2 + counter)], line[num:(num + 3)])
            # print(line)
    df_test.loc[ind, "newtext"] = line



# Abbreviation
# Except contraction, there are some abbreviation.
# Similarly, there is a abb dict and a slangs text,conbine them and check our text.
abbreviations = {"$": " dollar ",
"€": " euro ",
"4ao": "for adults only",
"a.m": "am",
"a3": "anytime anywhere anyplace",
"aamof": "as a matter of fact",
"acct": "account",
"adih": "another day in hell",
"afaic": "as far as i am concerned",
"afaict": "as far as i can tell",
"afaik": "as far as i know",
"afair": "as far as i remember",
"afk": "away from keyboard",
"app": "application",
"approx": "approximately",
"apps": "applications",
"asl": "age, sex, location",
"atk": "at the keyboard",
"ave.": "avenue",
"aymm": "are you my mother",
"ayor": "at your own risk",
"b&b": "bed and breakfast",
"b+b": "bed and breakfast",
"b.c": "bc",
"b2b": "business to business",
"b2c": "business to customer",
"b4": "before",
"b4n": "bye",
"b@u": "back at you",
"bae": "before anyone else",
"bak": "back at keyboard",
"bbbg": "bye bye be good",
"bbias": "be back in a second",
"bbl": "be back later",
"bbs": "be back soon",
"be4": "before",
"bfn": "bye for now",
"blvd": "boulevard",
"bout": "about",
"brb": "be right back",
"bros": "brothers",
"brt": "be right there",
"bsaaw": "happy",
"bwl": "happy",
"c/o": "care of",
"cf": "compare",
"csl": "happy",
"cu": "bye",
"cul8r": "see you later",
"cwot": "complete waste of time",
"cya": "bye",
"cyt": "bye",
"dae": "does anyone else",
"dbmib": "do not bother me i am busy",
"diy": "do it yourself",
"dm": "direct message",
"dwh": "during work hours",
"e123": "easy as one two three",
"eg": "example",
"embm": "early morning business meeting",
"encl": "enclosed",
"encl.": "enclosed",
"faq": "frequently asked questions",
"fawc": "for anyone who cares",
"fb": "facebook",
"fc": "fingers crossed",
"fig": "figure",
"fimh": "forever in my heart",
"ft.": "feet",
"ft": "featuring",
"ftl": "for the loss",
"ftw": "for the win",
"fwiw": "for what it is worth",
"fyi": "for your information",
"g9": "genius",
"gahoy": "get a hold of yourself",
"gal": "get a life",
"gcse": "general certificate of secondary education",
"gfn": "gone for now",
"gg": "good game",
"gl": "good luck",
"glhf": "good luck have fun",
"gmt": "greenwich mean time",
"gmta": "great minds think alike",
"gn": "good night",
"g.o.a.t": "greatest of all time",
"goat": "greatest of all time",
"goi": "get over it",
"gps": "global positioning system",
"gr8": "great",
"gratz": "congratulations",
"gyal": "girl",
"h&c": "hot and cold",
"hp": "horsepower",
"hr": "hour",
"hrh": "his royal highness",
"ht": "height",
"ibrb": "i will be right back",
"ic": "i see",
"icq": "i seek you",
"icymi": "in case you missed it",
"idc": "i do not care",
"idgadf": "i do not give a fuck",
"idgaf": "i do not give a fuck",
"idk": "i do not know",
"ie": "that is",
"i.e": "that is",
"ifyp": "i feel your pain",
"IG": "instagram",
"iirc": "if i remember correctly",
"ilu": "i love you",
"ily": "i love you",
"imho": "in my humble opinion",
"imo": "in my opinion",
"imu": "i miss you",
"iow": "in other words",
"irl": "in real life",
"j4f": "just for fun",
"jic": "just in case",
"jk": "just kidding",
"jsyk": "just so you know",
"l8r": "later",
"lb": "pound",
"lbs": "pounds",
"ldr": "long distance relationship",
"lmao": "happy",
"lmfao": "happy",
"lol": "happy",
"ltd": "limited",
"ltns": "long time no see",
"m8": "mate",
"mf": "motherfucker",
"mfs": "motherfuckers",
"mfw": "my face when",
"mofo": "motherfucker",
"mph": "miles per hour",
"mr": "mister",
"mrw": "my reaction when",
"ms": "miss",
"mte": "my thoughts exactly",
"nagi": "not a good idea",
"nbc": "national broadcasting company",
"nbd": "not big deal",
"nfs": "not for sale",
"ngl": "not going to lie",
"nhs": "national health service",
"nrn": "no reply necessary",
"nsfl": "not safe for life",
"nsfw": "not safe for work",
"nth": "nice to have",
"nvr": "never",
"nyc": "new york city",
"oc": "original content",
"og": "original",
"ohp": "overhead projector",
"oic": "oh i see",
"omdb": "over my dead body",
"omw": "on my way",
"p.a": "per annum",
"p.m": "pm",
"poc": "people of color",
"pov": "point of view",
"pp": "pages",
"ppl": "people",
"prw": "parents are watching",
"ps": "postscript",
"pt": "point",
"ptb": "please text back",
"pto": "please turn over",
"qpsa": "what happens",
"ratchet" : "rude",
"rbtl": "read between the lines",
"rlrt": "real life retweet",
"rofl": "happy",
"roflol": "happy",
"rotflmao": "happy",
"rt": "retweet",
"ruok": "are you ok",
"sfw": "safe for work",
"sk8": "skate",
"smh": "dismay",
"sq": "square",
"srsly": "seriously",
"ssdd": "same stuff different day",
"tbh": "to be honest",
"tbs": "tablespooful",
"tbsp": "tablespooful",
"tfw": "that feeling when",
"thks": "thanks",
"tho": "though",
"thx": "thanks",
"tia": "thanks",
"til": "today i learned",
"tl;dr": "too long i did not read",
"tldr": "too long i did not read",
"tmb": "tweet me back",
"tntl": "trying not to laugh",
"ttyl": "talk to you later",
"u": "you",
"u2": "you too",
"u4e": "yours for ever",
"utc": "coordinated universal time",
"w/": "with",
"w/o": "without",
"w8": "wait",
"wassup": "what is up",
"wb": "welcome back",
"wtf": "what the fuck",
"wtg": "way to go",
"wtpa": "where the party at",
"wuf": "where are you from",
"wuzup": "what is up",
"wywh": "wish you were here",
"yd": "yard",
"ygtr": "you got that right",
"ynk": "you never know",
"zzz": "sleeping",
"ig": "instagram",
"hwy": "highway",
"ave": "avenue",
"rd": "road",
'lmfao': 'happy'
}
# read slangs from the slang.txt and transform it into a dictionary
filename = 'C:/Users/shangjinyi/PycharmProjects/Slangs.txt'

slangs = {}
with open(filename,encoding='utf-8') as fh:
    for line in fh:
        command, description = line.strip().split(None, 1)
        slangs[command] = description.strip()

# merge abbreviations into slangs
slangs.update(abbreviations)
# Removed some  ambiguous ones
#del slangs['ft.']
#del slangs['goat']
#del slangs['ps']
#slangs['til']='till'
#slangs['tbs']='tablespoonful'
#slangs["2day"] = 'today'

# Function of Converting abbreviation
def convert_abbrev(word):
    return slangs[word.lower()] if word.lower() in slangs.keys() else word
def convert_abbrev_in_text(text):
    tokens = nltk.word_tokenize(text)
    tokens=[convert_abbrev(word) for word in tokens]
    text = ' '.join(tokens)
    return text

# Convert slangs in test and train data
df_train['newtext'] = df_train['newtext'].apply(lambda x: convert_abbrev_in_text(x))
df_test['newtext'] = df_test['newtext'].apply(lambda x: convert_abbrev_in_text(x))



#lowercase:
def lowerstrip_text(text):
    text = text.lower()
    text = text.strip()
    return text
df_train['newtext'] = df_train['newtext'].apply(lambda x: lowerstrip_text(x))
df_test['newtext'] = df_test['newtext'].apply(lambda x: lowerstrip_text(x))

# punctuation:
# Here I remove all of punctuation
def remove_punct(text):
    text = re.sub(r"û", " ", text)
    text = re.sub(r"ï", " ", text)
    text = re.sub(r"ª", " ", text)
    text = re.sub(r"ò", " ", text)
    text = re.sub(r"å", " ", text)
    text = re.sub(r"ó", " ", text)
    text = re.sub(r"ê", " ", text)
    text = re.sub(r"ì", " ", text)
    text = re.sub(r"ü", " ", text)
    text = re.sub(r"¤", " ", text)
    text = re.sub(r"/", " ", text)

    table = str.maketrans('','', '"!#@$%&\'()*+,-./:;<=>[\\]^_`{|}~')
    return text.translate(table)


df_train['newtext'] = df_train['newtext'].apply(lambda x: remove_punct(x))
df_test['newtext'] = df_test['newtext'].apply(lambda x: remove_punct(x))


# Stopwords
# Next, I delete some stop words, in text some of words are meaningless, like 'to','from'etc
# In the nltk.corpus,there is a stopword scorpus and we can use it to accomplish this.
stop=set(stopwords.words('english'))
stop = set(list(stop) + ["s", "nt", "m",])
stop.remove('no')
stop.remove('nor')
stop.remove('not')

def remove_stopwords(text):
    if text is not None:
        tokens = [x for x in word_tokenize(text) if x not in stop]
        return " ".join(tokens)
    else:
        return None

df_train['newtext']=df_train['newtext'].apply(lambda x : remove_stopwords(x))
df_test['newtext']=df_test['newtext'].apply(lambda x : remove_stopwords(x))


# Keywords Variable
# In the training set, some of the sentence have keywords variable. Therefore, we add this to the sentence
df_train['keyword']=df_train['keyword']+'kywrdsss'
df_train['keyword']=df_train['keyword'].fillna("")
df_test['keyword']=df_test['keyword']+'kywrdsss'
df_test['keyword']=df_test['keyword'].fillna("")
df_train['newtext']=df_train['keyword']+' '+df_train['newtext']
df_test['newtext']=df_test['keyword']+' '+df_test['newtext']




#Finally
# Here, we can see the final different between originaltext and newtext
Compare=list(zip(df_train['text'],df_train['newtext']))
#print(Compare)

#output as csv
train=df_train.drop(columns=['keyword','location','target'])
test=df_test.drop(columns=['keyword','location'])
Compare=df_train[['text','newtext']]

# train.to_csv('RealorNottrain.csv', index=False)
# test.to_csv('RealorNottest.csv', index=False)

# Compare.to_csv('RealorNotCompare.csv', index=False)
