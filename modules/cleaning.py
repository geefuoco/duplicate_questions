import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize


def lower(text):
    """Create a lower case version of a string and return a string
    E.g. 'Hello There! Nice to meet you' -> 'hello there! nice to meet you'
    Returns: str
    """
    if type(text) != type("hello"):
        print(f"{text} is not a string. It is of type {type(text)}")
        return str(text)
    return text.lower()

def remove_punctuation(text):
    """Removes all punctuation and single characters from a string 
    E.g. "What's going on here" -> "What going on here"
    Returns: str
    """
    subbed = re.sub(r"[^a-zA-Z ]", " ", text)
    subbed = re.sub(r"[0-9]", " ", subbed)
    subbed = re.sub(r"(^| ).($| )", " ", subbed) 
    subbed = re.sub(r"\s", " ", subbed) 
    return subbed

def tokenize(text):
    """Tokenizes a sentence into words and removes stop words
    E.g. How is it going -> ["How", "going"]
    Returns: List[str]"""
    
    tokens = word_tokenize(text)
    stop_words = stopwords.words("english")
    return [word for word in tokens if not word in stop_words]

def lemmatize(tokens):
    """Lemmatize a list of tokens
    E.g. ["we", "should", "go", "fishing"] -> ["we", "should", "go", "fish"]
    Returns: List[str]
    """
    
    lem = WordNetLemmatizer()
    return [lem.lemmatize(word) for word in tokens]


def clean(text):
    """Fully cleans a sentence of text by:
    - lowering text
    - removing punctuation
    - tokenizing words and removing stop words
    - lemmatizing words
    Returns: List[str]"""
    
    words = lower(text)
    words = remove_punctuation(words)
    words = tokenize(words)
    words = lemmatize(words)
    return words