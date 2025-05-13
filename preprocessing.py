import re
from textblob import TextBlob
import nltk
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory

nltk.download('punkt', download_dir='nltk_data')
nltk.data.path.append('./nltk_data')

# Setup
stemmer = StemmerFactory().create_stemmer()
stopword_remover = StopWordRemoverFactory().create_stop_word_remover()

def normalize_slang(text, slang_dict):
    wordlist = TextBlob(text).words
    normalized_words = []
    for word in wordlist:
        if word in slang_dict:
            normalized_words.append(slang_dict[word])
        else:
            normalized_words.append(word)
    return ' '.join(normalized_words)

def preprocess_text(text, slang_dict):
    text = text.lower()
    text = re.sub(r'@[A-Za-z0-9_]+','', text)
    text = re.sub(r'#\w+','', text)
    text = re.sub(r'RT[\s]','', text)
    text = re.sub(r'https?://\S+','', text)
    text = re.sub(r'r\$\w*', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'[0-9]+', '', text)
    text = re.sub(r'[^A-Za-z0-9 ]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'(.)\1+', r'\1\1', text)

    text = normalize_slang(text, slang_dict)
    text = stopword_remover.remove(text)
    text = stemmer.stem(text)
    return text
