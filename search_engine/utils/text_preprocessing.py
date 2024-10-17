import re
import unicodedata
from bs4 import BeautifulSoup
import contractions
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from langdetect import detect

# Ensure necessary NLTK data is downloaded
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
# nltk.download('punkt', download_dir='tokenizers/punkt/spanish')
# nltk.download('punkt', download_dir='tokenizers/punkt/portuguese')

def preprocess_text(raw_text):
    # Function to replace hashtags with comma-separated values
    def replace_hashtags(match):
        hashtags = match.group(0)
        # Remove the leading '#' and split by '#'
        tags = hashtags[1:].split('#')
        return ', '.join(tags)

    # Detect language
    try:
        language = detect(raw_text)
    except:
        language = 'es'  # Default to Spanish if detection fails

    # Map detected language to NLTK stopwords and tokenizer language codes
    if language == 'es':
        nltk_language = 'spanish'
    elif language == 'gl':
        nltk_language = 'galician'  # Use Galician if available, otherwise fallback to Spanish
    else:
        nltk_language = 'spanish'  # Default to Spanish

    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', raw_text, flags=re.MULTILINE)

    # Remove HTML tags
    text = BeautifulSoup(text, "html.parser").get_text()

    # Expand contractions
    text = contractions.fix(text)

    # Replace hashtags with comma-separated values
    text = re.sub(r'#(\w+)(#\w+)+', replace_hashtags, text)

    # Normalize unicode characters
    text = unicodedata.normalize('NFKD', text)

    # Replace accented characters with their non-accented counterparts
    text = ''.join([c for c in text if not unicodedata.combining(c)])

    # Convert to lowercase
    text = text.lower()

    # Remove special characters but keep alphanumeric, spaces, and some punctuation
    text = re.sub(r'[^a-z0-9áéíóúñ\s.,!?]', ' ', text)

    # Tokenize
    tokens = text.split()

    # Remove stopwords
    stop_words = set(stopwords.words(nltk_language))
    tokens = [token for token in tokens if token not in stop_words]

    # Lemmatize or stem
    if nltk_language == 'spanish':
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
    elif nltk_language == 'galician':
        stemmer = SnowballStemmer('spanish')  # Use Spanish stemmer for Galician
        tokens = [stemmer.stem(token) for token in tokens]

    # Join tokens back into a string
    preprocessed_text = ' '.join([t for t in tokens if len(t)<15])

    # Remove extra whitespace
    preprocessed_text = re.sub(r'\s+', ' ', preprocessed_text).strip()

    return preprocessed_text