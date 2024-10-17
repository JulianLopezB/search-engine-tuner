import pandas as pd
from bs4 import BeautifulSoup
import logging
from .text_preprocessing import preprocess_text
from tqdm import tqdm

logger = logging.getLogger(__name__)

class DataPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path

    def process_article(self, html_content):
        logger.debug("Preprocessing text")
        # Remove HTML tags
        soup = BeautifulSoup(html_content, 'html.parser')
        text = soup.get_text()
        
        text = preprocess_text(text)
        
        return text

    def _load_and_filter_data(self):
        df = pd.read_excel(self.file_path)
        df = df[df['obsoleto'].isna()]
        df = df[df['revisado'] == 's']
        return df.reset_index(drop=True)

    def _process_row(self, row):
        pregunta_html = str(row['pregunta'])
        respuesta_html = str(row['respuesta'])
        
        pregunta_text = self.process_article(pregunta_html)
        respuesta_text = self.process_article(respuesta_html)
        
        full_text = pregunta_text + ' ' + respuesta_text
        
        return {
            'id': row['id'],
            'text': full_text,
            'pregunta': row['pregunta'],
            'respuesta': row['respuesta'],
            'grupo': row['grupo']
        }

    def get_preprocessed_data(self):
        df = self._load_and_filter_data()
        return [self._process_row(row) for _, row in tqdm(df.iterrows())]

def load_training_data(file_path):
    preprocessor = DataPreprocessor(file_path)
    return preprocessor.get_preprocessed_data()