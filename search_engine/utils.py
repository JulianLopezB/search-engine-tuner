from bs4 import BeautifulSoup

def extract_text_from_html(html_content: str) -> str:
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Remove script and style elements
    for script_or_style in soup(['script', 'style']):
        script_or_style.decompose()
    
    # Get text and handle encoding
    text = soup.get_text(separator=' ', strip=True)
    return text
