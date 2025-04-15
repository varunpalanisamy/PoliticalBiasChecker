import joblib
import requests
from bs4 import BeautifulSoup
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
nltk.download('stopwords')


def scrape_website(url):
    try:
        response = requests.get(url)
        response.raise_for_status()  
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        text = " ".join([para.get_text() for para in paragraphs])
        return text
    except requests.exceptions.RequestException as e:
        print(f"Error fetching the website: {e}")
        return ""

def preprocess_text(text):
    import re
    from nltk.corpus import stopwords
    from nltk.stem import WordNetLemmatizer

    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    text = text.lower()

    text = re.sub(r'[^\w\s]', '', text)

    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return ' '.join(tokens)

def load_model_and_vectorizer():
    model = joblib.load('logreg_model.pkl')
    tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')
    return model, tfidf_vectorizer

def run_model_on_website(url):
    raw_text = scrape_website(url)

    if not raw_text:
        return "Error: No text found on the website."

    preprocessed_text = preprocess_text(raw_text)

    model, tfidf_vectorizer = load_model_and_vectorizer()

    vectorized_text = tfidf_vectorizer.transform([preprocessed_text])

    prediction = model.predict(vectorized_text)

    return f"The political bias of the article is: {prediction[0]}"

if __name__ == "__main__":
    url = input("Enter the URL of the news article: ")
    result = run_model_on_website(url)
    print(result)
