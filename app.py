from flask import Flask, request, render_template
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import requests
from bs4 import BeautifulSoup
import nltk
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')

app = Flask(__name__)

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
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    text = text.lower()

    text = re.sub(r'[^\w\s]', '', text)

    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    return ' '.join(tokens)

model = joblib.load('logreg_model.pkl')
tfidf_vectorizer = joblib.load('tfidf_vectorizer.pkl')

@app.route('/')
def home():
    return render_template('index.html') 

@app.route('/predict', methods=['POST'])
def predict():
    if 'url' in request.form and request.form['url']:
        url = request.form['url']
        raw_text = scrape_website(url)
        if not raw_text:
            return render_template('index.html', prediction="Error: Could not fetch content from the provided URL.")
    else:
        raw_text = request.form['text']

    preprocessed_text = preprocess_text(raw_text)

    vectorized_text = tfidf_vectorizer.transform([preprocessed_text])

    prediction = model.predict(vectorized_text)

    label_mapping = {0: 'Left', 1: 'Center', 2: 'Right'}
    predicted_label = label_mapping[prediction[0]]

    return render_template('index.html', prediction=f"The political bias of the article is: {predicted_label}")

if __name__ == '__main__':
    app.run(debug=True)

