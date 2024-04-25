from flask import Flask, render_template, request, jsonify
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
app = Flask(__name__)

# Load the pickled model
with open('model/best_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Preprocessing function
def preprocess_text(text):
    if isinstance(text, str):
        # Text Cleaning: Remove special characters, punctuation, and stopwords
        text = re.sub(r'[^\w\s]', '', text)  # Remove special characters
        text = re.sub(r'\d+', '', text)  # Remove digits
        text = text.lower()  # Convert text to lowercase
        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(text)
        filtered_text = [word for word in word_tokens if word not in stop_words]

        # Text Normalization: Lemmatization
        lemmatizer = WordNetLemmatizer()
        normalized_text = [lemmatizer.lemmatize(word) for word in filtered_text]

        return normalized_text
    else:
        return ''

# Define a route to render the index.html page
@app.route('/')
def index():
    return render_template('index.html')

# Define a route to accept user input and predict sentiment
@app.route('/predict_sentiment', methods=['POST'])
def predict_sentiment():
    # Get user input from POST request
    user_input_text = request.form['review']
    
    # Preprocess the input text
    processed_input_text = preprocess_text(user_input_text)
    processed_input_text = ' '.join(processed_input_text)  # Convert list to string
    
    # Use the loaded model to predict the sentiment of the input text
    predicted_sentiment = model.predict([processed_input_text])
    
    # Map predicted sentiment to human-readable labels
    predicted_sentiment_label = "Positive" if predicted_sentiment[0] == 1 else "Negative"
    
    # Return the predicted sentiment to the user
    return render_template('index.html', sentiment=predicted_sentiment_label, review=user_input_text)

if __name__ == '__main__':
    app.run(host="0.0.0.0")