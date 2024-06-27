from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load the best model and feature names
model = joblib.load('saved_model.pkl')
feature_names = joblib.load('feature_names.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    bedrooms = int(request.form['bedrooms'])
    baths = int(request.form['baths'])

    # Create a DataFrame for prediction with the correct feature names
    input_data = pd.DataFrame([[bedrooms, baths]], columns=['bedrooms', 'baths'])
    
    # Ensure all features are present and in the correct order
    for feature in feature_names:
        if feature not in input_data.columns:
            input_data[feature] = 0  # or another default value

    input_data = input_data[feature_names]

    # Predict the price
    prediction = model.predict(input_data)[0]

    return render_template('index.html', price=prediction)

if __name__ == '__main__':
    app.run(debug=True)
