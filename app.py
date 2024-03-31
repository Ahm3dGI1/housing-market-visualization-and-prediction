from flask import Flask

from flask import Flask, request, render_template


import numpy as np
from joblib import load
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
from neighborhoods import neighborhoods_list


# Load the model
model = load(r"ML\gross_rent_linear_predictor.joblib")
encoder = load(r"ML\new_encoder.joblib")
interaction = load(r"ML\interaction.joblib")


# Initialize Flask application
app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def home():
    predicted_rent = None  # Initialize to None or some default value
    # If it's a POST request, we'll process the form data
    if request.method == 'POST':
        # Extract information from form
        year = int(request.form.get('year'))
        neighborhood = request.form.get('neighborhood')
        # One-hot encode the input neighborhood
        neighborhood_encoded = encoder.transform([[neighborhood]])
        
        # Prepare the input features with interaction terms
        input_features = np.hstack([np.array([[year]]), neighborhood_encoded])

        input_interaction = interaction.transform(input_features)
        
        # Predict the gross rent using the trained model
        predicted_rent = model.predict(input_interaction)[0]


        # Return the prediction result in HTML
        # You can also pass this to your index.html using render_template if you have a placeholder for it
        return render_template('home.html', neighborhoods=neighborhoods_list, predicted_rent=predicted_rent)

    # If it's a GET request, render the empty form inside index.html
    return render_template('home.html',neighborhoods=neighborhoods_list)

@app.after_request
def add_header(response):
    response.cache_control.no_store = True
    return response

if __name__ == '__main__':
    app.run(debug=True)
