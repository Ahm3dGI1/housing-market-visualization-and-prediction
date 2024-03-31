from flask import Flask

from flask import Flask, request, render_template


import numpy as np
from joblib import load
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from neighborhoods import neighborhoods_list


model_path = "ML\\random_forest_rent_predictor.joblib"
model = load(model_path)

encoder = load(
    'ML\\encoder.joblib')
scaler = load(
    'ML\\scaler.joblib')
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
        sale_price_sqr_foot = 95
        housing_units = 374242

        # Prepare input data as a DataFrame
        input_df = pd.DataFrame([[year, neighborhood, sale_price_sqr_foot, housing_units]],
                                columns=['year', 'neighborhood', 'sale_price_sqr_foot', 'housing_units'])

        # Encode the 'neighborhood' feature
        neighborhood_encoded = encoder.transform(input_df[['neighborhood']])
        encoded_df = pd.DataFrame(
            neighborhood_encoded, columns=encoder.get_feature_names_out())

        # Concatenate encoded neighborhood with the rest of the features
        input_df.drop('neighborhood', axis=1, inplace=True)
        final_input_df = pd.concat([input_df, encoded_df], axis=1)

        # Scale the features
        scaled_features = scaler.transform(final_input_df)

        # Make prediction
        predicted_rent = model.predict(scaled_features)[0]
        print (predicted_rent)
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
