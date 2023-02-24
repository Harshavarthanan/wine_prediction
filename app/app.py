import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
import jsonpickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Initialize the Flask application
app = Flask(__name__)

# Load the saved model from the pickle file
with open('wine_pca_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('standard_scaler.pkl', 'rb') as f:
    sc = pickle.load(f)
with open('pca.pkl', 'rb') as f:
    pca = pickle.load(f)



# Define the home page that displays the HTML form
@app.route('/api/home')
def home():
    return render_template('index.html')

# Define the API endpoint for making predictions
@app.route('/api/predict', methods=['POST'])
def predict():
    # Get the input features from the form data
    feature1 = request.form['feature1']
    feature2 = request.form['feature2']
    feature3 = request.form['feature3']
    feature4 = request.form['feature4']
    feature5 = request.form['feature5']
    feature6 = request.form['feature6']
    feature7 = request.form['feature7']
    feature8 = request.form['feature8']
    feature9 = request.form['feature9']
    feature10 = request.form['feature10']
    feature11 = request.form['feature11']
    feature12 = request.form['feature12']
    feature13 = request.form['feature13']
    # repeat for all 13 features
    input_features = {
        'Feature 1': feature1,
        'Feature 2': feature2,
        'Feature 3': feature3,
        'Feature 4': feature4,
        'Feature 5': feature5,
        'Feature 6': feature6,
        'Feature 7': feature7,
        'Feature 8': feature8,
        'Feature 9': feature9,
        'Feature 10': feature10,
        'Feature 11': feature11,
        'Feature 12': feature12,
        'Feature 13': feature13
    }
    
    # Pack the input features into a NumPy array
    X_test = np.array([feature1, feature2, feature3, feature4, feature5, feature6, feature7, feature8, feature9, feature10, feature11, feature12, feature13])

# Transform the input data using the same StandardScaler and PCA objects

    X_test = sc.transform(X_test.reshape(1,-1))
    X_test = pca.transform(X_test)
    
    
    # Make the prediction using the trained model
    y_pred = model.predict(X_test)
    
    # Return the predicted label to the HTML page
    return render_template('index.html', prediction_text='Customer segment: {}'.format(y_pred),input_features=input_features)

# Run the Flask application
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
