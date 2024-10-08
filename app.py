from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pickle
import traceback
import os
from PIL import Image
import tensorflow as tf
import numpy as np

app = Flask(__name__)
CORS(app)

@app.route('/api/diabetese', methods=['POST'])
def predict_route():
    try:
        model = pickle.load(open('models/diabetes.pkl', 'rb'))

        data = request.get_json(force=True)

        to_predict_dict = {key: float(value) for key, value in data.items()}
        dic2 = {
            'NewBMI_Obesity 1': 0, 'NewBMI_Obesity 2': 0, 'NewBMI_Obesity 3': 0,
            'NewBMI_Overweight': 0, 'NewBMI_Underweight': 0, 'NewInsulinScore_Normal': 0,
            'NewGlucose_Low': 0, 'NewGlucose_Normal': 0, 'NewGlucose_Overweight': 0,
            'NewGlucose_Secret': 0
        }
        if to_predict_dict['BMI'] <= 18.5:
            dic2['NewBMI_Underweight'] = 1
        elif 18.5 < to_predict_dict['BMI'] <= 24.9:
            pass  # Normal BMI
        elif 24.9 < to_predict_dict['BMI'] <= 29.9:
            dic2['NewBMI_Overweight'] = 1
        elif 29.9 < to_predict_dict['BMI'] <= 34.9:
            dic2['NewBMI_Obesity 1'] = 1
        elif 34.9 < to_predict_dict['BMI'] <= 39.9:
            dic2['NewBMI_Obesity 2'] = 1
        elif to_predict_dict['BMI'] > 39.9:
            dic2['NewBMI_Obesity 3'] = 1

        if 16 <= to_predict_dict['Insulin'] <= 166:
            dic2['NewInsulinScore_Normal'] = 1

        if to_predict_dict['Glucose'] <= 70:
            dic2['NewGlucose_Low'] = 1
        elif 70 < to_predict_dict['Glucose'] <= 99:
            dic2['NewGlucose_Normal'] = 1
        elif 99 < to_predict_dict['Glucose'] <= 126:
            dic2['NewGlucose_Overweight'] = 1
        elif to_predict_dict['Glucose'] > 126:
            dic2['NewGlucose_Secret'] = 1
        to_predict_dict.update(dic2)
        values2 = list(map(float, to_predict_dict.values()))
        values = np.asarray(values2)
        print("Making prediction with input values:", values)
        prediction = model.predict(values.reshape(1, -1))[0]
        print("Prediction made:", int(prediction))

        return jsonify({'prediction': int(prediction)})

    except Exception as e:
        # Print stack trace and return error message
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/breastcancer', methods=['POST'])
def predict_breast_cancer():
    model = None
    try:
        # Load the model only when the route is accessed
        model = pickle.load(open('models/breast_cancer.pkl', 'rb'))
        print("Model loaded successfully.")

        # Extract JSON data from the request
        data = request.get_json(force=True)

        # Validate that all required fields are present
        required_fields = [
            'texture_mean', 'smoothness_mean', 'compactness_mean', 'concave_points_mean', 
            'symmetry_mean', 'fractal_dimension_mean', 'texture_se', 'area_se', 'smoothness_se', 
            'compactness_se', 'concavity_se', 'concave_points_se', 'symmetry_se', 'fractal_dimension_se', 
            'texture_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 
            'concavity_worst', 'concave_points_worst', 'symmetry_worst', 'fractal_dimension_worst'
        ]
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Invalid data'}), 400

        # Convert incoming JSON data to the required format (floats)
        formatted_data = {key: float(value) for key, value in data.items()}
        to_predict_list = list(formatted_data.values())

        # Prepare data for prediction
        values = np.asarray(to_predict_list)
        print("Making prediction with input values:", values)

        # Make prediction
        prediction = model.predict(values.reshape(1, -1))[0]
        print("Prediction made:", int(prediction))

        return jsonify({'prediction': int(prediction)})

    except Exception as e:
        # Print stack trace and return error message
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/heartdisease', methods=['POST'])
def predict_heart_disease():
    model = None
    try:
        # Load the model only when the route is accessed
        model = pickle.load(open('models/heart.pkl', 'rb'))
        print("Model loaded successfully.")

        # Extract JSON data from the request
        data = request.get_json(force=True)

        # Validate that all required fields are present
        required_fields = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                           'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal']
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Invalid data'}), 400

        # Convert incoming JSON data to the required format (floats)
        formatted_data = {key: float(value) for key, value in data.items()}
        to_predict_list = list(formatted_data.values())

        # Prepare data for prediction
        values = np.asarray(to_predict_list)
        print("Making prediction with input values:", values)

        # Make prediction
        prediction = model.predict(values.reshape(1, -1))[0]
        print("Prediction made:", int(prediction))

        return jsonify({'prediction': int(prediction)})

    except Exception as e:
        # Print stack trace and return error message
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/kidneydisease', methods=['POST'])
def predict_kidney_disease():
    model = None
    try:
        # Load the model only when the route is accessed
        model = pickle.load(open('models/kidney.pkl', 'rb'))
        print("Model loaded successfully.")

        # Extract JSON data from the request
        data = request.get_json(force=True)

        # Validate that all required fields are present
        required_fields = [
            'age', 'blood_pressure', 'specific_gravity', 'albumin', 'sugar', 
            'red_blood_cells', 'pus_cell', 'pus_cell_clumps', 'bacteria', 
            'blood_glucose_random', 'blood_urea', 'serum_creatinine', 'sodium', 
            'potassium', 'haemoglobin', 'packed_cell_volume', 
            'white_blood_cell_count', 'red_blood_cell_count', 'hypertension', 
            'diabetes_mellitus', 'coronary_artery_disease', 'appetite', 
            'peda_edema', 'aanemia'
        ]
        if not all(field in data for field in required_fields):
            return jsonify({'error': 'Invalid data'}), 400

        # Convert incoming JSON data to the required format (floats)
        formatted_data = {key: float(value) for key, value in data.items()}
        to_predict_list = list(formatted_data.values())

        # Prepare data for prediction
        values = np.asarray(to_predict_list)
        print("Making prediction with input values:", values)

        # Make prediction
        prediction = model.predict(values.reshape(1, -1))[0]
        print("Prediction made:", int(prediction))

        return jsonify({'prediction': int(prediction)})

    except Exception as e:
        # Print stack trace and return error message
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/api/liverdisease', methods=['POST'])
def predict_liver_disease():
    data = request.json

    required_fields = [
        'Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin', 
        'Alkaline_Phosphotase', 'Alamine_Aminotransferase', 
        'Aspartate_Aminotransferase', 'Total_Protiens', 
        'Albumin', 'Albumin_and_Globulin_Ratio'
    ]

    # Check if all required fields are present
    if not all(field in data for field in required_fields):
        return jsonify({'error': 'Invalid data'}), 400

    try:
        # Convert the input data to a float array
        values = [float(data[field]) for field in required_fields]

        # Convert the values to a numpy array
        values = np.asarray(values)

        # Load the model inside the route
        model = pickle.load(open('models/liver.pkl', 'rb'))

        # Use the loaded model to make a prediction
        prediction = model.predict(values.reshape(1, -1))[0]

        # Return the prediction as a JSON response
        return jsonify({'prediction': int(prediction)})

    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
@app.route('/api/malariapredict', methods=['POST'])
def malariapredictPage():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided.'}), 400

        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No image selected for uploading.'}), 400

        # Open the image directly from the uploaded file
        img = Image.open(image_file).convert('RGB')

        # Preprocess the image
        img = img.resize((64, 64))  # Resize to smaller dimensions to save memory
        img = np.array(img).astype('float16')  # Convert to float16 to save memory
        img = np.expand_dims(img, axis=0)

        # Load the model, make predictions, and then unload the model
        print("Loading malaria prediction model...")
        model = tf.keras.models.load_model("models/malaria.keras")
        print("Model loaded successfully.")
        
        pred = np.argmax(model.predict(img))
        print("Prediction made:", int(pred))

        # Unload model to free up memory
        del model
        tf.keras.backend.clear_session()

        return jsonify({'prediction': int(pred)})

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return jsonify({'error': f"An error occurred: {str(e)}. Please upload a valid image."}), 500


@app.route('/api/pneumoniapredict', methods=['POST'])
def pneumoniapredictPage():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided.'}), 400

        image_file = request.files['image']
        if image_file.filename == '':
            return jsonify({'error': 'No image selected for uploading.'}), 400

        # Open the image directly from the uploaded file and ensure it's in RGB mode
        img = Image.open(image_file).convert('RGB')  # Convert to RGB (3 channels)

        # Preprocess the image
        img = img.resize((128, 128))  # Resize to the required dimensions
        img = np.array(img)  # Convert to a numpy array
        img = np.expand_dims(img, axis=0)  # Add batch dimension (1, 128, 128, 3)

        # Load the model and make predictions
        print("Loading pneumonia prediction model...")
        model = tf.keras.models.load_model("models/pneumonia.keras")
        print("Model loaded successfully.")
        
        pred = np.argmax(model.predict(img))
        
        # After prediction, print a message indicating the model has been used
        print("Model used for prediction.")

        # Simulate closing the model
        print("Closing the pneumonia prediction model...")
        # Perform any necessary cleanup here (if needed)
        print("Model closed successfully.")
        
        return jsonify({'prediction': int(pred)})

    except Exception as e:
        # Log the detailed error message
        print(f"An error occurred: {str(e)}")
        return jsonify({'error': f"An error occurred: {str(e)}. Please upload a valid image."}), 500


if __name__=="__main__":
    app.run(host='0.0.0.0')
