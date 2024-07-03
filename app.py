from flask import Flask, render_template, request, redirect, url_for
from predict_disease import prediction_disease_type
import cv2
import numpy as np
app = Flask(__name__)
app.static_folder = "static/"
app.config['UPLOAD_FOLDER']="Uploads"
import os

# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Upload route
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        PDT = prediction_disease_type()
        
        # Assuming you have an image file uploaded
        plant_type = request.form['plant_type']
        image_file = request.files['image']
        # image_content=open(image_file,'rb').read()
        # print(image_content)
        if image_file.filename == '':
            return 'No selected file', 400

        # print(image)    
        print(image_file.filename)
        image_path = os.path.join(app.config['UPLOAD_FOLDER'], image_file.filename)
        print(image_path)
        try:
            image_file.save(image_path)
        except:
            print("error")
        disease="Rust"
        prob = 91
        try:
            result=PDT.get_label(image_path,plant_type)
            disease=result[2]
            prob=result[3]
            print("predicted disease",disease)
            
        except:
            print("error")
        remedy = "kkkkk"
        print("predicted disease",disease)
        return render_template('result.html', predicted_disease=disease, remedy=remedy,prob = prob)

    return render_template('upload.html')

# Result route
@app.route('/result')
def result():
    # Get data passed from the upload route
    predicted_disease = request.args.get('predicted_disease')
    # remedy = request.args.get('remedy')
    remedy = "spray"
    # Example print statement
    print(f"Predicted Disease: {predicted_disease}, Remedy: {remedy}")

    return render_template('result.html', predicted_disease=predicted_disease, remedy=remedy)

# Add any additional routes and logic as needed



if __name__ == '__main__':
    app.run(debug=True)