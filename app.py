from flask import Flask, render_template, request, redirect, url_for
# from predict_disease import prediction_disease_type
import cv2
import numpy as np
# import io

app = Flask(__name__)
app.static_folder = "static/"


# Home route
@app.route('/')
def home():
    return render_template('index.html')

# Upload route
@app.route('/upload', methods=['GET', 'POST'])
def upload():
    
    if request.method == 'POST':
        
        # Assuming you have an image file uploaded
        plant_type = request.form['plant_type']
        image_file = request.files['image']

        if image_file.filename == '':
            return 'No selected file', 400

        # in_memory_file = io.BytesIO()
        # file.save(in_memory_file)
        # in_memory_file.seek(0)
        
        # # Convert the image data to a numpy array
        # file_bytes = np.frombuffer(in_memory_file.read(), dtype=np.uint8)
        
        # # Decode the numpy array into an image
        # img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        # Save the file to a temporary location

        # # Read image data directly from the file object
        # image_bytes = image_file.read()

        # # Decode the image bytes using OpenCV's imdecode
        # image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
        # print(image)
        disease="Rust"
        prob = 91
        # Assuming the file name is in the format "plant_disease.jpg"
        # You may need to adjust this depending on your naming convention

        # print()
        # Fetch corresponding remedy and product image from MongoDB
        remedy = "kkkkk"

        # Redirect to the result page, passing necessary data
        # return redirect(url_for('result', predicted_disease=predicted_disease, remedy=remedy))
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