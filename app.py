from flask import Flask, render_template, request
from keras.models import load_model
import numpy as np
import cv2
from tensorflow.keras.utils import load_img
from tensorflow.keras.utils import img_to_array
from keras.applications.vgg16 import preprocess_input
import os
    
app = Flask(__name__)
model = load_model('model.h5',compile=False)
model.compile(optimizer='rmsprop',
    loss='categorical_crossentropy',
    metrics=None,
    loss_weights=None,
    weighted_metrics=None,
    run_eagerly=None,
    steps_per_execution=None,
    jit_compile=None)

target_img = os.path.join(os.getcwd() , 'static/images')
@app.route('/')
def index_view():
    return render_template('index.html')


ALLOWED_EXT = set(['jpg' , 'jpeg' , 'png'])
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXT
           
# Function to load and prepare the image in right shape
@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename): #Checking file format
            filename = file.filename
            file_path = os.path.join('static/images', filename)
            file.save(file_path)
            img = cv2.imread(file_path) 
            img = cv2.resize(img,(28,28))
            img = np.resize(img, (1, 28, 28, 1))
            img = img/255.0
            img = 1 - img 

            predict_prob=model.predict(img) 
            prediction=np.argmax(predict_prob,axis=1)
            return render_template('index.html', prediction = prediction[0], user_image = file_path)
        else:
            return "Unable to read the file. Please check file extension"
        
@app.route('/predict_api',methods=['POST'])
def predict_api():
    if request.method == 'POST':
        file = request.files['file']
        if file and allowed_file(file.filename): #Checking file format
            filename = file.filename
            file_path = os.path.join('static/images', filename)
            file.save(file_path)
            img = cv2.imread(file_path) 
            img = cv2.resize(img,(28,28))
            img = np.resize(img, (1, 28, 28, 1))
            img = img/255.0
            img = 1 - img 

            predict_prob=model.predict(img) 
            prediction=np.argmax(predict_prob,axis=1)
            return f"The Given Image is of Number:{prediction[0]}"
        else:
            return "Unable to read the file. Please check file extension"
        
if __name__ == '__main__':
    app.run(debug=True,use_reloader=False, port=8000)