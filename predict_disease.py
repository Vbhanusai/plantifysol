import tensorflow as tf #version 2.13.0
import keras #version 
from keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
import numpy as np
import cv2
import tensorflow as tf
import requests
import gdown
import io
import h5py
# def download():
#     # URL of the file on Google Drive
#     url = 'https://drive.google.com/uc?id=1sH-TwcZOPsUWeY_cqFzWqJtxECy1HrEI&export=download'


#     # Download the file
#     response = requests.get(url)
#     open('downloaded_file.h5', 'wb').write(response.content)

#     # Now you can load the file into your model
def download():
    url = 'https://drive.google.com/uc?id=1sH-TwcZOPsUWeY_cqFzWqJtxECy1HrEI'
    output = 'ml_model.h5'
    with io.BytesIO() as f:
        gdown.download(url, f)
    # gdown.download(url, output, quiet=False)

def load_pb_model(pb_file):
    try:
        # Load the protobuf model using TensorFlow
        with tf.io.gfile.GFile(pb_file, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
        
        # Import the graph definition into a new TensorFlow Graph
        with tf.Graph().as_default() as graph:
            tf.import_graph_def(graph_def, name="")
        
        print("Model loaded successfully!")
        return graph
    except Exception as e:
        print("Error loading model:", e)
        return None


class prediction_disease_type:
    def __init__(self):
        self.label_disease = {
            0 : 'Apple___Apple_scab',
            1 : 'Apple___Black_rot',
            2 : 'Apple___Cedar_apple_rust',
            3 : 'Apple___healthy',
            4 : 'Background_without_leaves',
            5 : 'Blueberry___healthy',
            6 : 'Cherry___Powdery_mildew',
            7 : 'Cherry___healthy',
            8 : 'Corn___Cercospora_leaf_spot_Gray_leaf_spot',
            9 : 'Corn___Common_rust',
            10: 'Corn___Northern_Leaf_Blight',
            11: 'Corn___healthy',
            12: 'Grape___Black_rot',
            13: 'Grape___Esca_(Black_Measles)',
            14: 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
            15: 'Grape___healthy',
            16: 'Orange___Haunglongbing_Citrus_greening',
            17: 'Peach___Bacterial_spot',
            18: 'Peach___healthy',
            19: 'Pepper_bell___Bacterial_spot',
            20: 'Pepper_bell___healthy',
            21: 'Potato___Early_blight',
            22: 'Potato___Late_blight',
            23: 'Potato___healthy',
            24: 'Raspberry___healthy',
            25: 'Soybean___healthy',
            26: 'Squash___Powdery_mildew',
            27: 'Strawberry___Leaf_scorch',
            28: 'Strawberry___healthy',
            29: 'Tomato___Bacterial_spot',
            30: 'Tomato___Early_blight',
            31: 'Tomato___Late_blight',
            32: 'Tomato___Leaf_Mold',
            33: 'Tomato___Septoria_leaf_spot',
            34: 'Tomato___Spider_mites_Two-,spotted_spider_mite',
            35: 'Tomato___Target_Spot',
            36: 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
            37: 'Tomato___Tomato_mosaic_virus',
            38: 'Tomato___healthy',
        }
        self.plant_label_disease={
            "apple":[0,1,2,3],
            "background_without_leaves":[4],
            "blueberry" : [5],
            "cherry" : [6,7],
            "corn" : [8,9,10,11],
            "grape" : [12,13,14,15],
            "orange" : [16] ,
            "peach" : [17,18],
            "pepper" : [19,20],
            "potato" : [21,22,23],
            "raspberry" : [24],
            "soybean" : [25],
            "squash" : [26],
            "strawberry" : [27,28],
            "tomato" : [29,30,31,32,33,34,35,36,37,38]
        }
        
        self.HEIGHT = 256
        self.WIDTH = 256
        
        # # Load the pickled model architecture
        # architecture_path = 'models/modelIncepAcc99.pkl'
        # with open(architecture_path, 'rb') as f:
        #     dnn_model = pickle.load(f)
        
        # # Load the pickled model weights
        # weights_path = 'models/model_weights.pkl'
        # with open(weights_path, 'rb') as f:
        #     dnn_model_weights = pickle.load(f)
        
        # # Create the model using the loaded architecture
        # dnn_model = tf.keras.models.model_from_json(dnn_model_architecture)
        
        # # Set the loaded weights to the model
        # dnn_model.set_weights(dnn_model_weights)
        
        
        #
        # url = 'https://drive.google.com/uc?id=1sH-TwcZOPsUWeY_cqFzWqJtxECy1HrEI'
        # output = 'ml_model.h5'
        # with io.BytesIO() as f:
        #     gdown.download(url, f)
        #     f.seek(0)  # Move the cursor to the beginning of the buffer
        IMAGE_SIZE = 256
        
        inception_model = tf.keras.applications.inception_v3.InceptionV3(
            include_top=False,
            # weights='imagenet',
            input_tensor=None,
            input_shape=(IMAGE_SIZE,IMAGE_SIZE,3),
            pooling='max'
        )

        for layer in inception_model.layers:
                layer.trainable = True
        dnn_model = Sequential([
            inception_model,
            BatchNormalization(axis= -1, momentum= 0.99, epsilon= 0.001),
            Dense(128, activation= 'relu'),
            Dropout(rate= 0.45, seed= 123),
            Dense(39, activation= 'softmax')
        ])
        weights_path = 'keras_savedmodel_weights.h5'
        dnn_model.load_weights(weights_path)
        # dnn_model = tf.saved_model.load(model_path)
        self.dnn_model = dnn_model
        print("done")
    def get_label(self,img,plant_type):
        self.img=img
        self.plant_type=plant_type
        
        process_img = cv2.resize(self.img, (self.HEIGHT, self.WIDTH),interpolation = cv2.INTER_LINEAR)
        process_img = process_img/(255)
        process_img = np.expand_dims(process_img, axis=0)
        
        
        y_pred = self.dnn_model.predict(process_img)
        print("y pred",y_pred)
        
        indx = np.argmax(y_pred)
        
        max_prob_indx = self.plant_label_disease[self.plant_type][0]
        
        for x in self.plant_label_disease[self.plant_type]:
            print(x,y_pred[0][x],max_prob_indx,y_pred[0][max_prob_indx])
            if y_pred[0][x]>y_pred[0][max_prob_indx]:
                max_prob_indx = x
        print("max prob is of plant type",y_pred[0][max_prob_indx],self.label_disease[max_prob_indx])
        print("max prob is of all plant type",y_pred[0][indx],self.label_disease[indx])
        return [(y_pred[0][max_prob_indx],self.label_disease[max_prob_indx]),(y_pred[0][indx],self.label_disease[indx])]
        # return [(75,"Potato"),(75,"Potato")]
     

if __name__ == "__main__":
    pass
        
