import os 
import numpy as np
import cv2
from flask import Flask
from flask import render_template, request, session
from breeds import breeds

#from tensorflow.keras.applications.resnet_v2 import preprocess_input
#from tensorflow.keras.models import load_model



app = Flask(__name__, template_folder='templates', static_folder='static')

UPLOAD_FOLDER = os.path.join('static', 'uploads')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = 'RD2002'

#model = load_model("model")


def prediction(image_name):
    path = "./static/uploads/" + image_name
    pred_img_array = cv2.resize(cv2.imread(path,cv2.IMREAD_COLOR),((224,224)))
    #pred_img_array = preprocess_input(np.expand_dims(np.array(pred_img_array[...,::-1].astype(np.float32)).copy(), axis=0))
    #pred_val = model.predict(np.array(pred_img_array,dtype="float32"))
    #pred_breed = sorted(breeds)[np.argmax(pred_val)]

    #return pred_breed

@app.route("/")
def main():
    return render_template('index.html')


@app.route("/", methods=['GET','POST'])
def classification():

    if request.method == "POST":
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.mkdir(app.config['UPLOAD_FOLDER'])

        _img = request.files['myPhoto']
        filename = _img.filename
        _img.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        session['uploaded_img_file_path'] = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        pred = prediction(filename)
                
        return render_template('prediction.html', breed=pred)       
    

if __name__ == "__main__":
    app.run(debug=False)