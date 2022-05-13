import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from PIL import Image
import PIL.ImageOps

x=np.load('image.npz')['arr_0']
y=pd.read_csv('labels.csv')['labels']

x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=0,train_size=7500,test_size=2500)
x_train_scaled=x_train/255.0
x_test_scaled=x_test/255.0

classifier=LogisticRegression(solver='saga',multi_class='multinomial')
classifier.fit(x_train_scaled,y_train)

def get_prediction(image):
    #Converting cv2 image to pil format
        im_pil=Image.open(image)
        
        image_bw=im_pil.convert('L')
        image_bw_resized = image_bw.resize((28,28), Image.ANTIALIAS)

        #scaling the picture
        px_filter=20
        #percentile function converts the values in scalar quantity
        min_pixel=np.percentile(image_bw_resized,px_filter)
        #using clipto limit the values between 0 to 255
        image_bw_resized_inverted_scaled=np.clip(image_bw_resized-min_pixel,0,255)
        max_pixel=np.max(image_bw_resized)
        #Change the data into an array to be used in our model for prediction
        image_bw_resized_inverted_scaled=np.asarray(image_bw_resized_inverted_scaled)/max_pixel
        
        #creating a test sample 
        test_sample=np.array(image_bw_resized_inverted_scaled).reshape(1,784)
        y_pred=classifier.predict(test_sample)

        return y_pred[0]

