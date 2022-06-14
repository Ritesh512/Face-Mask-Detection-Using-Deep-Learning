from flask import Flask,render_template,Response
import cv2
from keras.models import load_model
import numpy as np
import keras.backend as kb

model = load_model('result.model')
face_clsfr = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
labels_dict= {1:"Mask",0:"No Mask"}
color_dict = {1:(0,255,0),0:(0,0,255)}

app=Flask(__name__)
camera=cv2.VideoCapture(0)

def generate_frames():
    while True:
            
        ## read the camera frame
        success,frame=camera.read()
        face_clsfr = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        if not success:
            break
        else:
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces=face_clsfr.detectMultiScale(gray,1.1,7)
            for (x,y,w,h) in faces:
                face_img = gray[y:y+h,x:x+w]
                resize = cv2.resize(face_img,(100,100))
                normalized = resize/255.0
                reshape = np.reshape(normalized,(1,100,100,1))
                result = model.predict(reshape)
                
                label = np.argmax(result,axis=1)[0]
                
                cv2.rectangle(frame,(x,y),(x+w,y+h),color_dict[label],2)
                cv2.rectangle(frame,(x,y-40),(x+w,y),color_dict[label],-1)
                cv2.putText(frame,labels_dict[label],(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
                
                acc=round(np.max(result,axis=1)[0]*100,2)
                cv2.putText(frame,str(acc),(x+150,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)


            ret,buffer=cv2.imencode('.jpg',frame)
            frame=buffer.tobytes()

            yield(b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        kb.clear_session()


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__=="__main__":
    app.run(debug=True)