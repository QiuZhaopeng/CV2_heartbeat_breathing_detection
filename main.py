import numpy as np
import cv2
from matplotlib import pyplot as plt
import time

x1 = 0.4
x2 = 0.6
y1 = 0.1
y2 = 0.25
face_cascade=cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def getFaceROI(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.2, 5)

    if len(faces)>0:
        img=cv2.rectangle(img, (faces[0][0] + int(x1*faces[0][2]), faces[0][1]+ int(y1*faces[0][3]) ), (faces[0][0]+ int(x2*faces[0][2]), faces[0][1]+ int(y2*faces[0][3]) ), (255, 0, 0), 2)
        
        return [faces[0][0]+int(x1*faces[0][2]),  faces[0][1]+ int(y1*faces[0][3]),  faces[0][0]+ int(x2*faces[0][2]), faces[0][1]+ int(y2*faces[0][3])]
    else:
        return [0,0,0,0]


cv2.namedWindow("tracking")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

cap.set(cv2.CAP_PROP_FPS, 10)
fps = cap.get(cv2.CAP_PROP_FPS)
print (fps)
step = int(1000/fps)



def getColorSum(frame, color_id):
    return frame[:,:,color_id].sum()

def getColorAverage(frame, color_id):
    return frame[:,:,color_id].sum()*1.0 / (frame.shape[0]*frame.shape[1])


ok, frame=cap.read()
if not ok:
    print('Failed to read video from camera')
    exit()


idf = 0
rsums = []
gsums = []
bsums = []

plt.close() 
fig=plt.figure()
plt.grid(True) 
plt.ion()  #interactive mode on
plt.xlabel('Time'), 
plt.ylabel('Amplitude') 

plt.title('Heart rate and breath detection')
    
bbox = [0,0,10,10]
m_diff = 0

flag = True
while(flag):
    # start = time.clock()
    ret,frame = cap.read()

    
    if (idf==0):
        
        droi = getFaceROI(frame)
        if (droi[3] > 0) :
            bbox = droi
    
    if(idf > 0):
        df = cv2.absdiff(frame, previous_frame)
        m_diff = 1.0 * df.sum() / (df.shape[0]*  df.shape[1])

        if (m_diff > 15): 
            droi = getFaceROI(frame)
            if (droi[3] > 0) :
                bbox = droi 


    roi = frame[  bbox[1]:bbox[3], bbox[0]:bbox[2]];
    frame=cv2.rectangle(frame, (bbox[0] , bbox[1] ), ( bbox[2],  bbox[3] ), (255, 0, 0), 2)

    green = getColorAverage(roi, 1)  ## 2nd channel for Green color
    if(idf>50):
        gsums.append(green)

    idf+=1
    previous_frame = frame
    cv2.imshow("tracking", frame);

    if(idf> 0):
        
        plt.clf()                  
        plt.plot(gsums[-200:-1], 'g')
        plt.pause(0.001)           
        plt.ioff()            
    k = cv2.waitKey(step) & 0xff
    # end = time.clock()
    if k == 27:    # esc pressed
       flag = False
       break 

    # exit when tracking windows is closed
    if cv2.getWindowProperty('tracking', cv2.WND_PROP_AUTOSIZE) < 1:
        cap.release()
        break

plt.close() 
cv2.destroyAllWindows()
cv2.waitKey(1)
print("Execution has finished")