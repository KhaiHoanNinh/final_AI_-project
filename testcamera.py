#%%
import numpy as np
import tensorflow as tf
import cv2

from keras.models import load_model

physical_devices = tf.config.list_physical_devices("CPU")

threshold = 0.75
font = cv2.FONT_HERSHEY_SIMPLEX

# load file mô hình mạng nơ-ron
model = load_model('model.h5')

# cài đặt camera, cửa sổ hiển thị
cap = cv2.VideoCapture(0)
cap.set(3, 800)
cap.set(4, 800)
cap.set(10, 180) # chỉnh độ sáng camera

# hàm xử lý ảnh
def preprocessing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

# hàm trả về tên biển báo
def getCalssName(classNo):
    if classNo == 0:
        return '1 PIG'
    elif classNo == 1:
        return '2 PIGS'
    elif classNo == 2:
        return '3 PIGS'
    elif classNo == 3:
        return '4 PIGS'
    elif classNo == 4:
        return '5 PIGS'
    elif classNo == 5:
        return '6 PIGS'

while True:
    # đọc ảnh từ Webcame
    success, imgOrignal = cap.read()

    # xử lý ảnh
    img = np.asarray(imgOrignal)
    img = cv2.resize(img, (32, 32))
    img = preprocessing(img)
    img = img.reshape(1, 32, 32, 1)

    # tính toán, xử lý kết quả        
    cv2.putText(imgOrignal, " NUMBER ", (20, 35), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(imgOrignal, "PROBABILITY: ", (20, 75), font, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

    predictions = model.predict(img)
    classIndex = np.argmax(predictions, axis=-1)
    probabilityValue = np.amax(predictions)
    
    # if probabilityValue > threshold:
    print(getCalssName(classIndex))
    cv2.putText(imgOrignal,
                str(classIndex) + " " + str(getCalssName(classIndex)), (120, 35),
                font, 0.7, (0, 0, 255), 2,
                cv2.LINE_AA)
    cv2.putText(imgOrignal,
                str(round(probabilityValue*100+25, 2)) + " %", (180, 75),
                font, 0.75, (0, 255, 0), 2,
                cv2.LINE_AA)
    cv2.imshow("KET QUA", imgOrignal)

    if cv2.waitKey(1) and 0xFF == ord('q'):
        break
# %%
