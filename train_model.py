import pandas as pd
import numpy as np
import cv2
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
import model as br

s_epoch = 25
m_epoch = 5
batch = 5000
m_batch = 128
loaded = False
x = None
y = None

def load_data(args = 0.1):
    data_df = pd.read_csv('driving_log.csv', names=['center', 'left', 'right', 'steering', 'throttle', 'reverse', 'speed'])

    X = data_df[['center', 'left', 'right', 'speed']].values[20:]
    y = data_df[['steering', 'throttle', 'reverse']].values[20:]

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=args, random_state=0)

    return X_train, y_train, X_valid, y_valid

def get_train_data(bno):
    global loaded
    global x, y
    if(loaded == False):
        print("Loading Dataset ...")
        loaded = True
        x, y, na, nb = load_data(0)
    lenx, p = x.shape
    bl = int(lenx / batch)
    bno %= bl
    ansxa = np.zeros([batch * 1, 60, 160, 3])
    ansxb = np.zeros([batch * 1, 1])
    ansy = np.zeros([batch * 1, 1])

    for i in range(0, batch):
        for j in range(0, 3):
            image = cv2.imread(x[(bno * batch) + i][0])
            image = np.array((image[20:140:2, ::2]/1), dtype = 'float32')
            ansxa[i] = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            #cv2.imshow("Dusk", (image/256))
            ansxb[i][0] = (x[(bno * batch) + i][3]/30)
            ansy[i] = np.array([y[(bno * batch) + i][0]])
            #mm = cv2.waitKey(1)

    #cv2.destroyAllWindows()
    ansxa -= 128
    ansxa /= 128
    return (ansxa, ansxb), ansy

for i in range(0, s_epoch):
    (xa, xb), y = get_train_data(i)
    br.model.fit(xa, y, batch_size = m_batch, epochs = m_epoch, verbose = 0)

br.model.save_weights('model')
