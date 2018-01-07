from six.moves import cPickle as pickle
import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import random

from sklearn.utils import shuffle
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import Imputer, LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import confusion_matrix, fbeta_score, make_scorer, accuracy_score
from sklearn.cross_validation import cross_val_score, train_test_split


def load_image(folder):
    valid_images = [".jpg", ".gif", ".png", ".tga"]
    all_seeds = os.listdir(folder)
    imgs_data = []
    Y = []
    c=0
    # print(all_seeds)
    for current_char in all_seeds:
        char_dir = os.path.join(folder, current_char)
        for image_name in os.listdir(char_dir):
            c+=1
            
            if(c==100)  
            break

            image_dir = os.path.join(char_dir, image_name)
            ext = os.path.splitext(image_dir)[1]
            #             print(image_name, image_dir)

            if ext.lower() not in valid_images:
                continue

            img = Image.open(image_dir)
            img = img.resize((28, 28), Image.ANTIALIAS)

            # img = np.array(img.resize((28, 28), Image.ANTIALIAS))
            # img.resize(28, 28, 3)
            img = np.array(img)

            img = img.reshape(28 * 28 * 3) / 255
            imgs_data.append(img)
            Y.append(current_char)


            #             except:
            #                 print("unable to fetch image")

    # print(np.array(imgs_data).shape, np.array(Y).reshape(len(Y), 1).shape)
    return np.array(imgs_data, dtype=np.float32).T, np.array(Y).reshape(1, len(Y))


def randomize(X_var, Y_var):
    X, Y = shuffle(X_var.T, Y_var.T, random_state=random.randint(0,9))
    print(X.T.shape,Y.T.shape)
    return X.T, Y.T


def data_preprocessing(path, imgcnt):
    X,Y = load_image(path, imgcnt)
    # Y_1 = LabelEncoder().fit_transform(Y.T)
    # Y_1 = Y_1.reshape(1, Y.shape[1])
    # onehot = OneHotEncoder(categorical_features=[1])
    # Y = onehot.fit_transform(Y_1)
    # X_1, Y_1 = randomize(X, Y)
    # X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = data_per,random_state = 42)
    print('loaded images')
    return X, Y

if __name__ == '__main__':
	X,Y = data_preprocessing('/home/apurvnit/Projects/aws_deeplearn/images/Segmented',0.8)
	print(X.shape, Y.shape)