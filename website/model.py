# from typing import final
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Multiply
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model

import json,os,random,cv2
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import seaborn as sns
from PIL import Image

def random_image():

    path = './website/test_images'
    random_filename = random.choice([
        x for x in os.listdir(path)
        if os.path.isfile(os.path.join(path, x))
    ])
    
    im = np.array(Image.open(path+'/'+random_filename))
    cv2.imwrite('./website/static/random.png',im)

    def read_questions(path):
        with open(path, 'r') as file:
            qs = json.load(file)
        texts = [q[0] for q in qs]
        answers = [q[1] for q in qs]
        image_ids = [q[2] for q in qs]
        return (texts, answers, image_ids)

    test_qs,_, test_image_ids = read_questions('./website/test_answers.json')
    ind=np.where(np.array(test_image_ids)==int(random_filename[:-4]))
    df=pd.DataFrame(np.array(test_qs)[ind],columns=["Questions"])
    # print(df)
    qs_list=[]
    df1=np.array(df)
    for i in df1:
        qs_list.append(*i)
    
    test_image_path = path+'/'+random_filename
        
    return (test_image_path,random_filename,qs_list)
    



def predict(image,choice=1,qs_index=0,asked_qs="What shape is present?"):
    
    with open('./website/answers.txt', 'r') as file:
        all_answers = [a.strip() for a in file]
    
    def read_questions(path):
        with open(path, 'r') as file:
            qs = json.load(file)
        texts = [q[0] for q in qs]
        answers = [q[1] for q in qs]
        image_ids = [q[2] for q in qs]
        return (texts, answers, image_ids)

    test_qs, _, test_image_ids = read_questions('./website/test_answers.json')

    #print('\n--- Reading/processing images...')
    def load_and_proccess_image(image_path):
        # Load image, then scale and shift pixel values to [-0.5, 0.5]
        im = img_to_array(load_img(image_path))
        return im / 255 - 0.5

    def read_images(paths):
        # paths is a dict mapping image ID to image path
        # Returns a dict mapping image ID to the processed image
        ims = {}
        for image_id, image_path in paths.items():
            ims[image_id] = load_and_proccess_image(image_path)
        return ims

    # Read images from data/ folder
    def extract_paths(dir):
        paths = {}
        for filename in os.listdir(dir):
            if filename.endswith('.png'):
                image_id = int(filename[:-4])
                paths[image_id] = os.path.join(dir, filename)
        return paths

    test_ims  = read_images(extract_paths('./website/test_images'))

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(test_qs)

    # We add one because the Keras Tokenizer reserves index 0 and never uses it.
    # vocab_size = len(tokenizer.word_index) + 1

    #print('\n--- Converting questions to bags of words...')
    test_X_seqs = tokenizer.texts_to_matrix(test_qs)

    #print('\n--- Creating model input images...')
    test_X_ims = np.array([test_ims[id] for id in test_image_ids])
    
    #storing all the answers in an array form
    ans=np.array(all_answers)

    #storing all the question indexes where that particular image has been found
    ind=np.where(np.array(test_image_ids)==int(image[:-4]))
    
    #getting the final image and resizing it for making it compatible
    final_testing_image = test_X_ims[ind[0][0]]
    final_test_im = final_testing_image.reshape((1,64,64,3))
    
    #core prediction part starts here
    # print("If you want to ask a question from a pre-defined set of questions, enter 1")
    # print("If you want to ask a question on your own, enter 2")
    # choice = int(input("Enter your value here : "))
    


    if choice == 1:
        #showing all possible questions related to the randomly generated image
        # print("Pre-defined questions related to this image are as follows :")
        # df=pd.DataFrame(np.array(test_qs)[ind],columns=["Questions"])
        # print(df)
        # df1=np.array(df)
        # for i in df1:
        #     print (*i)
        #num=int(input("Enter a question index : "))

        final_testing_qs = test_X_seqs[ind[0][0] + qs_index]
        final_test_qs = final_testing_qs.reshape((1,27))

    elif choice == 2:
        final_testing_qs = asked_qs     #input("Enter your question here : ")
        #print('\n--- Converting questions to bags of words...')
        final_test_qs = tokenizer.texts_to_matrix([final_testing_qs])
        
    else:
        return "WRONG CHOICE!! TRY AGAIN"


    # global model1
    # print("model loaded")
    model1=load_model('./website/model.h5')
    #final prediction to be done
    #final_test_qs=user_choice(choice)
    final_output = model1.predict([final_test_im, final_test_qs])
    max_outputs=np.sort(final_output[0])[-3:][::-1]
    indices=np.argsort(final_output[0])[-3:][::-1]
    final_answer=ans[[i for i in indices]]
    
    #plotting top 3 probable answers for the asked question
    plt.figure()
    top_outputs = sns.barplot(x=max_outputs*100,y=final_answer)
    print(np.array(top_outputs))
    
    #can be implemented using a loop as per number of outputs
    top_outputs.annotate(round(max_outputs[0]*100,2),xy=(max_outputs[0]*100-10,0))
    top_outputs.annotate(round(max_outputs[1]*100,2),xy=(max_outputs[1]*100,1))
    top_outputs.annotate(round(max_outputs[2]*100,2),xy=(max_outputs[2]*100,2))
    
    plt.title("Predicted top 3 answers with confidence")
    plt.xlabel("Confidence (in %)", size=14)
    # plt.ylabel("Probable Answers", size=14)
    
    output_path="./website/static/output.png"
    plt.savefig(output_path)
    # cv2.imwrite('./website/static/random.png',np.asarray(top_outputs))
    # print("Plot saved")
    plt.clf()
    return output_path


def main():
    im_path,im_name,qs_list = random_image()
    for i in qs_list:
        print(i)
    choice=int(input("Enter 1 or 2 : "))
    qs_index=int(input(f"Enter between 0 and {len(qs_list)-1} : "))
    asked_qs=input("Enter a question : ")
    prediction=predict(im_name,choice,qs_index,asked_qs)
    print(prediction)

if __name__ == '__main__':
    main()