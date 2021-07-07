from flask import Blueprint,render_template,request
from .model import random_image,predict

views = Blueprint('views',__name__)


test_image_path=0
random_filename=0
qs_list=0
first_view=True
@views.route('/')
def home():
    first_view=True
    global test_image_path,random_filename,qs_list
    test_image_path,random_filename,qs_list = random_image()
    return render_template('index.html',qs_list=qs_list,first_view=first_view,asked_qs="")

@views.route('/predict', methods=['GET','POST'])
def prediction():
    if request.method == 'POST':
        global test_image_path,random_filename,qs_list
        print(test_image_path,random_filename,*qs_list)
        print("===================================================")
        choice=int(request.form['user_choice'])
        qs_index=int(request.form['qs_index'])-1
        asked_qs=request.form['asked_qs']
        print(choice)
        print(qs_index)
        print(asked_qs)
        print("Predict called")
        answer=predict(random_filename,choice,qs_index,asked_qs)
        print(f"Answer received {answer}")
        first_view=False
        return render_template('index.html',qs_list=qs_list,first_view=first_view,asked_qs=asked_qs)
    
    else:
        return "Khela hobe"