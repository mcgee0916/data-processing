from upload import setting
import os


if __name__ == '__main__':
    pass

#ibp upload

class Nest():
    def __init__(self):




def nested(time):
    file_y = os.getcwd() + '/' + time.y
    file_m = file_y + '/' + time.m
    file_d = file_m + '/' + time.d
    file_HM = file_d + '/' + time.HM

if not os.path.exists(file_y):
    os.mkdir(file_y)
    os.mkdir(file_m)
    os.mkdir(file_d)
    os.mkdir(file_HM)
else:
    if not os.path.exists(file_m):
        os.mkdir(file_m)
        os.mkdir(file_d)
        os.mkdir(file_HM)
    else:
        if not os.path.exists(file_d):
            os.mkdir(file_d)
            os.mkdir(file_HM)
        else:
            if not os.path.exists(file_HM):
                os.mkdir(file_HM)

totalfeature_of_the_day = file_d + '/totalfeature_of_the_day'
total = totalfeature_of_the_day + '/total'
ML_model = file_d + '/ML_model'
attribute_selection = ML_model + '/attribute_selection'
cross_validation = ML_model + '/cross_validation'
unknown_data = ML_model + '/unknown_data'
predict_result = file_d + '/predict_result'
original = file_HM + '/original'
butterworthfilter = file_HM + '/butterworthfilter'
spectral_subtraction= file_HM + '/spectral_subtraction'
vad = file_HM + '/vad'
feature = file_HM + '/feature'
spectrogram = file_HM + '/spectrogram'
totalfeature_of_this_moment = file_HM + '/feature/' + 'totalfeature_of_this_moment'


def mkdir_day(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
    else:
        print('totalfeature_of_the_day'+' exist')

path = totalfeature_of_the_day
mkdir_day(path)

def mkdir_total(total_path):
    folder = os.path.exists(total_path)
    if not folder:
        os.makedirs(total_path)
    else:
        print('total'+' exist')

total_path = total
mkdir_total(total_path)

def mkdir_model(model_path):
    folder = os.path.exists(model_path)
    if not folder:
        os.makedirs(model_path)
    else:
        print('model'+' exist')

model_path = ML_model
mkdir_model(model_path)

def mkdir_AS(AS_path):
    folder = os.path.exists(AS_path)
    if not folder:
        os.makedirs(AS_path)
    else:
        print('AS'+' exist')

AS_path = attribute_selection
mkdir_AS(AS_path)

def mkdir_CV(CV_path):
    folder = os.path.exists(CV_path)
    if not folder:
        os.makedirs(CV_path)
    else:
        print('CV'+' exist')

CV_path = cross_validation
mkdir_CV(CV_path)

def mkdir_unknown(unknown_path):
    folder = os.path.exists(unknown_path)
    if not folder:
        os.makedirs(unknown_path)
    else:
        print('unknown'+' exist')

unknown_path = unknown_data
mkdir_unknown(unknown_path)

def mkdir_predict(predict_path):
    folder = os.path.exists(predict_path)
    if not folder:
        os.makedirs(predict_path)
    else:
        print('predict'+' exist')

predict_path = predict_result
mkdir_predict(predict_path)

def put_data(predict_path):
    txt_file = os.path.isfile(predict_path)
    if not txt_file:
        f = open(predict_result + "/"+ "data" +  ".txt", 'a')
        f.write('0' + '\n')
        f.write('0' + '\n')
        f.close
    else:
        print('data.txt' + 'exist')

predict_path = predict_result
put_data(predict_path)

os.mkdir(original)
os.mkdir(butterworthfilter)
os.mkdir(spectral_subtraction)
os.mkdir(vad)
os.mkdir(feature)
os.mkdir(totalfeature_of_this_moment)