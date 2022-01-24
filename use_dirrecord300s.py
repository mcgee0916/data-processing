import pyaudio
import wave
import time
import math
import ctypes as ct
from datetime import datetime
from ftplib import FTP
from ftplib import error_perm
from posixpath import dirname
import os
import sys
import numpy as np
from scipy.io import wavfile
from scipy import signal
import soundfile as sf
import pandas as pd
import collections
import contextlib
import webrtcvad
import librosa
import scipy
import scipy.fftpack
from matplotlib import pyplot as plt
import weka.core.jvm as jvm
import weka.core.converters as converters
from weka.attribute_selection import ASSearch, ASEvaluation, AttributeSelection
from weka.classifiers import Classifier
import weka.plot.graph as graph
from weka.classifiers import PredictionOutput, KernelClassifier, Kernel
import weka.plot.classifiers as plcls
from weka.classifiers import Evaluation
from weka.core.classes import Random
from decimal import Decimal
import requests
import json

localtime = time.strftime('%Y%m%d_%H%M', time.localtime(time.time()))
txt_time = time.strftime('%Y-%m-%d  %H:%M', time.localtime(time.time()))
#print('localtime=' + localtime)
print(txt_time)
#ibp upload
url = 'http://ibp.bime.ntu.edu.tw/rest/sensorDataLogs/CH01/PH01/A_dir'
headers = {"Content-Type":"application/json"}

uploadtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))
y = time.strftime('%Y', time.localtime(time.time()))
m = time.strftime('%m', time.localtime(time.time()))
d = time.strftime('%d', time.localtime(time.time()))
HM = time.strftime('%H%M', time.localtime(time.time()))


file_y = os.getcwd() + '/' + y
file_m = file_y + '/' + m
file_d = file_m + '/' + d
file_HM = file_d + '/' + HM

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
#os.mkdir(spectrogram)

# WAVE_FILE = file_HM + '/original/' + 'dirchicken' + HM
# output = open(WAVE_FILE, 'w')

# output.write('localtime='+localtime)
# output.close()

CHUNK = 512
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 40
WAVE_OUTPUT_FILENAME = file_HM + '/original/' + 'dirchickenrecord' + localtime + '.wav'

os.close(sys.stderr.fileno())

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("Recording...")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("Recording Done")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()

# rename
# path = '/home/pi'
# oldname = '/home/pi/' + 'output.wav'
# newname = original + 'dirchickenrecord' + datetime.now().strftime("%Y%m%d_%H%M") + '.wav'
# os.rename(oldname,newname)

# upload
ftp = FTP("140.120.101.117")
ftp.login('PMML', 'enjoyresearch')
ftp.retrlines('LIST')
ftp.cwd('07_Experimental data/Changhua_Xianxi_poultryhouse/A_dir/20211231/original')
ftp.retrlines('LIST')
localfile = WAVE_OUTPUT_FILENAME
f = open(localfile, 'rb')
ftp.storbinary('STOR %s' % os.path.basename(localfile), f)

# BWfilter
sr, x = wavfile.read(WAVE_OUTPUT_FILENAME)
nyq = 0.5 * sr
#print(sr)
b, a = signal.butter(5, [500.0 / nyq, 5000.0 / nyq], btype='band')

x = signal.lfilter(b, a, x)
x = np.float32(x)
x /= np.max(np.abs(x))
#print(x)
WAVE_FILTER = file_HM + '/butterworthfilter/' + 'BW_' + localtime + '.wav'
#print(WAVE_FILTER)
wavfile.write(WAVE_FILTER, sr, x)

# pre audioread & audiowrite
#WAVE_FILTER = file_HM + '/butterworthfilter/' + 'BW_' + datetime.now().strftime("%Y%m%d_%H%M") + '.wav'
y, Fs = sf.read(WAVE_FILTER)
sf.write(WAVE_FILTER, y, Fs)
 
# upload filter file
ftp = FTP("140.120.101.117")
ftp.login('PMML', 'enjoyresearch')
ftp.retrlines('LIST')
ftp.cwd('07_Experimental data/Changhua_Xianxi_poultryhouse/A_dir/20211231/filter')
ftp.retrlines('LIST')
localfile = file_HM + '/butterworthfilter/' + 'BW_' + datetime.now().strftime("%Y%m%d_%H%M") + '.wav'
f = open(localfile, 'rb')
ftp.storbinary('STOR %s' % os.path.basename(localfile), f)


class FloatBits(ct.Structure):
    _fields_ = [
        ('M', ct.c_uint, 23),
        ('E', ct.c_uint, 8),
        ('S', ct.c_uint, 1)
    ]

class Float(ct.Union):
    _anonymous_ = ('bits',)
    _fields_ = [
        ('value', ct.c_float),
        ('bits', FloatBits)
    ]

def nextpow2(x):
    if x < 0:
        x = -x
    if x == 0:
        return 0
    d = Float()
    d.value = x
    if d.M == 0:
        return d.E - 127
    return d.E - 127 + 1


f = wave.open(WAVE_FILTER)
# (nchannels, sampwidth, framerate, nframes, comptype, compname)
params = f.getparams()
nchannels, sampwidth, framerate, nframes = params[:4]
fs = framerate
#print(fs)

str_data = f.readframes(nframes)
#print(len(str_data))
f.close()

x = np.frombuffer(str_data, dtype=np.short)
#print(x)

len_ = 20 * fs // 1000
PERC = 50
len1 = len_ * PERC // 100
len2 = len_ - len1

Thres = 3
Expnt = 2.0
beta = 0.002
G = 0.9

win = np.hamming(len_)
# normalization gain for overlap+add with 50% overlap
winGain = len2 / sum(win)
#print(winGain)

# Noise magnitude calculations - assuming that the first 5 frames is noise/silence
nFFT = 2 * 2 ** (nextpow2(len_))
noise_mean = np.zeros(nFFT)

j = 0
for k in range(1, 4):
    noise_mean = noise_mean + abs(np.fft.fft(win * x[j:j + len_], nFFT))
    j = j + len_
noise_mu = noise_mean / 3

# --- allocate memory and initialize various variables
k = 1
img = 1j
x_old = np.zeros(len1)
Nframes = len(x) // len2 - 1
#print(Nframes)
#print(len(x))
xfinal = np.zeros(Nframes * len2)
#print(xfinal)

# =========================    Start Processing   ===============================
for n in range(0, Nframes):
    #print(n)
    # Windowing
    x = np.frombuffer(str_data, dtype=np.short)
    insign = win * x[k - 1:k + len_ - 1]
    # compute fourier transform of a frame
    spec = np.fft.fft(insign, nFFT)
    # compute the magnitude
    sig = abs(spec)
    # save the noisy phase information
    theta = np.angle(spec)
    SNRseg = 10 * np.log10(np.linalg.norm(sig, 2) ** 2 / np.linalg.norm(noise_mu, 2) ** 2)
    #print(SNRseg)
    
    def berouti(SNR):
        if -5.0 <= SNR <= 20.0:
            a = 4 - SNR * 3 / 20
        else:
            if SNR < -5.0:
                a = 5
            if SNR > 20:
                a = 1
        return a

    def berouti1(SNR):
        if -5.0 <= SNR <= 20.0:
            a = 3 - SNR * 2 / 20
        else:
            if SNR < -5.0:
                a = 4
            if SNR > 20:
                a = 1
        return a

        
    if Expnt == 1.0:  
        alpha = berouti1(SNRseg)
    else:  
        alpha = berouti(SNRseg)
                
    #############
    sub_speech = sig ** Expnt - alpha * noise_mu ** Expnt   
    diffw = sub_speech - beta * noise_mu ** Expnt
            
    # beta negative components 
    def find_index(x_list):
        index_list = []
        for i in range(len(x_list)):
            if x_list[i] < 0:
                index_list.append(i)
        return index_list

    z = find_index(diffw)
    #print(z)
            
    if len(z) > 0:
        sub_speech[z] = beta * noise_mu[z] ** Expnt
        # --- implement a simple VAD detector --------------
        if SNRseg < Thres:  # Update noise spectrum
            noise_temp = G * noise_mu ** Expnt + (1 - G) * sig ** Expnt  
            noise_mu = noise_temp ** (1 / Expnt)  
        sub_speech[nFFT // 2 + 1:nFFT] = np.flipud(sub_speech[1:nFFT // 2])
        x_phase = (sub_speech ** (1 / Expnt)) * (np.array([math.cos(x) for x in theta]) + img * (np.array([math.sin(x) for x in theta])))
        # take the IFFT
        xi = np.fft.ifft(x_phase).real
        # --- Overlap and add ---------------
        xfinal[k - 1:k + len2 -1] = x_old + xi[0:len1]
        #print(k+len2-1)
        x_old = xi[0 + len1:len_]
        k = k + len2

print('Spectral subtraction Done')
WAVE_SS = file_HM + '/spectral_subtraction/' + 'SS_' + localtime + '.wav'
wf = wave.open(WAVE_SS, 'wb')
wf.setparams(params)
#print(params)
wave_data = (winGain * xfinal).astype(np.short)
#print(wave_data)
wf.writeframes(wave_data.tostring())
wf.close()

#resample 44.1 kHz to 16 kHz
src_sig, sr = sf.read(WAVE_SS)
dst_sig = librosa.resample(src_sig, sr, 16000)
sf.write(WAVE_SS, dst_sig, 16000)

# pre audioread & audiowrite
#y, Fs = sf.read(WAVE_SS)
#sf.write(WAVE_SS, y, Fs)

#voice activity detection

def read_wave(path):
    """Reads a .wav file.
    Takes the path, and returns (PCM audio data, sample rate).
    """
    with contextlib.closing(wave.open(path, 'rb')) as wf:
        num_channels = wf.getnchannels()
        assert num_channels == 1
        sample_width = wf.getsampwidth()
        assert sample_width == 2
        sample_rate = wf.getframerate()
        assert sample_rate in (8000, 16000, 32000)
        pcm_data = wf.readframes(wf.getnframes())
        return pcm_data, sample_rate


def write_wave(path, audio, sample_rate):
    """Writes a .wav file.
    Takes path, PCM audio data, and sample rate.
    """
    with contextlib.closing(wave.open(path, 'wb')) as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio)


class Frame(object):
    """Represents a "frame" of audio data."""

    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration


def frame_generator(frame_duration_ms, audio, sample_rate):
    """Generates audio frames from PCM audio data.
    Takes the desired frame duration in milliseconds, the PCM data, and
    the sample rate.
    Yields Frames of the requested duration.
    """
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n


def vad_collector(sample_rate, frame_duration_ms,
                  padding_duration_ms, vad, frames):
    """Filters out non-voiced audio frames.
    Given a webrtcvad.Vad and a source of audio frames, yields only
    the voiced audio.
    Uses a padded, sliding window algorithm over the audio frames.
    When more than 90% of the frames in the window are voiced (as
    reported by the VAD), the collector triggers and begins yielding
    audio frames. Then the collector waits until 90% of the frames in
    the window are unvoiced to detrigger.
    The window is padded at the front and back to provide a small
    amount of silence or the beginnings/endings of speech around the
    voiced frames.
    Arguments:
    sample_rate - The audio sample rate, in Hz.
    frame_duration_ms - The frame duration in milliseconds.
    padding_duration_ms - The amount to pad the window, in milliseconds.
    vad - An instance of webrtcvad.Vad.
    frames - a source of audio frames (sequence or generator).
    Returns: A generator that yields PCM audio data.
    """
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    # We use a deque for our sliding window/ring buffer.
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    # We have two states: TRIGGERED and NOTTRIGGERED. We start in the
    # NOTTRIGGERED state.
    triggered = False

    voiced_frames = []
    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        sys.stdout.write('1' if is_speech else '0')
        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            # If we're NOTTRIGGERED and more than 90% of the frames in
            # the ring buffer are voiced frames, then enter the
            # TRIGGERED state.
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                sys.stdout.write('+(%s)' % (ring_buffer[0][0].timestamp,))
                # We want to yield all the audio we see from now until
                # we are NOTTRIGGERED, but we have to start with the
                # audio that's already in the ring buffer.
                for f, s in ring_buffer:
                    voiced_frames.append(f)
                ring_buffer.clear()
        else:
            # We're in the TRIGGERED state, so collect the audio data
            # and add it to the ring buffer.
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            # If more than 90% of the frames in the ring buffer are
            # unvoiced, then enter NOTTRIGGERED and yield whatever
            # audio we've collected.
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
                triggered = False
                yield b''.join([f.bytes for f in voiced_frames])
                ring_buffer.clear()
                voiced_frames = []
    if triggered:
        sys.stdout.write('-(%s)' % (frame.timestamp + frame.duration))
    sys.stdout.write('\n')
    # If we have any leftover voiced audio when we run out of input,
    # yield it.
    if voiced_frames:
        yield b''.join([f.bytes for f in voiced_frames])

def main(args):
    path = file_HM + '/spectral_subtraction'
    # path = r"D:\workspace\test wer\ali_test_set\wav_total"
    files = os.listdir(path)
    files = [path + "/" + f for f in files if f.endswith('.wav')]
    for i in range(len(files)):

        #print(files[i])
        args = [3, files[i]]
        #print(args)
        if len(args) != 2:
            sys.stderr.write(
                'Usage: example.py <aggressiveness> <path to wav file>\n')
            sys.exit(1)
        audio, sample_rate = read_wave(args[1])
        vad = webrtcvad.Vad(int(args[0]))
        frames = frame_generator(10, audio, sample_rate)
        frames = list(frames)
        segments = vad_collector(sample_rate, 10, 100, vad, frames)
        for j, segment in enumerate(segments):
            path = file_HM + '/vad/' + files[i][-20:-4] + 'chunk-%002d.wav' % (j + 1,)
            print(path)
            print(' Writing %s' % (path,))
            write_wave(path, segment, sample_rate)

if __name__ == '__main__':
    main(sys.argv[1:])

#feature extraction
input_path = file_HM + '/vad'
output_path = file_HM + '/feature'
totalfeature_path = totalfeature_of_the_day
audio_list = os.listdir(input_path)
#print(input_path)
#print(output_path)

for test in audio_list:
    if test[-4:]=='.wav':
        this_path_input = os.path.join(input_path, test)
        this_path_output = os.path.join(output_path, test[:-4]+'.arff')
        cmd = 'cd /home/pi/opensmile/build/progsrc/smilextract && ./SMILExtract -C /home/pi/opensmile/config/is09-13/IS09_emotion.conf -I ' + this_path_input + ' -O ' + this_path_output
    os.system(cmd)

for testt in audio_list:
    if testt[-4:]=='.wav':
        this_path_input = os.path.join(input_path, testt)
        this_path_output = os.path.join(totalfeature_path, testt[:-4]+'.arff')
        cmd = 'cd /home/pi/opensmile/build/progsrc/smilextract && ./SMILExtract -C /home/pi/opensmile/config/is09-13/IS09_emotion.conf -I ' + this_path_input + ' -O ' + this_path_output
    os.system(cmd)
#plot spectrogram
#path = file_HM + '/vad'
#files = os.listdir(path)
#files = [path + "/" + f for f in files if f.endswith('.wav')]
#print(files)

#def SetFileName(WavFileName):
#    for i in range(len(files)):
#        FileName = files[i]
#        print('a')
#        print("SetFileName File Name is ", FileName)
#        FileName = WavFileName;

#def batchspec():
#for i in range(len(files)):
#        FileName = files[i]
#        #print(files[i])
#        f = wave.open(r"" + FileName, "rb")
#        params = f.getparams()
        #print(params)
#        nchannels, sampwidth, framerate, nframes = params[:4]
#        fs_rate, signal = wavfile.read(r""+FileName,"rb")
#        l_audio = len(signal.shape)
#        if l_audio == 2:
#            signal = signal.sum(axis=1) / 2

#        signal2 = signal + 480000000
#        N = signal2.shape[0]
#        print(N)
#        secs = N / float(fs_rate)
#        print(secs)
#        Ts = 1.0 / fs_rate  # sampling interval in time
        # print ("Timestep between samples Ts", Ts)
#        t = scipy.arange(0, secs, Ts)  # time vector as scipy arange field / numpy.ndarray

#        n = len(t)
        #print(n)

#        FFT20 = abs(scipy.fft(signal2[0:320000]))/n

#        FFT_side20 = FFT20[range(int(N / 2))]

#        freqs20 = scipy.fftpack.fftfreq(signal2[0:320000].size, t[1] - t[0])

#        freqs_side20 = freqs20[range(int(N / 2))]

#        abs(FFT_side20)

#        for a in range(60):
#            FFT_side20[a] = 0

#        plt.subplot(211)
#        p1 = plt.plot(t, signal2, "r")  # plotting the signal
#        plt.xlabel('Time')
#        plt.ylabel('Amplitude')

#        plt.subplot(212)
#        p3 = plt.plot(freqs_side20, FFT_side20, "b")  # plotting the positive fft spectrum
#        plt.xlabel('Frequency (Hz)')
#        plt.ylabel('Amplitude')

#        StepTotalNum = 0;
#        haha = 0
#        while StepTotalNum < nframes:

            # for j in range(int(Cutnum)):
#            print("Stemp=%d" % (haha))
#            FileName = files[i][29:-4] + ".png"
            #print(FileName)

#            haha = haha + 1
#            StepTotalNum = haha * nframes
#            os.chdir(spectrogram)

            #plt.get_current_fig_manager().window.state('zoomed')
#            fig = plt.gcf()
#            plt.pause(0.01)
#            fig.savefig(FileName, bbox_inches='tight')
#plt.close()

#if __name__ == '__main__':
#    batchspec()

csv_path = feature
csv_list = os.listdir(csv_path)
features_list = []
for file in csv_list:  
    if file[-5:] == '.arff':
        file_path = os.path.join(csv_path, file)
       
        f = open(file_path)
       
        last_line = f.read().splitlines()[-1]
        f.close()
        features = last_line.split(',')
        
        features = features[1:]
        features_list.append(features)
        data_m = pd.DataFrame(features_list)
        data_m.to_csv(os.path.join(totalfeature_of_this_moment, 'totaldata_this_moment.arff'), sep=',', header=False,
                      index=False)
        print('Feature file combine Over')

#combine the attribute file & data file
atrribute_path = '/home/pi/weka/attribute_format/attribute.arff'
total_path = totalfeature_of_this_moment + '/totaldata_this_moment.arff'

f = open(atrribute_path)
attribute = f.read().splitlines()
#print(attribute)
f.close()

f = open(total_path)
total = f.read().splitlines()
#print(total)
f.close()

attribute = pd.DataFrame(attribute)
#print(attribute)
total = pd.DataFrame(total)
#print(total)
data = pd.concat([attribute,total])

data.to_csv(unknown_data + '/total_unknown_data.arff', index=False, header=False, quotechar=' ')

#weka process
#data path
jvm.start()
data_dir = '/home/pi/weka/training_data'

#load data
data = converters.load_any_file(data_dir + '/Xinxi_full_feature.arff')
data.class_is_last()

#print(data)

#attribute selection
search = ASSearch(classname="weka.attributeSelection.BestFirst", options=["-D", "1", "-N", "5"])
evaluator = ASEvaluation(classname="weka.attributeSelection.CfsSubsetEval", options=["-P", "1", "-E", "1"])
attsel = AttributeSelection()
attsel.search(search)
attsel.evaluator(evaluator)
attsel.select_attributes(data)
#print('a')

#print("# attributes: " + str(attsel.number_attributes_selected))
#print("attributes: " + str(attsel.selected_attributes))
#print("result string:\n" + attsel.results_string)

with open(attribute_selection + "/select_attribute_result.txt", "w") as f:
    f.write("# attributes: " + str(attsel.number_attributes_selected))
    f.write("attributes: " + str(attsel.selected_attributes))
    f.write("result string:\n" + attsel.results_string)

#build classifier & save the model
cls = Classifier(classname="weka.classifiers.functions.SMO")
cls.build_classifier(data)
#print(cls.options)
#print(cls)

cls.serialize(ML_model + '/Machine_Learning_SMO.model')
#graph.plot_dot_graph(cls.graph)

#cross-validate
cls = KernelClassifier(classname="weka.classifiers.functions.SMO")
kernel = Kernel(classname="weka.classifiers.functions.supportVector.PolyKernel")
cls.kernel = kernel
pout = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.PlainText")
evl = Evaluation(data)
evl.crossvalidate_model(cls, data, 10, Random(1), pout)
#print('b')
#print(evl.summary())
#print(pout.buffer_content())

with open(cross_validation + "/cross_validation_result.txt", "w") as f:
    f.write(evl.summary())
    f.write(pout.buffer_content())
    
#test data
test_data = converters.load_any_file(unknown_data + '/total_unknown_data.arff')
test_data.class_is_last()
#print(test_data)

cls = Classifier(classname="weka.classifiers.functions.SMO")
cls.build_classifier(data)
#deserialize = cls.deserialize('/home/pi/test.model')
testout = PredictionOutput(classname="weka.classifiers.evaluation.output.prediction.PlainText")
test_evl = Evaluation(test_data)
test_evl.test_model(cls, test_data, testout)
#print(test_evl.summary())
#print(testout.buffer_content())
#print('c')

with open(predict_result + "/"+ localtime + "_predict_result" +  ".txt", "w") as f:
    f.write(test_evl.summary())
    f.write(testout.buffer_content())

#plcls.plot_classifier_errors(evl.predictions, wait=True)

#stop
jvm.stop()


f = open(predict_result + "/"+ localtime + "_predict_result" +  ".txt", 'r')
lines = f.read().splitlines()
#print(lines)
count = {'1:health':0, '2:sick':0, '3:notsound':0}
for line in lines:
    line1 = line.split(' ')
    for word in line1:
        if word in count:
            count[word] += 1
count = sorted(count.items(), key=None, reverse=False)
print(count)
print(count[0][-1])
print(count[1][-1])
print(count[2][-1])

health_number = count[0][-1]
sick_number = count[1][-1]
notsound_number = count[2][-1]
total_number_this_moment = health_number + sick_number + notsound_number
print(total_number_this_moment)

health = str(count[0][-1])
sick = str(count[1][-1])
notsound = str(count[2][-1])
str_total_number_this_moment = str(total_number_this_moment)
print(str_total_number_this_moment)

f = open(predict_result + "/" + "data" +  ".txt", 'r')
last_health = f.readline()
last_sick = f.readline()
last_notsound = f.readline()
last_health_number = int(last_health)
last_sick_number = int(last_sick)
last_notsound_number = int(last_notsound)
last_total_file = int(last_health_number + last_sick_number + last_notsound_number)
print(last_health_number)
print(last_sick_number)
print(last_notsound_number)
print(last_total_file)

total_file_number = last_total_file + total_number_this_moment
print(total_file_number)
str_total_file_number = str(total_file_number)

total_health_number = health_number + last_health_number
print(total_health_number)
str_total_health_number = str(total_health_number)

total_sick_number = sick_number + last_sick_number
print(total_sick_number)
str_total_sick_number = str(total_sick_number)

total_notsound_number = notsound_number + last_notsound_number
print(total_notsound_number)
str_total_notsound_number = str(total_notsound_number)

f = open(predict_result + "/" + "data" +  ".txt", 'w')
f.write(str_total_health_number + '\n')
f.write(str_total_sick_number + '\n')
f.write(str_total_notsound_number + '\n')
f.close

health_percent = float(total_health_number)/total_file_number
print(health_percent)
sick_percent = float(total_sick_number)/total_file_number
print(sick_percent)
notsound_percent = float(total_notsound_number)/total_file_number
print(notsound_percent)

rounding_health_percent = np.around(health_percent,3)
print(rounding_health_percent)
rounding_sick_percent = np.around(sick_percent,3)
print(rounding_sick_percent)
rounding_notsound_percent = np.around(notsound_percent,3)
print(rounding_notsound_percent)
#print(health_percent)
health_percent = rounding_health_percent*100.0
sick_percent = rounding_sick_percent*100.0
notsound_percent = rounding_notsound_percent*100.0


def rrr(self):
    pos_point = self.find('.')
    if pos_point != -1:
        for i in ['0000000000','9999999999']:
            pos_loop = self.find(i, pos_point)
            if pos_loop != -1:
                self = str(round(float(self), pos_loop))
                break
        if float(self) == round(float(self)):
            self = str(round(float(self)))
        return self
    else:
        return self

str_health_percent = str(health_percent)

str_sick_percent = str(sick_percent)

str_notsound_percent = str(notsound_percent)
renew_health = rrr(str_health_percent)
renew_sick = rrr(str_sick_percent)
renew_notsound = rrr(str_notsound_percent)
print(str(health_percent)+'%')
print(str(sick_percent)+'%')
print(str(notsound_percent)+'%')

number_of_vad_files = len([f for f in os.listdir(vad) if os.path.isfile(os.path.join(vad, f))])
print(number_of_vad_files)
number_of_vad_files = str(number_of_vad_files)

f = open(predict_result + "/"+ localtime + "_predict_result" +  ".txt", 'a')
f.write(' ' + '\n')
f.write('---------' + txt_time + '---------' + '\n')
f.write(' ' + '\n')
f.write('        Number of VAD files: ' + number_of_vad_files + '\n')
f.write('        Number of HEALTH this time: ' + health + '\n')
f.write('        Number of SICK this time: ' + sick + '\n')
f.write('        Number of NOTSOUND this time: ' + notsound + '\n')
f.write(' ' + '\n')
f.write('-----------Predict Result-----------' + '\n')
f.write(' ' + '\n')
f.write('        Total files: ' + str_total_file_number + '\n')
f.write('        Total number of HEALTH files: ' + str_total_health_number + '\n')
f.write('        Total number of SICK files: ' + str_total_sick_number + '\n')
f.write('        Total number of NOTSOUND files: ' + str_total_notsound_number + '\n')
f.write('        Percentage of HEALTH: ' + renew_health + '%' + '\n')
f.write('        Percentage of SICK: ' + renew_sick + '%' + '\n')
f.write('        Percentage of NOTSOUND: ' + renew_notsound + '%' + '\n')
f.close()

#upload predict result
result = datetime.now().strftime("%Y%m%d")
#print(result)
ftp = FTP("140.120.101.117")
ftp.login('PMML', 'enjoyresearch')
ftp.retrlines('LIST')
ftp.cwd('07_Experimental data/Changhua_Xianxi_poultryhouse/A_dir/20211231')

def upload_dir(ftp, path, first_call=True):
    try:
        ftp.cwd(path)
    except error_perm:
        upload_dir(ftp, dirname(path), False)
        ftp.mkd(path)
        if first_call:
            ftp.cwd(path)
path = result
upload_dir(ftp, path)

f = open(predict_result + "/"+ localtime + "_predict_result" +  ".txt", 'rb')
ftp.storbinary('STOR %s' % os.path.basename(predict_result + "/"+ localtime + "_predict_result" +  ".txt"), f)
print("___________________________")
#upload to IBP

query = {
  "vad": number_of_vad_files,
  "health": health,
  "sick": sick,
  "notsound": notsound,
  "totalvad": str_total_file_number,
  "totalhealth": str_total_health_number,
  "totalsick": str_total_sick_number,
  "totalnotsound": str_total_notsound_number,
  "frachealth": renew_health,
  "fracsick": renew_sick,
  "fracnotsound": renew_notsound,
  "dataTime": uploadtime,
  "_overwrite_": True,
  "_token_": "b8dd0f13-3918-4084-9087-8795de88ada0"}
print(query)
query_j = json.dumps(query)
print(query_j)

try:
    res = requests.post(url,data=query_j,headers=headers)
    print("successful upload to IBP")
except:
    print("Failed to upload to IBP")
print("Done")
