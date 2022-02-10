import upload
import pyaudio
import wave
import os
import sys
import numpy as np
from scipy.io import wavfile
from scipy import signal
timenow = upload.upload()
print(timenow.uploadtime)

url1 = 'http://ibp.bime.ntu.edu.tw/rest/sensorDataLogs/CH01/PH01/A_dir'
url2 = 'http://ibp.bime.ntu.edu.tw/rest/sensorDataLogs/CH01/PH01/A_dir'
headers = {"Content-Type":"application/json"}

uploadtime = timenow.uploadtime
file_y = os.getcwd() + '/' + timenow.y
file_m = file_y + '/' + timenow.m
file_d = file_m + '/' + timenow.d
file_HM = file_d + '/' + timenow.HM

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

CHUNK = 512
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 300
WAVE_OUTPUT_FILENAME = file_HM + '/original/' + 'dirchickenrecord' + timenow.uploadtime + '.wav'

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
localfile = WAVE_OUTPUT_FILENAME
upload.NASuploader(url1,localfile)
# BWfilter
sr, x = wavfile.read(WAVE_OUTPUT_FILENAME)
nyq = 0.5 * sr
b, a = signal.butter(5, [500.0 / nyq, 5000.0 / nyq], btype='band')
x = signal.lfilter(b, a, x)
x = np.float32(x)
x /= np.max(np.abs(x))
WAVE_FILTER = file_HM + '/butterworthfilter/' + 'BW_' + timenow.uploadtime + '.wav'
wavfile.write(WAVE_FILTER, sr, x)
upload.NASuploader(url2,WAVE_FILTER)