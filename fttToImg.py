import numpy as np 
import matplotlib.pyplot as plt 
from scipy import fftpack
from scipy.io import wavfile
import wave
import sys
import os
import pandas as pd
from PIL import Image as image
data_test = pd.read_csv('./meta/train.csv', header = None)
test = data_test.values[1:]

def labeExtraction(test):
    test_five = []
    for t in test:
        if(t[1] == 'Acoustic_guitar'):
            #Acoustic_guitar.append(1)
            test_five.append(t)
        elif(t[1] == 'Applause'):
            #Applause.append(1)
            test_five.append(t)
        elif(t[1] == 'Bark'):
            #Bark.append(1)
            test_five.append(t)
        elif(t[1] == 'Cough'):
            #Cough.append(1)
            test_five.append(t)
        elif(t[1] == 'Flute'):
            #Flute.append(1)
            test_five.append(t)
        else:
            pass
    return test_five

def FFTSaveImg(afile,type_):
    fname = './audio_test/' + str(afile)
    rate, s = wavfile.read(fname)
    imgFile = afile[:8]
    plt.clf()
    freqs = fftpack.fftfreq(len(s)) 
    fft = fftpack.fft(s)
    plt.axis('off')
    plt.plot(freqs[:len(freqs)//2], np.abs(fft[:len(freqs)//2]))
    plt.savefig("./testfft/"+"/"+imgFile+".png", bbox_inches="tight", pad_inches=0)
    plt.clf()

def compressImg(afile,type_):
    imgFile = afile[:8]
    im1 = image.open("./fft/"+type_+"/"+imgFile+".png")
    width = 100
    height = 100
    im2 = im1.resize((width, height), image.NEAREST)      # use nearest neighbour
    ext = ".png"
    im2.save("./compressedfft/"+type_+"/"+imgFile + ext)


extract = labeExtraction(test)
x = 1
for e in extract:
    print('total: ',len(extract))
    print('Current: ',x)
    afile = e[0]
    #FFTSaveImg(afile,'test')
    compressImg(afile,'train')
    x = x + 1
