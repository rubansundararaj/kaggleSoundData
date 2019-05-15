import numpy as np 
import matplotlib.pyplot as plt 
from scipy import fftpack
from scipy.io import wavfile
import wave
import sys

f = 10  # Frequency, in cycles per second, or Hertz
f_s = 100  # Sampling rate, or number of measurements per second

t = np.linspace(0, 2, 2 * f_s, endpoint=False)

#s = np.sin(40 * 2 * np.pi * t) + 0.5 * np.sin(90 * 2 * np.pi * t)+ 1.75 * np.sin(90 * 2 * np.pi * t)
#fft = np.fft.fft(s)
#fft = fftpack.fft(s)

#freqs = fftpack.fftfreq(len(s)) * f_s
#freq = fftpack.fftffreq(len(x))*f_s
#print(freqs)




wav = wave.open('audio_test/00ae03f6.wav')
fname = 'audio_test/00ae03f6.wav'
rate, s = wavfile.read(fname)

#s = np.sin(40 * 2 * np.pi * t) + 0.5 * np.sin(90 * 2 * np.pi * t)+ 1.75 * np.sin(90 * 2 * np.pi * t)
freqs = fftpack.fftfreq(len(s)) 
fft = fftpack.fft(s)
#N = s.shape[0]
#L = N / rate

#print(fft)
print(len(freqs),'   ',len(np.abs(fft)))
plt.ylabel("Amplitude")
plt.xlabel("Frequency[Hz]")
plt.plot(freqs[:len(freqs)//2], np.abs(fft[:len(freqs)//2]))
''''
plt.plot(fft)
'''
plt.show()
