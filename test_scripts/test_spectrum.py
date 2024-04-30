import numpy as np
import matplotlib.pyplot as plotter

# How many time points are needed i,e., Sampling Frequency
# number of samples per period
samplingFrequency   = 100;

# At what intervals time points are sampled
samplingInterval       = 1 / samplingFrequency;
beginTime           = 0;
endTime             = 10; 

# Frequency of the signals
signal1Frequency     = 1;
signal2Frequency     = 7;

time        = np.arange(beginTime, endTime, samplingInterval);

amplitude1 = np.sin(2*np.pi*signal1Frequency*time)
amplitude2 = np.sin(2*np.pi*signal2Frequency*time)

# Create subplot

figure, axis = plotter.subplots(2, 1)
plotter.subplots_adjust(hspace=1)

# Add the sine waves
amplitude = amplitude1 + amplitude2

# Time domain representation of the resultant sine wave
axis[0].set_title('Sine wave with multiple frequencies')
axis[0].plot(time, amplitude)
axis[0].set_xlabel('Time')
axis[0].set_ylabel('Amplitude')
 

# Frequency domain representation
fourierTransform = np.fft.fft(amplitude)/len(amplitude)           # Normalize amplitude
fourierTransform = fourierTransform[range(int(len(amplitude)/2))] # Exclude sampling frequency
print(int(len(amplitude)/2))

tpCount     = len(amplitude)
values      = np.arange(int(tpCount/2)) # only want the positive values
timePeriod  = tpCount/samplingFrequency # how many periods are there, total samples / # samples per period (aka samplingFrequency)
frequencies = values/timePeriod

print(tpCount, samplingFrequency, timePeriod)
coeffs = 2*abs(fourierTransform)
print(frequencies[10],frequencies[70])
print(coeffs[10], coeffs[70])
# Frequency domain representation
axis[1].set_title('Fourier transform depicting the frequency components')
axis[1].plot(frequencies,coeffs)
axis[1].set_xlabel('Frequency')
axis[1].set_ylabel('Amplitude')

 
plotter.savefig("./test.png")
plotter.close()

