import numpy as np
import matplotlib.pyplot as  plotter

# How many time points are needed i,e., Sampling Frequency
# number of samples per period
samplingFrequency   = 10;

# At what intervals time points are sampled
samplingInterval    = 1 / samplingFrequency;
beginTime           = 0;
endTime             = 1; 

[X, Y] = np.meshgrid(np.arange(10)/10, np.arange(10)/10)
S = np.sin(2*np.pi*X)# + np.cos(2*np.pi*Y) + np.sin(2*np.pi*X)*np.cos(2*np.pi*Y)
FS = 2 * np.fft.fft2(S)/100

coeffs = np.abs(FS[:5, :5])
mask = coeffs > 0.000001
print(coeffs[mask])

plotter.imshow(coeffs)

# plotter.imshow(np.abs(np.fft.fftshift(FS))**2,) 


# Create subplot

# figure, axis = plotter.subplots(2, 1)
# plotter.subplots_adjust(hspace=1)

# Add the sine waves

# Time domain representation of the resultant sine wave
# axis[0].set_title('Sine wave with multiple frequencies')
# axis[0].plot(time, amplitude)
# axis[0].set_xlabel('Time')
# axis[0].set_ylabel('Amplitude')
 
# Frequency domain representation
# axis[1].set_title('Fourier transform depicting the frequency components')
# axis[1].plot(frequencies,coeffs)
# axis[1].set_xlabel('Frequency')
# axis[1].set_ylabel('Amplitude')

 
plotter.savefig("./test_fft2.png")
plotter.close()

