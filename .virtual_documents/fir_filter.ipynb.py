import numpy as np
import json
import matplotlib.pyplot as plt
import math

get_ipython().run_line_magic("matplotlib", " inline")


def create_sinc_kernel(k):
    taps = []
    start = 0.5 - (k / 2.0)
    for i in range(k):
        cur = start + i
        taps.append(math.sin(2 * math.pi * cur / k) / (2 * math.pi * cur * k))
        
    taps = np.array(taps)
    gain = taps.sum()

    taps = taps / gain
    
    return taps


x = np.arange(0, 128)
y = np.array(create_sinc_kernel(128))

plt.plot(x, y)
plt.show()


def fir_filter(samples, kernel):
    if len(kernel) > len(samples):
        gain = kernel[0:len(samples)].sum()
        kernel = kernel / gain

    acc = 0.0
    for i in range(len(samples)):
        sample = samples[i]
        acc += sample * kernel[i]
        
    return acc


def apply_filter(samples, blockSize, kernel):
    means = []
    for i in range(1, len(samples)+1):
        start = 0 if i < blockSize else i - blockSize
        sliced = samples[start:i]
        means.append(fir_filter(sliced, kernel))
        
    return means


blockSize = 32
kernel = create_sinc_kernel(blockSize)
x = np.arange(0, 128)
y = np.concatenate((np.zeros(64), np.ones(64)))
y = apply_filter(y_step_r, blockSize, kernel)

plt.plot(x, y)
plt.show()


def mean(samples):
    return np.array(samples).mean()

def moving_average(samples, blockSize):
    means = []
    for i in range(1, len(samples)+1):
        start = 0 if i < blockSize else i - blockSize
        sliced = samples[start:i]
        means.append(mean(sliced))
        
    return np.array(means)


moving_average(imp, 32)


imp = np.zeros(32)
imp[0] = 1

x = np.arange(0, 32)
y = moving_average(imp, 32)

plt.plot(x, y)
plt.show()


blockSize = 16

imp = np.zeros(blockSize)
imp[0] = 1

kernel =  np.array(moving_average(imp, blockSize))
x = np.arange(0, 128)
y = np.concatenate((np.zeros(64), np.ones(64)))
y = apply_filter(y, blockSize, kernel)
#y = moving_average(y, blockSize)

plt.plot(x, y)
plt.show()


np.flip(np.array([1, 2]))
