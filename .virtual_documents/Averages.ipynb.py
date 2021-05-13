import numpy as np
import json
import matplotlib.pyplot as plt
import math

get_ipython().run_line_magic("matplotlib", " inline")


with open('./data.json', 'r') as read_file:
    data = json.load(read_file)
    data = np.array(data)


x = np.arange(0, len(data))
y = data


plt.figure(figsize=(15, 6))
plt.plot(x, y, color='#489cb5')
plt.show()


def mean(samples):
    return np.array(samples).mean()


def moving_average(samples, blockSize):
    means = []
    for i in range(1, len(samples)+1):
        start = 0 if i < blockSize else i - blockSize
        sliced = samples[start:i]
        means.append(mean(sliced))
        
    return means


x = np.arange(0, len(data))
y_av32 = np.array(moving_average(data, 32))
y_av64 = np.array(moving_average(data, 64))
y_av128 = np.array(moving_average(data, 128))


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30, 6))

axes[0].plot(x, y, alpha=0.3)
axes[0].plot(x, y_av32, color='#e0743a')
axes[1].plot(x, y, alpha=0.3)
axes[1].plot(x, y_av64, color='#e0743a')
axes[2].plot(x, y, alpha=0.3)
axes[2].plot(x, y_av128, color='#e0743a')

plt.show()


y_av128 = np.array(moving_average(y, 128))

plt.figure(figsize=(15, 6))
plt.plot(x, y, color='#489cb5', alpha=0.3)
plt.plot(x, y_av128, color='#e0743a')
plt.show()


def create_filter_kernel(k, w = None):
    w = k if w == None else w
    taps = []
    start = 0.5 - (k / 2.0)
    for i in range(k):
        cur = start + i
        taps.append(math.sin(2 * math.pi * cur / w) / (2 * math.pi * cur * w))
        
    taps = np.array(taps)
    gain = taps.sum()

    taps = taps / gain
    
    return taps


## xs = np.arange(0, 128)
ys = np.array(create_filter_kernel(128, 128))

plt.figure(figsize=(8, 3))
plt.plot(xs, ys)
plt.show()


def fir_filter(samples, taps):
    if len(taps) > len(samples):
        gain = taps[0:len(samples)].sum()
        taps = taps / gain

    acc = 0.0
    for i in range(len(samples)):
        sample = samples[i]
        acc += sample * taps[i]
        
    return acc


# taps = create_filter_kernel(6)
# samples = [5, 10, 15, 20, 25, 30]
# fir_filter(samples, taps)


create_filter_kernel(6)


def move_filter(samples, blockSize, w = None):
    means = []
    taps = create_filter_kernel(blockSize, w)
    for i in range(1, len(samples)+1):
        start = 0 if i < blockSize else i - blockSize
        sliced = samples[start:i]
        means.append(fir_filter(sliced, taps))
        
    return means


x = np.arange(0, len(data))
y_fir32 = np.array(move_filter(data, 32))
y_fir64 = np.array(move_filter(data, 64))
y_fir128 = np.array(move_filter(data, 128))


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30, 6))

axes[0].plot(x, y, alpha=0.3)
axes[0].plot(x, y_fir32, color='#db5670')
axes[1].plot(x, y, alpha=0.3)
axes[1].plot(x, y_fir64, color='#db5670')
axes[2].plot(x, y, alpha=0.3)
axes[2].plot(x, y_fir128, color='#db5670')

plt.show()


plt.figure(figsize=(15, 6))

plt.plot(x, y, alpha=0.3)
plt.plot(x, y_fir128, color='#db5670')

plt.show()


x = np.arange(0, len(data))

y_av32 = np.array(moving_average(data, 32))
y_av64 = np.array(moving_average(data, 64))
y_av128 = np.array(moving_average(data, 128))

y_fir32 = np.array(move_filter(data, 32))
y_fir64 = np.array(move_filter(data, 64))
y_fir128 = np.array(move_filter(data, 128))


fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(30, 6))

original_alpha = 0.1

axes[0].plot(x, y, alpha=original_alpha)
axes[0].plot(x, y_fir32, color='#db5670')
axes[0].plot(x, y_av32, color='#4ebf8c')
axes[1].plot(x, y, alpha=original_alpha)
axes[1].plot(x, y_av64, color='#4ebf8c')
axes[1].plot(x, y_fir64, color='#db5670')
axes[2].plot(x, y, alpha=original_alpha)
axes[2].plot(x, y_av128, color='#4ebf8c')
axes[2].plot(x, y_fir128, color='#db5670')

plt.show()


### moving average step and impulse responses

x_resp = np.arange(0, 128)
y_step_r = np.concatenate((np.zeros(64), np.ones(64)))
y_step_r = moving_average(y_step_r, 16)


x_imp = np.arange(0, 128)
y_imp_r = np.zeros(128)
y_imp_r[0] = 1
y_imp_r = moving_average(y_imp_r, 32)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 4))

axes[0].plot(x_resp, y_step_r)
axes[1].plot(x_resp, y_imp_r)
plt.show()


###### sinc fir step and impulse responses


x_resp = np.arange(0, 512)
y_step_r = np.concatenate((np.zeros(256), np.full(256, 1)))
y_step_r = move_filter(y_step_r, 64, 6.5)

x_imp = np.arange(0, 512)
y_imp_r = np.zeros(512)
y_imp_r[0] = 1
# what?
y_imp_r = move_filter(y_imp_r, 64, 6.5)

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 4))
axes[0].plot(x_resp, y_step_r)
axes[1].plot(x_imp, y_imp_r)

plt.show()
