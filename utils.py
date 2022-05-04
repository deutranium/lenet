import numpy as np
import pickle as pk

# pad the image in data by pad_length
def padder(data, pad_length):
    im = data.copy()
    m, n = im.shape
    M, N = m + 2*pad_length, n + 2*pad_length

    im2 = np.zeros((M, N))
    im2[pad_length:m+pad_length, pad_length:n+pad_length] = im[:, :]

    return im2

# image to matrix based on stride
def conv2d(inputs, weights, bias, padding, K, F, stride=1):
    C, W, H = inputs.shape
    WW = (W - K)//stride + 1
    HH = (H - K)//stride + 1

    feature_maps = np.zeros((F, WW, HH))

    for f in range(F):
        for w in range(0, W-K, stride):
            for h in range(0, H-K, stride):
                # ic(f, w, h, K, weights[f, :, :, :].shape, bias[f].shape, inputs[:, w:w+K, h:h+K].shape)
                feature_maps[f,w,h]=np.sum(inputs[:,w:w+K,h:h+K]*weights[f,:,:,:])+bias[f]

    return feature_maps


# take average of the image chunks based on stride and pool size
def avg_pool(data, pool_size, stride):
        C, W, H = data.shape

        new_width = (W - pool_size)//stride + 1
        new_height = (H - pool_size)//stride + 1

        out = np.zeros((C, new_width, new_height))

        for c in range(C):
            for w in range((W-pool_size)//stride):
                for h in range((H-pool_size)//stride):
                    out[c, w, h] = np.max(data[c, w*stride:w*stride+pool_size, h*stride:h*stride+pool_size])
  
        return out


# softmax function
def softmax(x):
    denom = np.sum(np.exp(x))
    y = x/denom;
    return y


# sigmoid function
def sigmoid(x):
    return 1/(1+np.exp(-x))


# tanh function
def tanh(x):
    a = 1.7159 
    s = 2/3
    return a*np.tanh(s*x)


# tanhh`
def back_tanh(x):
    a = 1.7159 
    s = 2/3
    return a*s*(1-np.tanh(s*x)**2)

# ReLu function
def relu(x):
    return np.max(x,0)


def vanilla(data, weights, bias, activation = 'Tanh'):
    if activation == 'Tanh':
        # try:
        return tanh(np.dot(np.transpose(weights),data) + bias)
        # except:
        #   ic(weights.shape,data.shape,bias.shape)
        #   exit()
    elif activation == 'sigmoid':
        return  sigmoid(np.dot(np.transpose(weights),data) + bias)


# store data as pickle at location loc
def store_pickle(data, loc):
    try:
        pk.dump(data, open(loc, "wb"))
        return 0
        
    except Exception as e:
        return e


# one hot encode
def one_hot(x):
    temp = np.zeros(10)
    temp[x] = 1
    return temp