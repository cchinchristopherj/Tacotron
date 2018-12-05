import numpy as np
import glob
import aifc
import scipy.signal as sp
import scipy
import math
from math import sqrt
from scipy.signal import butter, lfilter, freqz, filtfilt
import array
import wave
import os
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from keras.models import load_model
import scipy.signal as sp
import copy
from scipy.io import wavfile
import pylab as pl
from matplotlib import mlab
import tensorflow as tf
import keras.optimizers
from keras.optimizers import Adam
import keras.regularizers
from keras import initializers
from keras.callbacks import Callback
from keras import layers
from keras.layers import Input, Dense, TimeDistributed, Conv1D, MaxPooling1D, BatchNormalization
from keras.layers import Lambda, GRU, Reshape, Embedding, Dropout, Bidirectional
from keras.models import Model
from keras import backend as K
from keras.engine import InputSpec
K.set_image_data_format('channels_last')
%matplotlib inline
plt.rcParams['figure.figsize'] = (13,5)

# Based off ReadAIFF function from: https://github.com/nmkridler/moby/blob/master/fileio.py
def ReadAIFF(file,maxlen):
    ''' ReadAIFF Method
            Read AIFF and convert to numpy array of specified size
            
            Args: 
                file: string file to read 
                maxlen: desired number of samples in audio waveform
            Returns:
                numpy array containing whale audio clip      
                
    '''
    s = aifc.open(file,'r')
    nFrames = s.getnframes()
    strSig = s.readframes(nFrames)
    finalSig = np.frombuffer(strSig,np.short).byteswap()
    # If "finalSig" contains fewer samples than maxlen, continue to pad the last sample of
    # finalSig" to the end of the vector until the size of "finalSig" is equal to maxlen
    if len(finalSig) < maxlen:
        last_value = finalSig[-1] 
        zeros_append = np.ones(maxlen-len(finalSig))*last_value
        new_array = np.append(finalSig,zeros_append)
        finalSig = new_array
    return finalSig

# Based off H1Sample Function from: https://github.com/nmkridler/moby/blob/master/fileio.py
def SpecGram(file,params=None):
    ''' SpecGram Method 
            Convert audio file to spectrogram for CNN and pre-process input for
            input shape uniformity 
            
            Args:
                file: time series vector of audio waveform
                params: dictionary containing spectrogram parameters  
            Returns: 
                Pre-Processed Spectrogram matrix and frequency/time bins as 1-D arrays
                
    '''
    # Convert to spectrogram 
    P,freqs,bins = mlab.specgram(file,**params)
    m,n = P.shape
    # Ensure all image inputs to the CNN are the same size. If the number of time bins 
    # is less than 59, pad with zeros 
    if n < 59:
        Q = np.zeros((m,59))
        Q[:,:n] = P
    else:
        Q = P
    return Q,freqs,bins

# PlotSpecgram Function from: https://github.com/nmkridler/moby2/blob/master/plotting.py
def PlotSpecgram(P,freqs,bins):
    ''' PlotSpecgram Method 
            Plot the spectrogram
            
            Args:
                P: 2-D numpy array image
                freqs: 1-D array of frequency bins
                bins: 1-D array of time bins   
    '''
    # Use np.flipud so that the spectrogram plots correctly 
    Z = np.flipud(P)
    xextent = 0,np.amax(bins)
    xmin,xmax = xextent
    extent = xmin,xmax,freqs[0],freqs[-1]
    im = pl.imshow(Z,extent=extent)
    pl.axis('auto')
    pl.xlim([0.0,bins[-1]])
    pl.ylim([0,400])

def extract_labels(file):
    ''' extract_labels Method 
            Since the dataset file names contain the labels (0 or 1) right before
            the extension, appropriately parse the string to obtain the label 
            
            Args:
                file: string file to read 
            Returns: 
                int label of the file (0 or 1) 
                
    '''
    name,extension = os.path.splitext(file)
    label = name[-1]
    return int(label)

def minmaxscaling(X,minmaxscaler,flag=1):
    ''' minmaxscaling Method 
            Scales the input to the desired range using sklearn's MinMaxScaler()
            
            Args:
                X: Dataset 
                minmaxscaler: Instance of sklearn's MinMaxScaler() with pre-defined feature range
                flag: 1 indicates MinMaxScaler() should be fit to data then transform it, while
                      0 indicates MinMaxScaler() should solely transform the data
            Returns: 
                Scaled verion of X
                
    '''
    # Dimensions of X
    num_samples = X.shape[0]
    height = X.shape[1]
    width = X.shape[2]
    # Reshape X into a 2D array for MinMaxScaler()
    dataset = X.reshape((num_samples,height*width))
    # Flag value of 1 indicates MinMaxScaler() has already been fit
    if flag == 1:
        dataset = minmaxscaler.fit_transform(dataset)
    elif flag == 0:
        dataset = minmaxscaler.transform(dataset)
    # Reshape dataset into the original shape
    dataset = dataset.reshape((num_samples,height,width))
    X = dataset
    return X

# Code from: https://github.com/bkvogel/griffin_lim/blob/master/audio_utilities.py
def reconstruct_signal_griffin_lim(magnitude_spectrogram, fft_size, hopsamp, iterations):
    ''' reconstruct_signal_griffin_lim Method
            Reconstruct an audio signal from a magnitude spectrogram.
            Given a magnitude spectrogram as input, reconstruct
            the audio signal and return it using the Griffin-Lim algorithm from the paper:
            "Signal estimation from modified short-time fourier transform" by Griffin and Lim,
            in IEEE transactions on Acoustics, Speech, and Signal Processing. Vol ASSP-32, No. 2, April 1984.
            
            Args:
                magnitude_spectrogram (2-dim Numpy array): The magnitude spectrogram. The rows correspond to the time slices
                    and the columns correspond to frequency bins.
                fft_size (int): The FFT size, which should be a power of 2.
                hopsamp (int): The hope size in samples.
                iterations (int): Number of iterations for the Griffin-Lim algorithm. Typically a few hundred
                    is sufficient.
            Returns:
                The reconstructed time domain signal as a 1-dim Numpy array.
                
    '''
    time_slices = magnitude_spectrogram.shape[0]
    len_samples = int(time_slices*hopsamp + fft_size)
    # Initialize the reconstructed signal to noise.
    x_reconstruct = np.random.randn(len_samples)
    n = iterations # number of iterations of Griffin-Lim algorithm.
    while n > 0:
        n -= 1
        reconstruction_spectrogram = stft_for_reconstruction(x_reconstruct, fft_size, hopsamp)
        reconstruction_angle = np.angle(reconstruction_spectrogram)
        # Discard magnitude part of the reconstruction and use the supplied magnitude spectrogram instead.
        proposal_spectrogram = magnitude_spectrogram*np.exp(1.0j*reconstruction_angle)
        prev_x = x_reconstruct
        x_reconstruct = istft_for_reconstruction(proposal_spectrogram, fft_size, hopsamp)
        diff = sqrt(sum((x_reconstruct - prev_x)**2)/x_reconstruct.size)
        print('Reconstruction iteration: {}/{} RMSE: {} '.format(iterations - n, iterations, diff))
    return x_reconstruct
    
# Code from: https://github.com/bkvogel/griffin_lim/blob/master/audio_utilities.py
def stft_for_reconstruction(x, fft_size, hopsamp):
    ''' stft_for_reconstruction Method
            Compute and return the STFT of the supplied time domain signal x.
            
            Args:
                x (1-dim Numpy array): A time domain signal.
                fft_size (int): FFT size. Should be a power of 2, otherwise DFT will be used.
                hopsamp (int):
            Returns:
                The STFT. The rows are the time slices and columns are the frequency bins.
                
    '''
    window = np.hanning(fft_size)
    fft_size = int(fft_size)
    hopsamp = int(hopsamp)
    return np.array([np.fft.rfft(window*x[i:i+fft_size]) for i in range(0, len(x)-fft_size, hopsamp)])

# Code from: https://github.com/bkvogel/griffin_lim/blob/master/audio_utilities.py
def istft_for_reconstruction(X, fft_size, hopsamp):
    ''' istft_for_reconstruction Method
            Invert a STFT into a time domain signal.
            
            Args:
                X (2-dim Numpy array): Input spectrogram. The rows are the time slices and columns are the frequency bins.
                fft_size (int):
                hopsamp (int): The hop size, in samples.
            Returns:
                The inverse STFT.
                
    '''
    fft_size = int(fft_size)
    hopsamp = int(hopsamp)
    window = np.hanning(fft_size)
    time_slices = X.shape[0]
    len_samples = int(time_slices*hopsamp + fft_size)
    x = np.zeros(len_samples)
    for n,i in enumerate(range(0, len(x)-fft_size, hopsamp)):
        x[i:i+fft_size] += window*np.real(np.fft.irfft(X[n]))
    return x

# Code from: https://github.com/bkvogel/griffin_lim/blob/master/audio_utilities.py
def save_audio_to_file(x, sample_rate, outfile='out.wav'):
    ''' save_audio_to_file Method
            Save a mono signal to a file.
            
            Args:
                x (1-dim Numpy array): The audio signal to save. The signal values should be in the range [-1.0, 1.0].
                sample_rate (int): The sample rate of the signal, in Hz.
                outfile: Name of the file to save.
                
    '''
    x_max = np.max(abs(x))
    assert x_max <= 1.0, 'Input audio value is out of range. Should be in the range [-1.0, 1.0].'
    x = x*32767.0
    data = array.array('h')
    for i in range(len(x)):
        cur_samp = int(round(x[i]))
        data.append(cur_samp)
    f = wave.open(outfile, 'w')
    f.setparams((1, 2, sample_rate, 0, "NONE", "Uncompressed"))
    f.writeframes(data.tobytes())
    f.close()
    
def fix_batch(X_train,num_to_add):
    ''' fix_batch Method
            Select "num_to_add" random samples from the training set and add them to the end of the training
            set to ensure that the total number of samples in the training set is a multiple of the batch size
            
            Args:
                X_train: Training set
                num_to_add: Number of random samples from the training set to append to the end
            Returns:
                Training set with "num_to_add" newly-appended samples
                
    '''
    for ii in range(num_to_add):
        ind = np.random.randint(X_train.shape[0])
        X_train = np.append(X_train,X_train[ind,:,:][np.newaxis,...],axis=0)
    return X_train
    
# The neural network model outputs raw spectrogram frames, which can subsequently be converted to audio waveforms 
# using the Griffin-Lim algorithm
# The architecture is based on the Tacotron model from the original paper, which implements a CBHG module
# (1-D convolution bank + highway network + Bidirectional GRU) capable of extracting excellent representations from 
# sequences by convolving the sequence first with a bank of 1-D convolutional filters to extract local information, 
# passing it through a highway network to extract higher-level features, and finally passing the sequence through a 
# Bidirectional GRU to learn long-term dependencies in the forward and backward directions. In the Tacotron model, 
# an encoder uses this CBHG module to extract a sequential representation of input text, which the attention-based
# decoder uses to create a sequence of spectrogram frames that can be used to synthesize the correspnoding waveform.
# For simplicity, the decoder targets are 80-band mel spectrograms (a compressed representation that can be used by 
# a post-processing-net later on in the model to synthesize raw spectrograms). This post-processing-net is once again
# composed of a CBHG module, which learns to predict spectral magnitudes on a linear frequency scale due to the use
# of the Griffin-Lim algorithm to create waveforms. A final important design choice made by the authors was the
# prediction of groups of non-overlapping spectrogram frames for each step of the decoder (instead of one frame at
# a time). This design choice reduced the total number of decoder steps, the model size, and increased convergence
# speed.

# For the purposes of this application, the neural network has a much simpler task: predicting the next spectrogram
# frame based on the previous spectrogram frame (and information also learned from all previous frames in the
# sequence). A seq2seq model with attention is not necessary, since text to speech conversion is not performed, and
# much of the complexity of the Tacotron model can be removed (such as the use of highway networks, banks of 1-D
# convolutional filters, etc.). The prediction of (compressed) 80-band mel spectograms is also not necessary, since 
# the low sampling rate of the audio recordings in the dataset (2000 samples/second) means only 129 frequency bins
# are needed per spectrogram. The model can therefore directly predict linear spectrogram frames for the Griffin-Lim
# algorithm. 

# After initial experimentation, a simple neural network incorporating fully-connected layers, 1-D max pooling to 
# provide translation invariance, and a stacked (two-layer) RNN comprised of Bidirectional GRUs was decided upon. 
# The stacked Bidirectional RNN was specifically chosen for its ability to learn long-term dependencies in sequences 
# in both the forward and backward directions, making it especially applicable for this task of learning sequences of 
# spectrogram frames. 

# It was also hypothesized that predicting groups of spectrogram frames (instead of one frame at a time) would improve
# results, since the model might be able to more easily capture the characteristic spectro-temporal shape of an upcall
# in predictions of multiple frames. A group size of 5 (num_frames) was decided upon, so that each prediction from
# the model was in fact a prediction for five frames of the spectrogram.

# Model parameters
# Batch size
n_batch=32
# Number of frames in each group predicted by the model
num_frames = 5
# Dimension of GRUs
RNN_DIM = 128
# # Number of mel bands
# MEL_DIM = 80
# Number of bins in linear frequency scale
LINEAR_DIM = 129
# # Number of convolutional filters per layer
# BANK_DIM = 128
# Due to the design of the GRUs as implemented by Keras, each prediction from the GRU for a time step is a vector
# (not a matrix). Therefore, in order to predict non-overlapping groups of spectrogram frames, the prediction of the 
# GRU for a time step must be interpreted as a flattened matrix of frames. 
# Concretely, note that the input to an RNN must be of shape (batch_size,time_steps,num_features). The GRU will output
# a prediction vector for each time step that is num_features long. Therefore, if the number of bins in one spectrogram
# frame is LINEAR_DIM, and the desired number of frames in a group is num_frames, the number of features for each time
# step must be equal to num_frames*LINEAR_DIM.
FEATURES = num_frames*LINEAR_DIM
# The training set must likewise be constructed in such a way that each time step corresponds to a group of spectrogram
# frames, i.e. the training set is of shape (batch_size,time_steps,num_features) and each sample in the training set
# must consist of vectors at each time step that are FEATURES long (flattened matrices corresponding to the groups of
# spectrogram frames).

# Original neural network architecture that more closely followed the CBHG module design from the original Tacotron 
# paper, incorporating 1-D convolutional filter banks and the initial prediction of 80-band mel spectrograms
# def create_model(batch_input_shape,flag=1,rnn_dim=RNN_DIM,mel_dim=MEL_DIM,linear_dim=LINEAR_DIM,bank_dim=BANK_DIM):
#     ''' create_model Method
#             Create neural network model
            
#             Args:
#                 batch_input_shape: Input shape including batch axis
#                 flag: 1 indicates training, while 0 indicates prediction
#                 rnn_dim: Dimension of GRUs
#                 mel_dim: Number of mel bands
#                 linear_dim: Number of bins in linear frequency scale
#                 bank_dim: Number of convolutional filters per layer
#             Returns:
#                 Neural network model
                
#     '''
#     input_shape = (batch_input_shape[1],batch_input_shape[2])
#     X_input = Input(batch_shape=batch_input_shape,name='input')
#     prenet1 = Dense(64,activation='relu',name='prenet1')
#     X = TimeDistributed(prenet1,name='td_prenet1')(X_input)
#     X = Dropout(0.5,name='dropout1')(X)
#     prenet2 = Dense(64,activation='relu',name='prenet2')
#     X = TimeDistributed(prenet2,name='td_prenet2')(X)
#     X = Dropout(0.5,name='dropout2')(X)
#     if flag == 1:
#         rnn1 = Bidirectional(GRU(rnn_dim,name='rnn1',return_sequences=True))
#     elif flag == 0:
#         rnn1 = Bidirectional(GRU(rnn_dim,name='rnn1',return_sequences=True,stateful=True))
#     X = rnn1(X)
#     adapter1 = Dense(mel_dim,use_bias=False,name='adapter1')
#     X_output1 = TimeDistributed(adapter1,name='td_adapter1')(X)
#     X_bank1 = Conv1D(bank_dim,3,padding='same',activation='relu',name='bank1')(X_output1)
#     X_bn1 = BatchNormalization(name='bn1')(X_bank1)
#     X_bank2 = Conv1D(bank_dim,5,padding='same',activation='relu',name='bank2')(X_output1)
#     X_bn2 = BatchNormalization(name='bn2')(X_bank2)
#     X_bank3 = Conv1D(bank_dim,7,padding='same',activation='relu',name='bank3')(X_output1)
#     X_bn3 = BatchNormalization(name='bn3')(X_bank3)
#     X_bank = keras.layers.concatenate([X_bn1,X_bn2,X_bn3],name='concat')
#     X = MaxPooling1D(strides=1,padding='same',name='maxpool1')(X_bank)
#     adapter2 = Dense(bank_dim,use_bias=False,name='adapter2')
#     X = TimeDistributed(adapter2,name='td_adapter2')(X)
#     if flag == 1:
#         rnn2 = Bidirectional(GRU(rnn_dim,name='rnn2',return_sequences=True))
#     elif flag == 0:
#         rnn2 = Bidirectional(GRU(rnn_dim,name='rnn2',return_sequences=True,stateful=True))
#     X = rnn2(X)
#     adapter3 = Dense(linear_dim,use_bias=False,name='adapter3')
#     X_output2 = TimeDistributed(adapter3,name='td_adapter3')(X)
#     model = Model(inputs=X_input,outputs=[X_output1,X_output2])
#     return model

def create_model(batch_input_shape,flag=1,rnn_dim=RNN_DIM,features=FEATURES):
    ''' create_model Method
            Create neural network model
            
            Args:
                batch_input_shape: Input shape including batch axis
                flag: 1 indicates training, while 0 indicates prediction
                rnn_dim: Dimension of GRUs
                features: Number of features for each time step in the training set
            Returns:
                Neural network model
                
    '''
    input_shape = (batch_input_shape[1],batch_input_shape[2])
    X_input = Input(batch_shape=batch_input_shape,name='input')
    prenet1 = Dense(128,activation='relu',name='prenet1')
    X = TimeDistributed(prenet1,name='td_prenet1')(X_input)
    X = Dropout(0.5,name='dropout1')(X)
    X = MaxPooling1D(strides=1,padding='same',name='maxpool1')(X)
    adapter2 = Dense(128,use_bias=False,name='adapter2')
    X = TimeDistributed(adapter2,name='td_adapter2')(X)
    if flag == 1:
        rnn1 = Bidirectional(GRU(rnn_dim,name='rnn1',return_sequences=True))
        rnn2 = Bidirectional(GRU(rnn_dim,name='rnn2',return_sequences=True))
    elif flag == 0:
        rnn1 = Bidirectional(GRU(rnn_dim,name='rnn1',return_sequences=True,stateful=True))
        rnn2 = Bidirectional(GRU(rnn_dim,name='rnn2',return_sequences=True,stateful=True))
    X = rnn1(X)
    X = rnn2(X)
    adapter3 = Dense(features,use_bias=False,name='adapter3')
    X_output = TimeDistributed(adapter3,name='td_adapter3')(X)
    model = Model(inputs=X_input,outputs=X_output)
    return model

def sample(model_pred,X_train,minmaxscaler_linear,num_frames):
    ''' sample Method
            Use the trained neural network to make a prediction for a new spectrogram
            from the state space (and corresponding audio waveform)
            
            Args:
                model_pred: The model used to make predictions
                X_train: Training set
                minmaxscaler_linear: Pre-fit instance of sklearn's MinMaxScaler()
                num_frames: Number of frames in each group
            Returns:
                samples: 2D array of the spectrogram generated by the neural network
                
    '''    
    # First instantiate an empty array of zeros with the expected shape of the output to be
    # predicted by the neural network. The shape is, once again, of the form 
    # (batch_size,time_steps,num_features)
    # Number of time steps for the output
    ts = X_train.shape[1]+1
    # Note that the number of time steps for the output is one greater than the number of time
    # steps for samples in the training set. Recall that the input to the neural network is X_train, 
    # the sequences of flattend groups of linear spectrogram frames, and the neural network is tasked
    # with predicting the next flattened group of frames for every input flattened group of frames. In 
    # other words, given the first flattened group, the model must predict the second. Given the second 
    # flattened group, the model must predict the third, and so on. For this reason, X_train must consist 
    # of flattened groups of spectrogram frames from the first to the second to last in each sequence, 
    # since the model will make predictions for the second flattened group up to the last flattened group. 
    # Y_train is therefore comprised of the second flattened group to the last flattened group, making it 
    # the ground truth for predictions made by the model.
    # X_train and Y_train are therefore both one flattened group shorter than the number of flattened groups
    # in the original dataset. 
    # Due to sklearn's MinMaxScaler() having been used to normalize values for the original dataset of 
    # (non-grouped) spectrogram frames, the output sequence of flattened groups of spectrogram frames from
    # the neural network must be reshaped into the dimensions of spectrograms in the original dataset in order
    # for the same instance of MinMaxScaler() to be applied to it.
    # However, due to non-overlapping groups of spectrogram frames being predicted by the model (and the number
    # of time steps for spectrograms in the original dataset being a prime number (59), it is not possible
    # for the output of the neural network to be evenly reshaped into the dimensions of spectrograms in the
    # original dataset. The variable "ts" is made one (flattened group of spectrogram frames) longer in order
    # to accommodate this discrepancy). In other words, after the sequence of flattened groups of spectrogram
    # frames is reshaped into the sequence of (non-grouped) spectrogram frames as in the original dataset,
    # frames from the extra time step prediction made by the neural network will be added to the end of the 
    # sequence so that the dimensions of the final spectrogram are as desired). 
    samples = np.zeros((1,ts,X_train.shape[2]))
    # Since the model makes predictions for every subsequent flattened group in a sequence, it can only make 
    # predictions from the second flattened group up to the last flattened group. Therefore, the first flattened
    # group in the sequence must be known (or assumed) a priori.
    # In this case, the first flattened group from a random sample in the training set will be used as the first 
    # flattened group in the new sequence
    ind = np.random.randint(X_train.shape[0])
    samples[:,0] = X_train[ind][0]
    # The model will take every flattened group starting from the first as input and make a prediction for the
    # subsequent flattened group. Save each flattened group in the appropriate location in "samples."
    for t in range(1, ts):
    #         sample_mel,sample_linear = model_pred.predict_on_batch([samples[:,t-1:t]])
        sample_linear = model_pred.predict_on_batch([samples[:,t-1:t]])
        sample_linear = sample_linear.reshape((sample_linear.shape[1],sample_linear.shape[2]))
        samples[:, t] = sample_linear
    output_final = np.zeros((59,129))
    counter = 0
    final_index = 0
    for ii in range(0,output_final.shape[0]-num_frames,num_frames):
        temp = samples[:,counter].reshape((num_frames,129))
        output_final[ii:ii+num_frames] = temp
        counter += 1
        final_index = ii
    final_index = final_index + num_frames
    output_final[final_index:] = samples[:,-1].reshape((num_frames,129))[0:(output_final.shape[0]-final_index)]
    # Note that X_train and Y_train contain normalized samples (with values ranging from 0 to 1), constraining 
    # the model to learn how to predict normalized flattened groups of spectrogram frames.
    # The inverse_transform() function of the instance of MinMaxScaler() originally used to normalize the dataset
    # will be used to transform the values of the predicted (normalized) sequence of flattened groups of
    # spectrogram frames back to the scale of the original dataset. 
    # First reshape the "samples" array into a 2D array as expected by MinMaxScaler().
    samples_n = output_final[np.newaxis,...]
    samples_n = samples_n.reshape((samples_n.shape[0],samples_n.shape[1]*samples_n.shape[2]))
    samples_n = minmaxscaler_linear.inverse_transform(samples_n)
    samples_n = samples_n.reshape((output_final.shape[0],output_final.shape[1]))
    # Use the Griffin Lim algorithm to synthesize an audio waveform corresponding to the 
    # magnitude spectrogram
    x_reconstruct = reconstruct_signal_griffin_lim(samples_n, fft_size=256, hopsamp=192, iterations=20)
    # Normalize the samples of the audio waveform
    x_reconstruct = x_reconstruct/max(abs(x_reconstruct))
    sample_rate=2000
    # Save the waveform as a .wav file
    save_audio_to_file(x_reconstruct,sample_rate,outfile='out1.wav')
    return samples_n

def butter_bandpass(lowcut, highcut, fs, order=5):
    ''' butter_bandpass Method
            Derive filter coefficients for Butterworth bandpass filter
            
            Args:
                lowcut: (unnormalized) lower cutoff frequency
                highcut: (unnormalized) upper cutoff frequency
                fs: sampling rate
                order: filter order
            Returns:
                b,a: Butterworth bandpass filter coefficients
                
    '''    
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    ''' butter_bandpass_filter Method
        Apply Butterworth bandpass filter to input signal

        Args:
            data: input signal
            lowcut: (unnormalized) lower cutoff frequency
            highcut: (unnormalized) upper cutoff frequency
            fs: sampling rate
            order: filter order
        Returns:
            b,a: Butterworth bandpass filter coefficients
                
    '''    
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y


def bandpass_filter(X):
    ''' bandpass_filter Method
            Helper Function to facilitate calling butter_bandpass_filter in
            a list comprehension with desired values of the arguments

            Args:
                X: input signal
            Returns:
                y: filtered output signal
                
    '''    
    # Filter requirements
    order = 6
    fs = 2000      
    lowcut = 50
    highcut = 440
    data = X
    y = butter_bandpass_filter(data,lowcut,highcut,fs,order)
    return y

def partition_sequences(X, num_frames):
    ''' partition_sequences Method
        Take the original dataset in which samples consist of sequences of (non-grouped) 
        spectrogram frames and re-partition each sequence into flattened groups of 
        spectrogram frames, with each group consisting of num_frames frames
        
        Args:
            X: Dataset
            num_frames: Number of frames in each group
        Returns:
            X_final: New dataset in which samples consist of sequences of flattened groups
                     of spectrogram frames
                
    '''    
    num_subsequences = int(X.shape[1]/num_frames)
    X_final = np.zeros((X.shape[0],num_subsequences,num_frames*X.shape[2]))
    for ii in range(X.shape[0]):
        counter = 0
        for jj in range(0,X.shape[1]-num_frames,num_frames): 
            temp = X[ii,jj:jj+num_frames]
            temp = temp.reshape(-1)
            X_final[ii,counter] = temp
            counter += 1
    return X_final

# Spectrogram parameters 
params = {'NFFT':256,'Fs':2000,'noverlap':192}
# Identify the location of the dataset
path = 'Documents/Bioacoustics_MachineLearning/train2'
filenames = glob.glob(path+'/*.aif')
# Extract labels for each file from the file names
Y_train = np.array([extract_labels(x) for x in filenames])
# Identify which samples of the dataset contain upcalls (labeled 1)
pos_indexes = np.where(Y_train==1)[0]
pos_filenames = list(np.array(filenames)[pos_indexes])
# Create a dataset of audio files containing upcalls
X_train_audio = np.array([np.array(ReadAIFF(x,4000)) for x in pos_filenames])
# Filter the audio files in X_train_audio using bandpass_filter()
X_train_filt = np.array([np.array(bandpass_filter(x)) for x in X_train_audio])
# Extract the spectrograms for each for each filtered audio file
X_train = np.array([SpecGram(x,params=params)[0] for x in X_train_filt])
# Option for instead extracting spectrograms with vertically-enhanced contrast
# X_train = np.array([extract_featuresV(x,params=params) for x in pos_filenames])

# Instance of MinMaxScaler() for linear spectrograms (default feature range of 0 to 1)
minmaxscaler_linear = MinMaxScaler()

# Scale the datasets of linear spectrograms for input to the neural network
X_train = minmaxscaling(X_train,minmaxscaler_linear,flag=1)

# Add 4 random samples from X_train to the end of the dataset to ensure that the total
# number of samples is a multiple of the batch size
num_to_add = 4
X_train = fix_batch(X_train,num_to_add)

# The reconstruct_signal_griffin_lim algorithm requires the input spectrograms to have rows that 
# correspond to time slices and columns that correspond to frequency bins (the reverse of the shape
# output by the SpecGram function used to generate the training set). Therefore, swap the 1st and 
# 2nd axes of the X_train dataset
X_train = np.swapaxes(X_train,1,2)

# Re-partition the dataset into sequences of flattened groups of spectrogram frames, with each
# group consisting of num_frames frames
num_frames = 5
X_train = partition_sequences(X_train,num_frames)

# Create the training set (X_train) and corresponding set of ground truth annotations (Y_train_linear)
Y_train_linear = np.copy(X_train)[:,1:,:]
X_train = X_train[:,0:-1,:]

# Specify the input shape, including the batch axis
batch_input_shape=(n_batch, X_train.shape[1], X_train.shape[2])
# Create the model and compile it. 
model_train = create_model(batch_input_shape)
model_train.compile(loss='mean_absolute_error',optimizer='adam')

# Train the model
model_train.fit(X_train,Y_train_linear,batch_size=32,epochs=2)
# After training save the weights of the model
file_name = 'taco.h5'
model_train.save_weights(file_name)

# Use the trained model to create a prediction for a new spectrogram from the state space
# A different iteration of the neural network will be created for prediction since the 
# batch size for prediction is equal to 1 (the model will be predicting on a frame-by-frame
# basis). The weights will be loaded into the corresponding layers from the trained neural 
# network
n_batch=1
batch_input_shape=(n_batch,1,X_train.shape[2])
model_pred = create_model(batch_input_shape,flag=0)
model_pred.compile(loss='mean_absolute_error',optimizer='adam')
file_name = 'taco.h5'
model_pred.load_weights(file_name)
# Predict a new sequence of spectrogram frames and save the corresponding audio waveform
# as a .wav file
samples = sample(model_pred,X_train,minmaxscaler_linear,num_frames)
