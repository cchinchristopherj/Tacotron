Tacotron
=========================

Generative Models to Synthesize Audio Waveforms Part I

Deep learning has made tremendous leaps and bounds in the field of speech synthesis. While more classical TTS (text-to-speech) pipelines relied on domain expertise to develop different models of linguistic features, vocoders, etc., deep learning-based approaches can be fully end-to-end, thereby preventing errors from compounding, and allowing the neural network to learn the features of importance that would otherwise be hand-engineered. Several different methodologies have been used to tackle this particular problem: one, [Tacotron](https://arxiv.org/pdf/1703.10135.pdf), uses a seq2seq model with attention to encode an input sequence of characters and decode it into raw, linear spectrogram frames. The Griffin-Lim algorithm can then be used to synthesize an audio waveform from the spectal magnitudes. Two other popular approaches are Wavenet and [SampleRNN](https://arxiv.org/pdf/1612.07837.pdf), autoregressive models that generate audio directly on a sample-by-sample basis. While Wavenet relies on dilated convolutional layers, SampleRNN uses a hierarchy of RNN modules to model long-term dependencies at different temporal scales.

The goal of this application was to synthesize a 2-second-long audio waveform of a right whale upcall via generative modeling. Due to the computationally-intensive requirements of sample-level generation, a model based on Tacotron that predicted spectrogram images (of much lower dimensionality than raw audio waveforms) was taken up first. 

The original [Tacotron paper](https://arxiv.org/pdf/1703.10135.pdf) describes a CBHG module (1-D convolution bank + highway network + Bidirectional GRU) capable of extracting excellent representations from sequences by convolving the sequence first with a bank of 1-D convolutional filters to extract local information, passing it through a highway network to extract higher-level features, and finally passing the sequence through a Bidirectional GRU to learn long-term dependencies in the forward and backward directions. In the Tacotron model, an encoder uses this CBHG module to extract a sequential representation of input text, which the attention-based decoder uses to create a sequence of spectrogram frames that can be used to synthesize the correspnoding waveform. 

![CBHG](https://github.com/cchinchristopherj/Tacotron/blob/master/CBHG.png)

*Source: [Tacotron: Towards End-To-End Speech Synthesis](https://arxiv.org/pdf/1703.10135.pdf)*

For simplicity, the decoder targets are 80-band mel spectrograms (a compressed representation that can be used by a post-processing-net later on in the model to synthesize raw spectrograms). This post-processing-net is once again composed of a CBHG module, which learns to predict spectral magnitudes on a linear frequency scale (due to the use of the Griffin-Lim algorithm to create waveforms). A final important design choice made by the authors was the prediction of groups of non-overlapping spectrogram frames for each step of the decoder (instead of one frame at a time). This design choice reduced the total number of decoder steps, the model size, and increased convergence speed.

For the purposes of this application, the neural network has a much simpler task: predicting the next spectrogram frame based on the previous spectrogram frame (and information also learned from all previous frames in the sequence). A seq2seq model with attention is not necessary, since text to speech conversion is not performed, and much of the complexity of the Tacotron model can be removed (such as the use of highway networks, banks of 1-D convolutional filters, etc.). The prediction of (compressed) 80-band mel spectograms is also not necessary, since the low sampling rate of the audio recordings in the dataset (2000 samples/second) means only 129 frequency bins are needed per spectrogram. The model can therefore directly predict linear spectrogram frames for the Griffin-Lim algorithm. 

After initial experimentation, a simple neural network incorporating fully-connected layers, 1-D max pooling to provide translation invariance, and a stacked (two-layer) RNN comprised of Bidirectional GRUs was decided upon. The stacked Bidirectional RNN was specifically chosen for its ability to learn long-term dependencies in sequences in both the forward and backward directions, making it especially applicable for this task of learning sequences of spectrogram frames. 

For the purpose of comparison, here are two representative examples of spectrograms from the training set, in which the corresponding audio files were bandpass-filtered between 50Hz and 440Hz (the characteristic frequency range of upcalls).

![real1](https://github.com/cchinchristopherj/Tacotron/blob/master/real1.png)

*The upcall is visible, but embedded in a background of ambient noise.*

![real2](https://github.com/cchinchristopherj/Tacotron/blob/master/real2.png)

*With less ambient noise present, the upcall is more clearly visible in the image.*

A representative example of an audio file synthesized by the Griffin-Lim algorithm from these spectrograms can be found [here](https://github.com/cchinchristopherj/Tacotron/blob/master/tacotron_real.mp3)

*There is a distinct "choppiness" in the synthesized audio not present in the original audio files of the dataset (most likely due to use of the Griffin-Lim algorithm).*

For comparison, [here](https://github.com/cchinchristopherj/Tacotron/blob/master/tacotron_original.mp3) is a representative example of one of the original audio files of an upcall from the dataset.

After training for 5 epochs, the neural network was tasked with predicting a new spectrogram from the state space. Below is a representative example: 

![oneframe](https://github.com/cchinchristopherj/Tacotron/blob/master/oneframe.png)

*The spectrogram appears to have captured some of the characteristics of ambient noise, but the shape of the upcall is not clearly present.*

The corresponding audio waveform synthesized via the Griffin-Lim algorithm can be found [here](https://github.com/cchinchristopherj/Tacotron/blob/master/tacotron_oneframe.mp3)

*The sound is reminiscent of the audio synthesized by Griffin-Lim for the spectrograms of the original dataset. However, the "choppy" audio artifacts can still be heard, suggesting that an alternative to the Griffin-Lim algorithm should be explored in future work.*

Due to the lack of upcall shape in the generated spectrogram, it was hypothesized that predicting groups of spectrogram frames (instead of one frame at a time) would improve results. (The model might be able to more easily capture the characteristic spectro-temporal shape of an upcall, which spans several time steps, if tasked with predicting groups of spectrogram frames). A group size of 5 (num_frames) was decided upon, so that each prediction from the model was in fact a prediction for five frames of the spectrogram.

After training for 5 epochs, the neural network was tasked with predicting a new spectrogram, with a representative example displayed below:

![manyframes](https://github.com/cchinchristopherj/Tacotron/blob/master/manyframes.png)

*The upcall shape is still not present. However, the neural network appears to be closer to capturing the higher magnitude values (indicated by the brighter yellow colors) associated with frequencies of the upcall. 

The corresponding audio waveform synthesized via the Griffin-Lim algorithm can be found [here](https://github.com/cchinchristopherj/Tacotron/blob/master/tacotron_manyframes.mp3)

*The audio is very similar to that generated by the spectrogram trained to predict individual frames, suggesting an alternative architecture, hyperparameters, or synthesis algorithm are necessary to improve results.*
