# neural_filter_tfcnn
A Feed-forward Neural Network trained to learn a low-pass filter using TFCNNv3.

## Preamble
Originally I used Tensorflow Keras to train the original model [in this repository](https://github.com/jcwml/neural_filter). In that version the training sets are split into 9 sample chunks and a FNN is trained to take in 9 samples and output 9 filtered samples, obviously this is not great for continuity but it was easy to implement using only the high-level Python API for Tensorflow Keras, the original idea was to take 9 samples as input and output one sample and then shift the input one sample forward so that the input was always one sample increments with its four neighbours either side and that is what I have implemented in this version and [TFCNNv3](https://github.com/TFCNN/TFCNNv3) made that job a lot easier for me.

The output is still not great and to be honest not that different from the original but the frequency spectra graph now looks a lot cleaner, in the original Keras version you could see the effect of the rigid 9 input sample chunks in the output frequency spectra.

## How to generate the training dataset
To generate your input files, I used Audacity, Generate > Noise, save that as train_x.raw as Unsigned 8-bit PCM, then apply a low-pass filter to it and save that as train_y.raw in the same format as last time. Then find a song, load it into Audacity, Tracks > Mix > Mix Stereo Down to Mono, then export that as song.raw as Unsigned 8-bit PCM again. Once these three files are placed in the same directory as fit.py you should now be able to execute python3 fit.py 6 32 999 tanh adam 1 9 32. The training process will also generate your first neural transformation of your song.raw and output it as song_output.raw.

## Compile
Just chmod 0700 or similar and execute `compile.sh` you will need GCC installed, you can find GCC in most _(if not all)_ package managers.

