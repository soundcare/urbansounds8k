import numpy as np
import librosa

def extract_log_mel_spectrogram(samples,sample_rate,n_mels=128,n_fft=1024,
        hop_length = 512):
    """
    Note that for a multichannel audio file the channels are being averaged
    into a single two dimentional array
    """
    feature_set = []
    if samples.shape[1:]:
        channels = samples.shape[1:][0]
    else:
        channels = 1
        samples = np.expand_dims(samples, axis=1)
    for i in range(channels):
        sample_channel_x = samples[:,i]
        mel_spectrogram = librosa.feature.melspectrogram(y=sample_channel_x,
                                                     sr=sample_rate,
                                                     n_fft=n_fft,
                                                     n_mels = n_mels,
                                                     hop_length = hop_length,
                                                     power=2
                                                        )
        decibel_spec = librosa.power_to_db(mel_spectrogram,ref=np.max)
        feature_set.append(decibel_spec)
    feature_set = np.array(feature_set)
    feature_set = np.mean(feature_set,axis=0)
    return feature_set
