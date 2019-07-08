import numpy as np
import librosa
import random

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


def extract_frame_sequences_of_size_x(spectrogram, desired_shape_x,
                                    random_sample_pct = None):
    """
    Given a spectrogram (as a two dimentional numpy array), extract
    as many smaller versions of the spectrogram as possible given the desired
    shape
    """
    shape_y, shape_x = spectrogram.shape
    n_shifts = max(shape_x-desired_shape_x,0)
    possible_starting_points = range(n_shifts)
    if random_sample_pct:
        starting_points = random.sample(possible_starting_points,
                            int(random_sample_pct*len(possible_starting_points))
                            )
    else:
        starting_points = possible_starting_points
    if len(starting_points):
        results = []
        for idx_x in starting_points:
            end_x = idx_x + shape_x
            results.append(spectrogram[:,idx_x:idx_x+desired_shape_x])
        return results
    return [spectrogram]
