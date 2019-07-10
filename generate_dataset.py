"""
Script to generate dataset. This will be in raw spectrogram details as a numpy
array as well as images
"""

import feature_extraction
import pandas as pd
import essentia
# as there are 2 operating modes in essentia which have the same algorithms,
# these latter are dispatched into 2 submodules:
import essentia.standard
import essentia.streaming
from essentia.standard import *
import sys
import json
import os
from multiprocessing import Pool

def _load_audio_file(filename,desired_sample_rate = 44100):
    loader = essentia.standard.MonoLoader(filename = filename, sampleRate = desired_sample_rate)
    audio = loader()
    return audio

def _extract_frequency_for_sample(row_tuple,
                    desired_frame_size = 128,
                    desired_sample_rate=44100,
                    random_sample_pct=0.2,
                    max_samples = 8):
    index = row_tuple[0]
    row = row_tuple[1]
    if not os.path.exists('{}spectrograms_{}_array/fold{}'.format(
                                        urban_sounds_folder,
                                        desired_frame_size,
                                        row['fold'])):
        try:
            os.makedirs('{}spectrograms_{}_array/fold{}'.format(
                                                urban_sounds_folder,
                                                desired_frame_size,
                                                row['fold']))
        except Exception as e:
            pass
    if not os.path.exists('{}spectrograms_{}_array/fold{}/{}'.format(
                                        urban_sounds_folder,
                                        desired_frame_size,
                                        row['fold'],
                                        row['class'])):

        try:
            os.makedirs('{}spectrograms_{}_array/fold{}/{}'.format(
                                            urban_sounds_folder,
                                            desired_frame_size,
                                            row['fold'],
                                            row['class']))
        except Exception as e:
            pass
    # get audio
    audio_file_name = "{}audio/fold{}/{}".format(urban_sounds_folder,
                                                row['fold'],
                                                row['slice_file_name'])
    try:
        audio = _load_audio_file(audio_file_name,desired_sample_rate=desired_sample_rate)
        # get spectrogram
        spectrogram = feature_extraction.extract_log_mel_spectrogram(audio,
                                                    sample_rate=desired_sample_rate,
                                                    n_mels=128,
                                                    n_fft=1024)
        #get continuous spectrograms for desired frame size
        standardized_spectrograms = \
            feature_extraction.extract_frame_sequences_of_size_x(spectrogram,
                            desired_frame_size,
                            random_sample_pct = random_sample_pct,
                            max_samples = max_samples)
        #save files
        for l_idx, spc in enumerate(standardized_spectrograms):
            _name = '{}spectrograms_{}_array/fold{}/{}/{}_shift_{}.json'.format(urban_sounds_folder,
                                desired_frame_size,
                                row['fold'],
                                row['class'],
                                row['slice_file_name'],
                                l_idx)
            with open(_name, 'w') as fp:
                json.dump(spc.tolist(), fp)
        print("Saved {} spectrograms for file in index {}".format(
                                    len(standardized_spectrograms),
                                    index))
    except Exception as e:
        print("Error with file on index {}".format(index))

def extract_frequency_representation(urban_sounds_folder,
                    metadata_location,
                    desired_frame_size = 128,
                    start_row= None,
                    end_row = None,
                    desired_sample_rate=44100):
    metadata = pd.read_csv(metadata_location)
    relevant_observations = metadata[start_row or 0:end_row or len(metadata)]
    with Pool(64) as thread_pool:
        thread_pool.map(_extract_frequency_for_sample,
                        relevant_observations.iterrows())

if __name__ == "__main__":
    arg = sys.argv[1]
    if arg == "extract_json":
        if len(sys.argv)>2:
            start_row = int(sys.argv[2])
        else:
            start_row = None
        if len(sys.argv)>3:
            end_row = int(sys.argv[3])
        else:
            end_row = None

        urban_sounds_folder = "/media/romulo/6237-3231/urban_sound_challenge/"
        metadata_location = urban_sounds_folder+'metadata/UrbanSound8K.csv'
        extract_frequency_representation(urban_sounds_folder,
                                        metadata_location,
                                        start_row=start_row,
                                        end_row=end_row
                                        )
