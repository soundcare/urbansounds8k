import feature_extraction
import pandas as pd
import essentia
import numpy as np
import sys
import os
from multiprocessing import Pool
from pylab import plot, show, figure, imshow
import matplotlib.pyplot as plt

def generate_img_from_npy(row_tuple):
    class_map = { 0: 'air_conditioner',
        1:'car_horn',
        2:'children_playing',
        3:'dog_bark',
        4:'drilling',
        5:'engine_idling',
        6:'gun_shot',
        7:'jackhammer',
        8:'siren',
        9:'street_music'}
    urban_sounds_folder = '/media/romulo/3363-3835/spectrograms_128_array/'
    row = row_tuple[1]
    index = row_tuple[0]
    class_id = int(row['classID'])
    row['class'] = class_map[class_id]
    desired_frame_size = 128
    if not os.path.exists('{}spectrograms_{}_array/png/fold{}'.format(
                                        urban_sounds_folder,
                                        desired_frame_size,
                                        row['fold'])):
        try:
            os.makedirs('{}spectrograms_{}_array/png/fold{}'.format(
                                                urban_sounds_folder,
                                                desired_frame_size,
                                                row['fold']))
        except Exception as e:
            pass
    if not os.path.exists('{}spectrograms_{}_array/png/fold{}/{}'.format(
                                        urban_sounds_folder,
                                        desired_frame_size,
                                        row['fold'],
                                        row['class'])):

        try:
            os.makedirs('{}spectrograms_{}_array/png/fold{}/{}'.format(
                                            urban_sounds_folder,
                                            desired_frame_size,
                                            row['fold'],
                                            row['class']))
        except Exception as e:
            pass
    # get audio
    spectrogram = np.load(urban_sounds_folder+row['location'])[:128,:128]
    #only need to pad over the y axis
    y_pad_size = 128-spectrogram.shape[1]
    if y_pad_size > 0:
        spectrogram = np.pad(spectrogram, ((0,0),(0,y_pad_size)), 'constant')

    #save file
    filename = '{}spectrograms_{}_array/png/fold{}/{}/{}-{}.png'.format(
                                            urban_sounds_folder,
                                            desired_frame_size,
                                            row['fold'],
                                            row['class'],
                                            row['fsID'],
                                            index
                                            )
    fig = plt.figure()
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.imshow(spectrogram, aspect = 'equal', origin='lower', interpolation='none', cmap='magma',filternorm=False)
    ax.set_axis_off()
    fig.add_axes(ax)
    plt.savefig("{}".format(filename),bbox_inches='tight')
    plt.close()
    print(index,filename, "generated")

if __name__ == "__main__":
    arg = sys.argv[1]
    if arg == "extract_png":
        base_path = '/media/romulo/3363-3835/spectrograms_128_array/'
        metadata = pd.read_csv(base_path+'location_mapping.csv')
        with Pool(64) as thread_pool:
            idx_location = thread_pool.map(generate_img_from_npy,
                            metadata.iterrows())
