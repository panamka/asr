import os
import glob
from tqdm.auto import tqdm
import pandas as pd
import soundfile as sf



def get_csv(root_utts):
    audio_files_utts = glob.glob(f'{root_utts}/**/*.wav', recursive=True)
    dict_audio = {}
    dict_text = {}

    for path in tqdm(audio_files_utts):
        with sf.SoundFile(path) as f:
            frames = f.frames
        dict_audio.update({path: frames})
        src_file = '-'.join(path.split('-')[:-1]) + '.trans.txt'
        idx = path.split('/')[-1].split('.')[0]
        with open(src_file, 'r') as fp:
            for line in fp:
                if idx == line.split(' ')[0]:
                    text = line[:-1].split(' ', 1)[1]
        dict_text.update({path: text})

    df_audio = pd.DataFrame.from_dict(dict_audio, orient='index').reset_index()
    df_audio.columns = ['path', 'frames']
    df_audio = df_audio.sort_values(by=['frames'])

    df_text = pd.DataFrame.from_dict(dict_text, orient='index').reset_index()
    df_text.columns = ['path', 'text']

    df = df_audio.merge(df_text)



    return df

def main():
    root_utts = '/home/huawei/Shared/Datasets/LibriSpeechWav'
    df = get_csv(root_utts)
    print(df.shape)



    df_test = df.sample(int(0.8*df.shape[0]))
    df_test = df_test.sort_values(by=['frames'])
    df_train = df[~df.index.isin(df_test.index)]
    df_train = df_train.sort_values(by=['frames'])


    df_train.to_csv('libri_data_train.csv', index=False)
    df_test.to_csv('libri_data_test.csv', index=False)


    df_train = pd.read_csv('libri_data_train.csv')
    print(df_train)
    # df_train = df_train.reset_index()[1☺
    # df_train.to_csv('libri_data_train.csv')
    #
    # df_test = pd.read_csv('libri_data_test.csv', names=['path', 'frames', 'text'])
    # df_test = df_test.reset_index()[1☺
    # df_test.to_csv('libri_data_train.csv')
    #
    # df_train = pd.read_csv('libri_data_train.csv', names=['path', 'frames', 'text'])
    # print(df_train)

if __name__ == '__main__':
    main()
