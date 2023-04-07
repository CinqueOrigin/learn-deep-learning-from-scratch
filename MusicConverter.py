import os
from midi2audio import FluidSynth
import wave
from pydub import AudioSegment
from tqdm import tqdm

def traverse_words_dir_recurrence(words_dir):
    file_list = []
    for root, h, files in os.walk(words_dir):
        for file in files:
            if file.endswith(".au"):
                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(words_dir)+1:]
                file_list.append(pure_path)
    return file_list

def convert_folder_to_wav(input_path,output_path=None):
    if output_path is None:
        output_path = input_path
    for root, dirs, files in os.walk(input_path): 
        for file in files:
            if file.endswith('.au'):
                midi_file_path = os.path.join(root, file)
                wav_file_path = os.path.join(output_path, file).replace('.mid','.wav')
                FluidSynth().midi_to_audio(midi_file_path, wav_file_path)


import pygame.midi
import pygame.mixer
def convert_midi_to_wav(midi_file_path,wav_file_path):
    # initialize pygame
    pygame.init()
    pygame.midi.init()
    pygame.mixer.init()

    # load the MIDI file
    # ...

    # convert the MIDI file to WAV using pygame
    pygame.mixer.music.load(midi_file_path)
    pygame.mixer.music.set_endevent(pygame.USEREVENT)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pygame.time.wait(100)
    pygame.mixer.music.stop()
    pygame.mixer.music.save(wav_file_path)

    # clean up pygame
    pygame.mixer.quit()
    pygame.midi.quit()
    pygame.quit()


def convert_Au_to_Wav(input_file,output_file):
    song = AudioSegment.from_file(input_file, format='au')
    song.export(output_file, format='wav')

def convert_Au_to_Wav_For_dir(input_path,output_path):
    file_list = traverse_words_dir_recurrence(input_path)
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    for file in tqdm(file_list):
        input_file = os.path.join(input_path, file)
        output_file = os.path.join(output_path,file).replace('.au', '.wav')
        output_pure_dir = os.path.join(output_path,file.split('/')[0])
        if not os.path.exists(output_pure_dir):
            os.mkdir(output_pure_dir)
        convert_Au_to_Wav(input_file,output_file)


if __name__ == '__main__':
    # convert_folder_to_wav('/data/tianjh/data/Dataset/midis')
    convert_Au_to_Wav_For_dir('/data/tianjh/data/Dataset/genres','/data/tianjh/data/Dataset/genreswavs')
    # convert_midi_to_wav('/data/tianjh/data/Dataset/midis/Q1__8v0MFBZoco_0.mid','Q1__8v0MFBZoco_0.wav')
