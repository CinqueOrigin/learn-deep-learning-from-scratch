import os
import librosa
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
from model import CNNForClassification
from torch.nn.parallel import DataParallel
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def extract_audio_features(directory):
    '''
    This function takes in a directory of .wav files and returns a 
    DataFrame that includes several numeric features of the audio file 
    as well as the corresponding genre labels.
    
    The numeric features incuded are the first 13 mfccs, zero-crossing rate, 
    spectral centroid, and spectral rolloff.
    
    Parameters:
    directory (int): a directory of audio files in .wav format
    
    Returns:
    df (DataFrame): a table of audio files that includes several numeric features 
    and genre labels.
    '''
    
    # Creating an empty list to store all file names
    files = []
    labels = []
    zcrs = []
    spec_centroids = []
    spec_rolloffs = []
    mfccs_1 = []
    mfccs_2 = []
    mfccs_3 = []
    mfccs_4 = []
    mfccs_5 = []
    mfccs_6 = []
    mfccs_7 = []
    mfccs_8 = []
    mfccs_9 = []
    mfccs_10 = []
    mfccs_11 = []
    mfccs_12 = []
    mfccs_13 = []
    
    # Looping through each file in the directory
    for file in os.scandir(directory):
        
        # Loading in the audio file
        y, sr = librosa.core.load(file)
        
        # Adding the file to our list of files
        files.append(file)
        
        # Adding the label to our list of labels
        label = str(file).split('.')[0]
        labels.append(label)
        
        # Calculating zero-crossing rates
        zcr = librosa.feature.zero_crossing_rate(y)
        zcrs.append(np.mean(zcr))
        
        # Calculating the spectral centroids
        spec_centroid = librosa.feature.spectral_centroid(y)
        spec_centroids.append(np.mean(spec_centroid))
        
        # Calculating the spectral rolloffs
        spec_rolloff = librosa.feature.spectral_rolloff(y)
        spec_rolloffs.append(np.mean(spec_rolloff))
        
        # Calculating the first 13 mfcc coefficients
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=512, n_mfcc=13)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        mfccs_1.append(mfcc_scaled[0])
        mfccs_2.append(mfcc_scaled[1])
        mfccs_3.append(mfcc_scaled[2])
        mfccs_4.append(mfcc_scaled[3])
        mfccs_5.append(mfcc_scaled[4])
        mfccs_6.append(mfcc_scaled[5])
        mfccs_7.append(mfcc_scaled[6])
        mfccs_8.append(mfcc_scaled[7])
        mfccs_9.append(mfcc_scaled[8])
        mfccs_10.append(mfcc_scaled[9])
        mfccs_11.append(mfcc_scaled[10])
        mfccs_12.append(mfcc_scaled[11])
        mfccs_13.append(mfcc_scaled[12])
    
    # Creating a data frame with the values we collected
    df = pd.DataFrame({
        'files': files,
        'zero_crossing_rate': zcrs,
        'spectral_centroid': spec_centroids,
        'spectral_rolloff': spec_rolloffs,
        'mfcc_1': mfccs_1,
        'mfcc_2': mfccs_2,
        'mfcc_3': mfccs_3,
        'mfcc_4': mfccs_4,
        'mfcc_5': mfccs_5,
        'mfcc_6': mfccs_6,
        'mfcc_7': mfccs_7,
        'mfcc_8': mfccs_8,
        'mfcc_9': mfccs_9,
        'mfcc_10': mfccs_10,
        'mfcc_11': mfccs_11,
        'mfcc_12': mfccs_12,
        'mfcc_13': mfccs_13,
        'labels': labels
    })
    
    # Returning the data frame
    return df


def traverse_words_dir_recurrence(words_dir):
    file_list = []
    for root, h, files in os.walk(words_dir):
        for file in files:
            if file.endswith(".au"):
                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(words_dir)+1:]
                file_list.append(pure_path)
    return file_list

def make_mel_spectrogram_df(directory,mel_x,mel_y):
    '''
    This function takes in a directory of audio files in .wav format, computes the
    mel spectrogram for each audio file, reshapes them so that they are all the 
    same size, flattens them, and stores them in a dataframe.
    
    Genre labels are also computed and added to the dataframe.
    
    Parameters:
    directory (int): a directory of audio files in .wav format
    
    Returns:
    df (DataFrame): a dataframe of flattened mel spectrograms and their 
    corresponding genre labels
    '''
    
    # Creating empty lists for mel spectrograms and labels
    labels = []
    mel_specs = []
    file_list = traverse_words_dir_recurrence(directory)
    # Looping through each file in the directory
    for file in tqdm(file_list):
        print('filename',file)
        # exit()
        if not file.endswith('.au'):
            continue
        # Loading in the audio file
        y, sr = librosa.core.load(os.path.join(directory,file))
        
        # Extracting the label and adding it to the list
        label = str(file.split('/')[0])
        labels.append(label)
        
        # Computing the mel spectrograms
        spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
        spect = librosa.power_to_db(spect, ref=np.max)
        
        # Adjusting the size to be 128 x 660
        if spect.shape[1] != mel_y:
            spect.resize(mel_x,mel_y, refcheck=False)
            # print('resizing spectrogram')
        
        # Flattening to fit into dataframe and adding to the list
        spect = spect.flatten()
        mel_specs.append(spect)
        
    # Converting the lists to arrays so we can stack them
    mel_specs = np.array(mel_specs)
    labels = np.array(labels).reshape(len(labels),1)
    
    # Create dataframe
    df = pd.DataFrame(np.hstack((mel_specs,labels)))
    
    # Returning the mel spectrograms and labels
    return df

def extract_audio_features(directory):
    '''
    This function takes in a directory of .wav files and returns a 
    DataFrame that includes several numeric features of the audio file 
    as well as the corresponding genre labels.
    
    The numeric features incuded are the first 13 mfccs, zero-crossing rate, 
    spectral centroid, and spectral rolloff.
    
    Parameters:
    directory (int): a directory of audio files in .wav format
    
    Returns:
    df (DataFrame): a table of audio files that includes several numeric features 
    and genre labels.
    '''
    
    # Creating an empty list to store all file names
    files = []
    labels = []
    zcrs = []
    spec_centroids = []
    spec_rolloffs = []
    mfccs_1 = []
    mfccs_2 = []
    mfccs_3 = []
    mfccs_4 = []
    mfccs_5 = []
    mfccs_6 = []
    mfccs_7 = []
    mfccs_8 = []
    mfccs_9 = []
    mfccs_10 = []
    mfccs_11 = []
    mfccs_12 = []
    mfccs_13 = []
    
    # Looping through each file in the directory
    for file in os.scandir(directory):
        
        # Loading in the audio file
        y, sr = librosa.core.load(file)
        
        # Adding the file to our list of files
        files.append(file)
        
        # Adding the label to our list of labels
        label = str(file).split('.')[0]
        labels.append(label)
        
        # Calculating zero-crossing rates
        zcr = librosa.feature.zero_crossing_rate(y)
        zcrs.append(np.mean(zcr))
        
        # Calculating the spectral centroids
        spec_centroid = librosa.feature.spectral_centroid(y)
        spec_centroids.append(np.mean(spec_centroid))
        
        # Calculating the spectral rolloffs
        spec_rolloff = librosa.feature.spectral_rolloff(y)
        spec_rolloffs.append(np.mean(spec_rolloff))
        
        # Calculating the first 13 mfcc coefficients
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=512, n_mfcc=13)
        mfcc_scaled = np.mean(mfcc.T, axis=0)
        mfccs_1.append(mfcc_scaled[0])
        mfccs_2.append(mfcc_scaled[1])
        mfccs_3.append(mfcc_scaled[2])
        mfccs_4.append(mfcc_scaled[3])
        mfccs_5.append(mfcc_scaled[4])
        mfccs_6.append(mfcc_scaled[5])
        mfccs_7.append(mfcc_scaled[6])
        mfccs_8.append(mfcc_scaled[7])
        mfccs_9.append(mfcc_scaled[8])
        mfccs_10.append(mfcc_scaled[9])
        mfccs_11.append(mfcc_scaled[10])
        mfccs_12.append(mfcc_scaled[11])
        mfccs_13.append(mfcc_scaled[12])
    
    # Creating a data frame with the values we collected
    df = pd.DataFrame({
        'files': files,
        'zero_crossing_rate': zcrs,
        'spectral_centroid': spec_centroids,
        'spectral_rolloff': spec_rolloffs,
        'mfcc_1': mfccs_1,
        'mfcc_2': mfccs_2,
        'mfcc_3': mfccs_3,
        'mfcc_4': mfccs_4,
        'mfcc_5': mfccs_5,
        'mfcc_6': mfccs_6,
        'mfcc_7': mfccs_7,
        'mfcc_8': mfccs_8,
        'mfcc_9': mfccs_9,
        'mfcc_10': mfccs_10,
        'mfcc_11': mfccs_11,
        'mfcc_12': mfccs_12,
        'mfcc_13': mfccs_13,
        'labels': labels
    })
    
    # Returning the data frame
    return df

def check_file_size(input_dir):
    # Creating an empty list to store sizes in
    sizes = []

    # Looping through each audio file
    for file in os.scandir(input_dir):
            
        # Loading in the audio file
        y, sr = librosa.core.load(file)
            
        # Computing the mel spectrograms
        spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
        spect = librosa.power_to_db(spect, ref=np.max)
        
        # Adding the size to the list
        sizes.append(spect.shape)
        
    # Checking if all sizes are the same
    print(f'The sizes of all the mel spectrograms in our data set are equal: {len(set(sizes)) == 1}')
    print(f'The maximum size is: {max(sizes)}')
    print(sizes)    
    if not len(set(sizes)) == 1:
        raise('The size of all the mel spect are not equal')
    # Checking the max size


def prepare_data(input_dir,output_csv,mel_x,mel_y):
    print('Preparing data for training and testing')
    #0.check if the input sizes are the same
    # check_file_size(input_dir)
    #1.extract_mel_spectrogram from wav file,and convert them to DataFrame
    mel_spectrogram_df = make_mel_spectrogram_df(input_dir,mel_x,mel_y)
    dataset = mel_spectrogram_df
    print('finished converting audio samples to mel spectrogram')
    #2.creating labels from csv
    # dataset['files'] = dataset['files'].map(lambda x: x[3:])
    dataset = dataset.rename(columns={mel_x*mel_y: 'labels'})
    label_dict = {
        'jazz': 0,
        'reggae': 1,
        'rock': 2,
        'blues': 3,
        'hiphop': 4,
        'country': 5,
        'metal': 6,
        'classical': 7,
        'disco': 8,
        'pop': 9
    }
    dataset['y'] = dataset['labels'].map(label_dict)
    #3.save as csv file
    print('converting mel spectrum to csv')
    with tqdm(total=len(dataset)) as pbar:
        dataset.to_csv(output_csv)
        pbar.update(len(dataset))
    print('finished loading dataset and converting mel spectrum into csv')
    return dataset


class MelDataset(Dataset):
    def __init__(self, texts, labels):
        self.x = texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = torch.tensor(self.x[index],dtype=torch.float32)
        label = self.labels[index]
        label=int(label) 
        label = torch.tensor(label,dtype=torch.long)
        return x,label

def train(model, train_loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    for i, (x, labels) in tqdm(enumerate(train_loader)):
        x = x.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        x =x.permute(0,3,2,1)
        outputs = model(x)
        outputs2 = torch.squeeze(outputs)  # 去掉不必要的维度
        preds = torch.argmax(outputs2, dim=1)  # 找到最大预测值对应的类别
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * x.size(0)
        total_acc += preds.eq(labels.view_as(preds)).sum().item()
        # a=1
    return total_loss / len(train_loader.dataset), total_acc / len(train_loader.dataset)

def evaluate(model, val_loader, criterion):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    with torch.no_grad():
        for i, (x, labels) in tqdm(enumerate(val_loader)):
            x = x.to(device)
            labels = labels.to(device)
            # optimizer.zero_grad()
            x =x.permute(0,3,2,1)
            outputs = model(x)
            outputs2 = torch.squeeze(outputs)  # 去掉不必要的维度
            preds = torch.argmax(outputs2, dim=1)  # 找到最大预测值对应的类别
            loss = criterion(outputs, labels)
            # loss.backward()
            # optimizer.step()
            total_loss += loss.item() * x.size(0)
            total_acc += preds.eq(labels.view_as(preds)).sum().item()
        return total_loss / len(val_loader.dataset), total_acc / len(val_loader.dataset)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--input_file', type=str,default='/data/tianjh/data/Dataset/genres',)
    parser.add_argument('--output_csv', type=str,default='/data/tianjh/data/Dataset/genres/genre.csv')
    parser.add_argument('--train_batch_size', type=int,default=24)
    parser.add_argument('--valid_batch_size', type=int,default=24)

    mel_x=128
    mel_y=660
    
    args = parser.parse_args()
    print('runing data processing')
    #1.load data from dir or csv file
    if args.output_csv.endswith('.csv') and os.path.exists(args.output_csv):
        print('loading csv from existing {}'.format(args.output_csv))
        dataset = pd.read_csv(args.output_csv)
    else:
        print('loading data input dir')
        dataset = prepare_data(args.input_file,args.output_csv,mel_x,mel_y)
    # print(dataset.keys())
    #2.split the dataset
    X = np.array(dataset.iloc[:,:84480])
    y = np.array(dataset['y'])
    print('try to split dataset')
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, stratify=y, test_size=.2)
    print('the minimum value of the training dataset is {}'.format(X_train.min()))
    X_train = X_train.reshape(X_train.shape[0], mel_x, mel_y, 1)
    X_test = X_test.reshape(X_test.shape[0], mel_x, mel_y, 1)
    train_dataset = MelDataset(X_train, y_train)
    val_dataset = MelDataset(X_test,y_test)
    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.valid_batch_size, shuffle=False)

    model = CNNForClassification(mel_x,mel_y,num_labels=10)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    num_epochs = 1000
    best_acc =0
    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion)
        val_loss, val_acc = evaluate(model, val_loader, criterion)
        if val_acc>best_acc:
            best_acc = val_acc
        print(f'Epoch {epoch+1} Train Loss: {train_loss:.4f} Train Acc: {train_acc:.4f} Val Loss: {val_loss:.4f} Val Acc: {val_acc:.4f}')
    print('the best val acc is: ',best_acc)
