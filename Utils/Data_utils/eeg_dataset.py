import os
import torch
import numpy as np
import pandas as pd

from scipy.io import arff
from scipy import stats
from copy import deepcopy
from torch.utils.data import Dataset
from Utils.masking_utils import noise_mask
from Models.interpretable_diffusion.model_utils import normalize_to_neg_one_to_one, unnormalize_to_zero_to_one
from sklearn.preprocessing import MinMaxScaler


class EEGDataset(Dataset):
    def __init__(
        self, 
        data_root, 
        window=64, 
        save2npy=True, 
        neg_one_to_one=True,
        period='train',
        output_dir='./OUTPUT'
    ):
        super(EEGDataset, self).__init__()
        assert period in ['train', 'test'], 'period must be train or test.'

        self.auto_norm, self.save2npy = neg_one_to_one, save2npy
        self.data_0, self.data_1, self.scaler = self.read_data(data_root, window)
        self.labels = np.zeros(self.data_0.shape[0] + self.data_1.shape[0]).astype(np.int64)
        self.labels[self.data_0.shape[0]:] = 1
        self.rawdata = np.vstack([self.data_0, self.data_1])
        self.dir = os.path.join(output_dir, 'samples')
        os.makedirs(self.dir, exist_ok=True)

        self.window, self.period = window, period
        self.len, self.var_num = self.rawdata.shape[0], self.rawdata.shape[-1]

        self.samples = self.normalize(self.rawdata)

        # np.save(os.path.join(self.dir, 'eeg_ground_0_truth.npy'), self.data_0)
        # np.save(os.path.join(self.dir, 'eeg_ground_1_truth.npy'), self.data_1)

        self.sample_num = self.samples.shape[0]

    def read_data(self, filepath, length):
        """
        Reads the data from the given filepath, removes outliers, classifies the data into two classes,
        and scales the data using MinMaxScaler.

        Args:
            filepath (str): Path to the .arff file containing the EEG data.
            length (int): Length of the window for classification.
        """
        data = arff.loadarff(filepath)
        df = pd.DataFrame(data[0])
        df['eyeDetection'] = df['eyeDetection'].astype('int')

        df = self.__OutlierRemoval__(df)
        df_0, df_1 = self.__Classify__(df, length=length)
        # df_0.to_csv('./EEG_Eye_State_0.csv', index=False)
        # df_1.to_csv('./EEG_Eye_State_1.csv', index=False)

        data_0 = df_0.values.reshape(df_0.shape[0], length, -1)
        data_1 = df_1.values.reshape(df_1.shape[0], length, -1)

        # print(f"Class 0: {data_0.shape}, Class 1: {data_1.shape}")

        data = np.vstack([data_0.reshape(-1, data_0.shape[-1]), data_1.reshape(-1, data_1.shape[-1])])

        scaler = MinMaxScaler()
        scaler = scaler.fit(data)

        return data_0, data_1, scaler
    
    @staticmethod
    def __OutlierRemoval__(df):
        """
        Removes outliers from the dataframe using z-score method and interpolates the missing values.

        Args:
            df (pd.DataFrame): Dataframe containing the EEG data.

        Returns:
            pd.DataFrame: Cleaned dataframe with outliers removed and missing values interpolated.
        """
        temp_data_frame = deepcopy(df)
        clean_data_frame = deepcopy(df)
        for column in temp_data_frame.columns[:-1]:
            temp_data_frame[str(column)+'z_score'] = stats.zscore(temp_data_frame[column])
            clean_data_frame[column] = temp_data_frame.loc[temp_data_frame[str(column)+'z_score'].abs()<=3][column]

        clean_data_frame.interpolate(method='linear', inplace=True)

        temp_data_frame = deepcopy(clean_data_frame)
        clean_data_frame_second = deepcopy(clean_data_frame)

        for column in temp_data_frame.columns[:-1]:
            temp_data_frame[str(column)+'z_score'] = stats.zscore(temp_data_frame[column])
            clean_data_frame_second[column] = temp_data_frame.loc[temp_data_frame[str(column)+'z_score'].abs()<=3][column]

        clean_data_frame_second.interpolate(method='linear', inplace=True)
        return clean_data_frame
    
    @staticmethod
    def __Classify__(df, length=100):
        """
        Classifies the data into two classes based on the eyeDetection column and creates signals for the two classes.

        Args:
            df (pd.DataFrame): Dataframe containing the EEG data.
            length (int): Length of the window for classification.

        Returns:
            pd.DataFrame: Dataframe containing the signals for class 0.
            pd.DataFrame: Dataframe containing the signals for class 1.
        """
        # normalize the columns between -1 and 1
        mean, max, min = df.mean(), df.max(), df.min()
        df = 2*(df - mean) / (max - min)

        df['edge'] = df['eyeDetection'].diff()
        df['edge'][0] = 0.0

        starting = df['edge'][df['edge']==2]
        starting = starting.iloc[:-1]
        starting_time = starting.index.values
        ending = df['edge'][df['edge']==-2]
        ending_time = ending.index.values
        end_time_with_0 = np.insert(ending_time, 0, 0)
        
        signal_0 = []
        singal_1 = []

        # create signal 0
        for start, end in zip(end_time_with_0, starting_time):
            for i in range(start+50, end-length-50, 1):
                temp = []
                for channel in df.columns[:-2]:
                    temp.append(df[channel][i:i+length])

                signal_0.append(np.hstack(temp))

        # create signal 1
        for start, end in zip(starting_time, ending_time):
            for i in range(start+50, end-length-50, 1):
                temp = []
                for channel in df.columns[:-2]:
                    temp.append(df[channel][i:i+length])

                singal_1.append(np.hstack(temp))

        df_0 = pd.DataFrame(signal_0)
        df_1 = pd.DataFrame(singal_1)

        # min_samples = min(df_0.shape[0], df_1.shape[0]) - 1
        # # chop the data to the same length
        # df_0 = df_0.iloc[:min_samples, :]
        # df_1 = df_1.iloc[:min_samples, :]

        return df_0, df_1

    def __getitem__(self, ind):
        if self.period == 'test':
            x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
            y = self.labels[ind]  # (1,) int
            return torch.from_numpy(x).float(), torch.tensor(y)
        x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
        return torch.from_numpy(x).float()

    def __len__(self):
        return self.sample_num

    def normalize(self, sq):
        d = self.__normalize(sq.reshape(-1, self.var_num))
        data = d.reshape(-1, self.window, self.var_num)
        return data

    def __normalize(self, rawdata):
        data = self.scaler.transform(rawdata)
        if self.auto_norm:
            data = normalize_to_neg_one_to_one(data)
        return data

    def unnormalize(self, sq):
        d = self.__unnormalize(sq.reshape(-1, self.var_num))
        return d.reshape(-1, self.window, self.var_num)

    def __unnormalize(self, data):
        if self.auto_norm:
            data = unnormalize_to_zero_to_one(data)
        x = data
        return self.scaler.inverse_transform(x)
    
    def shift_period(self, period):
        assert period in ['train', 'test'], 'period must be train or test.'
        self.period = period