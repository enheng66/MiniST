import os
import pickle
import sys
import torch
import numpy as np
import threading
import multiprocessing as mp

class DataLoader(object):
    def __init__(self, data, idx, seq_len, horizon, bs, logger, pad_last_sample=False):
        if pad_last_sample:
            num_padding = (bs - (len(idx) % bs)) % bs
            idx_padding = np.repeat(idx[-1:], num_padding, axis=0)
            idx = np.concatenate([idx, idx_padding], axis=0)
        
        self.data = data
        self.idx = idx
        self.size = len(idx)
        self.bs = bs
        self.num_batch = int(self.size // self.bs)
        self.current_ind = 0
        logger.info('Sample num: ' + str(self.idx.shape[0]) + ', Batch num: ' + str(self.num_batch))
        
        self.x_offsets = np.arange(-(seq_len - 1), 1, 1)
        self.y_offsets = np.arange(1, (horizon + 1), 1)
        self.seq_len = seq_len
        self.horizon = horizon


    def shuffle(self):
        perm = np.random.permutation(self.size)
        idx = self.idx[perm]
        self.idx = idx


    def write_to_shared_array(self, x, y, idx_ind, start_idx, end_idx):
        for i in range(start_idx, end_idx):
            x[i] = self.data[idx_ind[i] + self.x_offsets, :, :]
            y[i] = self.data[idx_ind[i] + self.y_offsets, :, :1]


    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.bs * self.current_ind
                end_ind = min(self.size, self.bs * (self.current_ind + 1))
                idx_ind = self.idx[start_ind: end_ind, ...]

                x_shape = (len(idx_ind), self.seq_len, self.data.shape[1], self.data.shape[-1])
                x_shared = mp.RawArray('f', int(np.prod(x_shape)))
                x = np.frombuffer(x_shared, dtype='f').reshape(x_shape)

                y_shape = (len(idx_ind), self.horizon, self.data.shape[1], 1)
                y_shared = mp.RawArray('f', int(np.prod(y_shape)))
                y = np.frombuffer(y_shared, dtype='f').reshape(y_shape)

                array_size = len(idx_ind)
                num_threads = len(idx_ind) // 2
                chunk_size = array_size // num_threads
                threads = []
                for i in range(num_threads):
                    start_index = i * chunk_size
                    end_index = start_index + chunk_size if i < num_threads - 1 else array_size
                    thread = threading.Thread(target=self.write_to_shared_array, args=(x, y, idx_ind, start_index, end_index))
                    thread.start()
                    threads.append(thread)

                for thread in threads:
                    thread.join()

                yield (x, y)
                self.current_ind += 1

        return _wrapper()


class StandardScaler():
    def __init__(self, mean, std, device):
        self.mean = torch.tensor(mean).to(device)
        self.std = torch.tensor(std).to(device)
    
    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean

def generate_train_val_test(data_path,args):
    data = np.load(os.path.join(data_path, args.years, 'his.npz'))['data']
    # print('original data shape:', data.shape)  #sd 35040*716
    # sys.exit()
    seq_length_x, seq_length_y = args.seq_len, args.horizon
    x_offsets = np.arange(-(seq_length_x - 1), 1, 1)   #-11 0
    y_offsets = np.arange(1, (seq_length_y + 1), 1)    #1 12

    num_sample, _ , _ =data.shape 
    min_t = abs(min(x_offsets))
    max_t = abs(num_sample - abs(max(y_offsets)))  # Exclusive
    idx = np.arange(min_t, max_t, 1)   #

    num_samples = len(idx)
    num_train = round(num_samples * 0.6)
    num_val = round(num_samples * 0.2)
    # split idx
    idx_train = idx[:num_train]
    idx_val = idx[num_train: num_train + num_val]
    idx_test = idx[num_train + num_val:]

    #normalize
    data_mu=np.mean(data[:idx_val[0], :,0],axis=0, keepdims=True,dtype=np.float32) #(1, 716)
    data_sigma=np.std(data[:idx_val[0], :,0],axis=0, keepdims=True,dtype=np.float32)
    data_sigma[ data_sigma < 1e-6] = 1

    data[..., 0]= (data[..., 0] - data_mu) / data_sigma
    data_mu=data_mu.T
    data_sigma=data_sigma.T

    return data,data_mu,data_sigma,idx_train,idx_val,idx_test



def load_dataset(data_path, args, logger):
    ptr,mean,std,idx_train,idx_val,idx_test = generate_train_val_test(data_path,args)
    logger.info('Data shape: ' + str(ptr.shape))
    
    dataloader = {}
    for cat in ['train', 'val', 'test']:
        if cat=='train': idx=idx_train
        elif cat=='val':idx=idx_val
        else:idx=idx_test
        dataloader[cat + '_loader'] = DataLoader(ptr[..., :args.input_dim], idx, \
                                                 args.seq_len, args.horizon, args.bs, logger)
    scaler = StandardScaler(mean=mean, std=std, device=args.device)


    return dataloader, scaler



def load_adj_from_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def load_adj_from_numpy(numpy_file):
    return np.load(numpy_file)


def get_dataset_info(dataset):
    base_dir = os.getcwd() + '/data/'
    d = {
         'ca': [base_dir+'ca', base_dir+'ca/ca_rn_adj.npy', 8600],
         'gla': [base_dir+'gla', base_dir+'gla/gla_rn_adj.npy', 3834],
         'gba': [base_dir+'gba', base_dir+'gba/gba_rn_adj.npy', 2352],
         'sd': [base_dir+'sd', base_dir+'sd/sd_rn_adj.npy', 716],
        }
    assert dataset in d.keys()
    return d[dataset]


def metapath(dataset):
    base_dir = os.getcwd() + '/data/'

    d = {
         'sd': [base_dir + dataset + '/' + dataset + '_meta.csv'],
         'gba': [base_dir + dataset + '/' + dataset + '_meta.csv'],
         'gla': [base_dir + dataset + '/' + dataset + '_meta.csv'],
         'ca': [base_dir + dataset + '/' + dataset + '_meta.csv'],
        }
    assert dataset in d.keys()
    return d[dataset]

def get_dataset_nodetime_emb(dataset):
    d = {
         'ca':{
                "node_dim": 128,
                "embed_dim": 32,
                "num_layer": 4,
                "if_node": True,
                "if_T_i_D": True,
                "if_D_i_W": True,
                "temp_dim_tid": 32,
                "temp_dim_diw": 32,
                "time_of_day_size": 96,
                "day_of_week_size": 7},
         'gla':{
                "node_dim": 128,
                "embed_dim": 32,
                "num_layer": 4,
                "if_node": True,
                "if_T_i_D": True,
                "if_D_i_W": True,
                "temp_dim_tid": 32,
                "temp_dim_diw": 32,
                "time_of_day_size": 96,
                "day_of_week_size": 7},
         'gba':{
                "embed_dim": 32,
                "num_layer": 4,
                "if_node": True,
                "node_dim": 128,
                "if_T_i_D": True,
                "if_D_i_W": True,
                "temp_dim_tid": 32,
                "temp_dim_diw": 32,
                "time_of_day_size": 96,
                "day_of_week_size": 7},
         'sd':{
                "embed_dim": 32,
                "num_layer": 4,
                "if_node": True,
                "node_dim": 64,
                "if_T_i_D": True,
                "if_D_i_W": True,
                "temp_dim_tid": 32,
                "temp_dim_diw": 32,
                "time_of_day_size": 96,
                "day_of_week_size": 7},
        }
    assert dataset in d.keys()
    return d[dataset]
