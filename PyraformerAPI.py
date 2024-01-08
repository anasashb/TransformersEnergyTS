import argparse

import numpy as np
import time
import torch
import torch.optim as optim

import Pyraformer.pyraformer.Pyraformer_LR as Pyraformer
from Pyraformer.long_range_main import *
from tqdm import tqdm
from Pyraformer.data_loader import *
from Pyraformer.utils.tools import TopkMSELoss, metric


###########################################################################################################################
# Our Simple User Interface ###############################################################################################
class PyraformerTS():
    '''
    Our custom wrapper class (in progress) to provide an user-friendly interface to fitting and testing the Informer model. 
    The class will be extended to accomodate other models within the scope of the project.
    For simplicity of use, methods included align with naming used by Keras.      
    '''
    def __init__(self, model='pyraformer'):
        if model != 'pyraformer':
            raise ValueError("Model not supported. Please use 'pyraformer'.")
        # running mode
        #parser.add_argument('-eval', action='store_true', default=False)
        self.eval = False
        # Path parameters
        #parser.add_argument('-data', type=str, default='ETTh1')
        #parser.add_argument('-root_path', type=str, default='./data/ETT/', help='root path of the data file')
        #parser.add_argument('-data_path', type=str, default='ETTh1.csv', help='data file')
        self.data = "SYNTHh1"
        self.root_path = "./SYNTHDataset/"
        self.data_path = "SYNTHh1.csv"
        # Dataloader parameters.
        #parser.add_argument('-input_size', type=int, default=168)
        #parser.add_argument('-predict_step', type=int, default=168)
        #parser.add_argument('-inverse', action='store_true', help='denormalize output data', default=False)
        self.input_size = 168
        self.predict_step = 168
        self.inverse = False
        # Architecture selection.
        #parser.add_argument('-model', type=str, default='Pyraformer')
        #parser.add_argument('-decoder', type=str, default='FC') # selection: [FC, attention]
        self.model = "Pyraformer"
        self.decoder = "FC"
        # Training parameters.
        #parser.add_argument('-epoch', type=int, default=5)
        #parser.add_argument('-batch_size', type=int, default=32)
        #parser.add_argument('-pretrain', action='store_true', default=False)
        #parser.add_argument('-hard_sample_mining', action='store_true', default=False)
        #parser.add_argument('-dropout', type=float, default=0.05)
        #parser.add_argument('-lr', type=float, default=1e-4)
        #parser.add_argument('-lr_step', type=float, default=0.1)
        self.epoch = 5
        self.batch_size = 32
        self.pretrain = False
        self.hard_sample_mining = False
        self.dropout = 0.05
        self.lr = 1e-4
        self.lr_step = 0.1


        # Common Model parameters.
        #parser.add_argument('-d_model', type=int, default=512)
        #parser.add_argument('-d_inner_hid', type=int, default=512)
        #parser.add_argument('-d_k', type=int, default=128)
        #parser.add_argument('-d_v', type=int, default=128)
        #parser.add_argument('-d_bottleneck', type=int, default=128)
        #parser.add_argument('-n_head', type=int, default=4)
        #parser.add_argument('-n_layer', type=int, default=4)
        self.d_model = 512
        self.d_inner_hid = 512
        self.d_k = 128
        self.d_v = 128
        self.d_bottleneck = 128
        self.n_head = 6
        self.n_layer = 4

        # Pyraformer parameters.
        #parser.add_argument('-window_size', type=str, default='[4, 4, 4]') # The number of children of a parent node.
        #parser.add_argument('-inner_size', type=int, default=3) # The number of ajacent nodes.
        # CSCM structure. selection: [Bottleneck_Construct, Conv_Construct, MaxPooling_Construct, AvgPooling_Construct]
        #parser.add_argument('-CSCM', type=str, default='Bottleneck_Construct')
        #parser.add_argument('-truncate', action='store_true', default=False) # Whether to remove coarse-scale nodes from the attention structure
        #parser.add_argument('-use_tvm', action='store_true', default=False) # Whether to use TVM.
        self.window_size = [4,4,4]
        self.inner_size = 3
        self.CSCM = "Bottleneck_Construct"
        self.truncate = False
        self.use_tvm = False
        # Experiment repeat times.
        #parser.add_argument('-iter_num', type=int, default=5) # Repeat number.
        self.iter_num = 5
        #opt = parser.parse_args()
        #return opt
        self.device = "cuda"

        parser = argparse.ArgumentParser()

        datapars = dataset_parameters(parser,self.data)

        self.enc_in =parser.enc_in
        self.dec_in =parser.dec_in
        self.covariate_size = parser.covariate_size
        self.seq_num = parser.seq_num
        self.embed_type = parser.embed_type

    def compile(self, learning_rate=1e-4, loss='mse', early_stopping_patience=3):
        '''
        Compiles the Fedformer model for training.
        Args:
            learning_rate (float): Learning rate to be used. Default: '1e-4'.
            loss (str): Loss function to be used. Default: 'mse'.
            early_stopping_patience (int): Amount of epochs to beak training loop after no validation performance improvement. Default: 3.
        '''
        if loss != 'mse':
            raise ValueError("Loss function not supported. Please use 'mse'.")
        self.lr = learning_rate
        self.loss = loss
        self.patience = early_stopping_patience

    def fit(self, data = "SYNTHh1", data_root_path = "./SYNTHDataset/", batch_size = 32 , epochs = 5 , pred_len = 24):
        """ Main function. """
        #print('[Info] parameters: {}'.format(self))

                # temporary line
        possible_datasets = ['SYNTHh1', 'SYNTHh2', 'SYNTH_additive' , 'SYNTH_multiplicative', 'DEWINDh_large', 'DEWINDh_small']
        if data not in possible_datasets:
            raise ValueError("Dataset not supported. Please use one of the following: 'SYNTHh1', 'SYNTHh2', SYNTH_additive', 'SYNTH_multiplicative', 'DEWINDh_large', 'DEWINDh_small'.")
        # temporary line
        possible_predlens = [24, 48, 96, 168, 336, 720]
        if pred_len not in possible_predlens:
            raise ValueError('Prediction length outside current experiment scope. Please use either 24, 48, 96, 168, 336, 720.')
        self.data = data
        self.root_path = data_root_path
        self.data_path = f'{self.data}.csv'
        self.epoch = epochs
        self.batch_size = batch_size
        self.predict_step = pred_len

        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device('cpu')
        """ prepare model """
        model = eval(self.model).Model(self)

        model.to(self.device)

        """ number of parameters """
        num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print('[Info] Number of parameters: {}'.format(num_params))

        """ train or evaluate the model """
        model_save_dir = 'models/LongRange/{}/{}/'.format(self.data, self.predict_step)
        os.makedirs(model_save_dir, exist_ok=True)
        model_save_dir += 'best_iter.pth'
        if self.eval:
            best_metrics = evaluate(model, self, model_save_dir,1)
        else:
            """ optimizer and scheduler """
            optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), self.lr)
            scheduler = optim.lr_scheduler.StepLR(optimizer, 1, gamma=self.lr_step)
            best_metrics = train(model, optimizer, scheduler, self, model_save_dir,1)

        print('Iteration best metrics: {}'.format(best_metrics))
        return best_metrics
    
    def predict(self):
        """ Epoch operation in evaluation phase for returning predictions only. """
        model_save_dir = 'models/LongRange/{}/{}/'.format(self.data, self.predict_step)
        """ prepare dataloader """
        self.batch_size = 1
        _, _, test_dataloader, test_dataset = prepare_dataloader(self)
        model = eval(self.model).Model(self)
        model.eval()
        print(test_dataset.seq_len,test_dataset.pred_len)
        preds = []
        with torch.no_grad():
            for batch in tqdm(test_dataloader, mininterval=1, desc='  - (Validation) ', leave=False):
                """ prepare data """
                batch_x, batch_y, batch_x_mark, batch_y_mark, mean, std = map(lambda x: x.float().to(self.device), batch)
                dec_inp = torch.zeros_like(batch_y).float()

                # forward
                if self.decoder == 'FC':
                    # Add a predict token into the history sequence
                    predict_token = torch.zeros(batch_x.size(0), 1, batch_x.size(-1), device=self.device)
                    batch_x = torch.cat([batch_x, predict_token], dim=1)
                    batch_x_mark = torch.cat([batch_x_mark, batch_y_mark[:, 0:1, :]], dim=1)
                outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark, False)

                # if inverse, the output is denormalized
                if self.inverse:
                    outputs = test_dataset.inverse_transform(outputs, mean, std)

                preds.append(outputs.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        return preds
