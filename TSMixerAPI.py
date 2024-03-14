import os
import argparse
import glob
import logging
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers

### TSMixer API
###########################################################################################################################
###########################################################################################################################
def drop_last_for_tensorflow(df, batch_size, seq_len, pred_len):
    '''
    Emulates PyTorch dataloaders' option for drop_last = True
    '''
    total_length = len(df) - (seq_len + pred_len - 1)
    excess = total_length % batch_size
    if excess > 0:
        adjusted_length = len(df) - excess
        df = df.iloc[:adjusted_length]
    return df


# Metrics #################################################################################################################
### ALL METRICS ARE INTERCHANGEABLE WITH INFORMER AUTOFORMER
def RSE(pred, true):
    '''
    Calculates relative quared error.
    '''
    return np.sqrt(np.sum((true-pred)**2)) / np.sqrt(np.sum((true-true.mean())**2))

def CORR(pred, true):
    '''
    Calculates correlation coefficient.
    '''
    u = ((true-true.mean(0))*(pred-pred.mean(0))).sum(0) 
    d = np.sqrt(((true-true.mean(0))**2*(pred-pred.mean(0))**2).sum(0))
    return (u/d).mean(-1)

def MAE(pred, true):
    '''
    Calculates mean absolute error.
    '''
    return np.mean(np.abs(pred-true))

def MSE(pred, true):
    '''
    Calculates mean squared error.
    '''
    return np.mean((pred-true)**2)

def RMSE(pred, true):
    '''
    Calculates root mean suared error.
    '''
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true):
    '''
    Calculates mean absolute percentage error.
    '''
    return np.mean(np.abs((pred - true) / true))

def MSPE(pred, true):
    '''
    Calculates mean squared percentage error.
    '''
    return np.mean(np.square((pred - true) / true))

def metric(pred, true):
    '''
    Wraps up metric functions, calculates and returns all.
    '''
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    
    return mae,mse,rmse,mape,mspe


# Dot dictionary ##########################################################################################################
class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__



###########################################################################################################################
#  Data loader and dependencies ###########################################################################################  
class TSFDataLoader:
    """Generate data loader from raw data."""

    def __init__(
            self, root_path, batch_size, seq_len, pred_len, data_path='SYNTHh.csv', features='S', target='TARGET'
            ):
        self.root_path = root_path
        self.data_path = data_path
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.features = features
        self.target = target
        self.target_slice = slice(0, None)

        self._read_data()

    def _read_data(self):
        """Load raw data and split datasets."""
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        # S: univariate-univariate, M: multivariate-multivariate, MS:
        # multivariate-univariate
        df = df_raw.set_index('date')
        
        if self.features == 'S':
            df = df[[self.target]]
        elif self.features == 'MS': ## TODO check how this functions with multivariate once we have it
            target_idx = df.columns.get_loc(self.target)
            self.target_slice = slice(target_idx, target_idx + 1)

        # split train/valid/test
        n = len(df)
        # THE SPLITS below match the splits of Informer, Crossformer, Autoformer, Fedformer
        if self.data_path.startswith('ETTh'): # keeping this here bc we wanna include ETTh
            train_end = 12 * 30 * 24
            val_end = train_end + 4 * 30 * 24
            test_end = val_end + 4 * 30 * 24
        # I added two more elifs for synth and wind data, we can do the split provision here too
        elif self.data_path.startswith('SYNTHh'):
            train_end = 18 * 30 * 24
            val_end = train_end + 3 * 30 * 24
            test_end = val_end + 3 * 30 * 24
        elif self.data_path.startswith('DEWINDh'):
            train_end = 18 * 30 * 24
            val_end = train_end + 3 * 30 * 24
            test_end = val_end + 3 * 30 * 24
        else: # results to the good old train-val-test split by ratios
            train_end = int(n * 0.7)
            val_end = n - int(n * 0.2)
            test_end = n

        train_df = df[:train_end]
        val_df = df[train_end - self.seq_len : val_end]
        test_df = df[val_end - self.seq_len : test_end]
       
        # Drop last (if incomplete) batches
        train_df = drop_last_for_tensorflow(train_df, self.batch_size, self.seq_len, self.pred_len)
        val_df = drop_last_for_tensorflow(val_df, self.batch_size, self.seq_len, self.pred_len)
        test_df = drop_last_for_tensorflow(test_df, self.batch_size, self.seq_len, self.pred_len)
        
        # standardize by training set
        self.scaler = StandardScaler()
        self.scaler.fit(train_df.values)

        def scale_df(df, scaler):
            data = scaler.transform(df.values)
            return pd.DataFrame(data, index=df.index, columns=df.columns)

        self.train_df = scale_df(train_df, self.scaler)
        self.val_df = scale_df(val_df, self.scaler)
        self.test_df = scale_df(test_df, self.scaler)
        self.n_feature = self.train_df.shape[-1]

    def _split_window(self, data):
        inputs = data[:, : self.seq_len, :]
        labels = data[:, self.seq_len :, self.target_slice]
        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.seq_len, None])
        labels.set_shape([None, self.pred_len, None])
        return inputs, labels

    def _make_dataset(self, data, shuffle=True):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.utils.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=(self.seq_len + self.pred_len),
            sequence_stride=1, # window stride
            shuffle=shuffle,
            batch_size=self.batch_size,
            )
        ds = ds.map(self._split_window)
        return ds

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

    def get_train(self, shuffle=True):
        return self._make_dataset(self.train_df, shuffle=shuffle)

    def get_val(self):
        return self._make_dataset(self.val_df, shuffle=False)

    def get_test(self):
        return self._make_dataset(self.test_df, shuffle=False)


###########################################################################################################################
# Reversible Instance Normalization #######################################################################################
class RevNorm(layers.Layer):
    """Reversible Instance Normalization."""

    def __init__(self, axis, eps=1e-5, affine=True):
        super().__init__()
        self.axis = axis
        self.eps = eps
        self.affine = affine

    def build(self, input_shape):
        if self.affine:
            self.affine_weight = self.add_weight(
               'affine_weight', shape=input_shape[-1], initializer='ones'
               )
            self.affine_bias = self.add_weight(
               'affine_bias', shape=input_shape[-1], initializer='zeros'
               )

    def call(self, x, mode, target_slice=None):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x, target_slice)
        else:
            raise NotImplementedError
        return x

    def _get_statistics(self, x):
        self.mean = tf.stop_gradient(
           tf.reduce_mean(x, axis=self.axis, keepdims=True)
           )
        self.stdev = tf.stop_gradient(
           tf.sqrt(
              tf.math.reduce_variance(x, axis=self.axis, keepdims=True) + self.eps
              )
            )

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x, target_slice=None):
        if self.affine:
           x = x - self.affine_bias[target_slice]
           x = x / self.affine_weight[target_slice]
        x = x * self.stdev[:, :, target_slice]
        x = x + self.mean[:, :, target_slice]
        return x
  

###########################################################################################################################
# TSMIxer Block ###########################################################################################################
def res_block(inputs, norm_type, activation, dropout, ff_dim):
    """Residual block of TSMixer."""

    norm = (
       layers.LayerNormalization if norm_type == 'L' else layers.BatchNormalization
       )

    # Temporal Linear
    x = norm(axis=[-2, -1])(inputs)
    x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Channel, Input Length]
    x = layers.Dense(x.shape[-1], activation=activation)(x)
    x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Input Length, Channel]
    x = layers.Dropout(dropout)(x)
    res = x + inputs

  # Feature Linear
    x = norm(axis=[-2, -1])(res)
    x = layers.Dense(ff_dim, activation=activation)(
       x
    )  # [Batch, Input Length, FF_Dim]
    x = layers.Dropout(dropout)(x)
    x = layers.Dense(inputs.shape[-1])(x)  # [Batch, Input Length, Channel]
    x = layers.Dropout(dropout)(x)
    return x + res    
  
###########################################################################################################################
# Build TSMixer with Reversible Instance Normalization ####################################################################
def build_model(
      input_shape,
      pred_len,
      norm_type,
      activation,
      n_block,
      dropout,
      ff_dim,
      target_slice,
    ):
    
    """Build TSMixer with Reversible Instance Normalization model."""

    inputs = tf.keras.Input(shape=input_shape)
    x = inputs  # [Batch, Input Length, Channel]
    rev_norm = RevNorm(axis=-2)
    x = rev_norm(x, 'norm')
    for _ in range(n_block):
        x = res_block(x, norm_type, activation, dropout, ff_dim)

    if target_slice:
        x = x[:, :, target_slice]

    x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Channel, Input Length]
    x = layers.Dense(pred_len)(x)  # [Batch, Channel, Output Length]
    outputs = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Output Length, Channel])
    outputs = rev_norm(outputs, 'denorm', target_slice)
    return tf.keras.Model(inputs, outputs)


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # FATAL
logging.getLogger('tensorflow').setLevel(logging.FATAL)

class TSMixer():
    '''
    I am so thoroughly exhausted you cannot imagine
    '''
    
    def __init__(self, model='tsmixer_rev_in'):
        self.args = dotdict()
        self.args.model = model ## I keep these redundant model args to maybe then combine all API-s
        self.args.seed = 100
        
        ## Variables for Multivariate ##################################################################
        ## TODO ## 
        ## I need to take this outside in a method called get_data maybe?
        ## Choices for this and other API-s must be ['S', 'M', 'MS] <-- check if Crossformer can handle
        self.args.features = 'S' ## currently tailored to synth and wind series
        self.args.target = 'TARGET' ## Because I give this name to the column
        ################################################################################################
        
        
        self.args.checkpoints = './checkpoints'
        self.args.delete_checkpoint = False ## I am not sure this is correct default

        ## Variables for TS
        self.args.seq_len = 168 # used the default of other models, authors set it to 336

        # Model Architecture
        #self.kernel_size = 4 ## deactivated because we do not fit CNN
        self.args.n_block = 2 ## number of blocks for deep architecture
        self.args.ff_dim = 2048 ## fully-connected feature dimension
        self.args.dropout = 0.05 ## dropout rate
        self.args.norm_type = 'B' ## BatchNorm. Authors included alternative -- 'L' LayerNorm
        self.args.activation = 'relu' ## Authors included possible alternative -- 'gelu'
        self.args.temporal_dim = 16 ## temporal feature dimension
        self.args.hidden_dim = 64 ## hidden feature dimension
        self.args.num_workers = 0 
        self.args.itr = 3



        # Add root_path, data_path as args. 
        # root path serves as LOCAL_CACHE_DIR
        # data_path serves as data + '.csv'


    def compile(self, learning_rate=1e-4, loss='mse', early_stopping_patience=5):
        ## should include
        ## loss, 
        if loss != 'mse':
            raise ValueError("Loss function not supported. Please use 'mse'.")
        self.args.loss = loss
        self.args.learning_rate = learning_rate
        self.args.patience = early_stopping_patience

    
    def fit(self, data='SYNTHh', data_root_path='./SynthDataset/', batch_size=32, epochs=100, pred_len=24 , 
            seq_len = 168 , features = 'S' , target = 'TARGET' , iter = 1):
        ## Should include
        ## data, data_root_path, batch_size, epochs, pred_len
        possible_predlens = [24, 48, 96, 168, 336, 720]
        if pred_len not in possible_predlens:
            raise ValueError('Prediction length outside current experiment scope. Please use either 24, 48, 96, 168, 336, 720.')
        self.args.data = data ## NOTE this is redundant because the self.args.data in the other wrappers is used because it is needed for data_parser. Here parsing happens inside the data loader.
        self.args.root_path = data_root_path
        self.args.data_path = f'{self.args.data}.csv'
        self.args.pred_len = pred_len
        self.args.batch_size = batch_size ## 32 is the authors' default
        self.args.train_epochs = epochs ## 100 is the authors' default
        self.args.seq_len = seq_len
        self.args.iter = iter
        self.args.features = features
        self.args.target = target

        print('Beginning to fit the model with the following arguments:')
        print(f'{self.args}')
        print('='*150)  

        self.setting = f'TSMixer_{self.args.data}_{self.args.features}_sl{self.args.seq_len}_pl{self.args.pred_len}_lr{self.args.learning_rate}_nt{self.args.norm_type}_{self.args.activation}_nb{self.args.n_block}_dp{self.args.dropout}_fd{self.args.ff_dim}_iter{self.args.iter}'
        
        tf.keras.utils.set_random_seed(self.args.seed)
        
        # Initialize the data loader
        data_loader = TSFDataLoader(
            root_path=self.args.root_path,
            batch_size=self.args.batch_size,
            seq_len=self.args.seq_len,
            pred_len=self.args.pred_len,
            data_path=self.args.data_path,
            features=self.args.features,
            target=self.args.target,
        )
        # Load train, val, test data
        self.train_data = data_loader.get_train()
        self.val_data = data_loader.get_val()
        self.test_data = data_loader.get_test()
        # Build model
        model = build_model(
            input_shape=(self.args.seq_len, data_loader.n_feature),
            pred_len=self.args.pred_len,
            norm_type=self.args.norm_type,
            activation=self.args.activation,
            dropout=self.args.dropout,
            n_block=self.args.n_block,
            ff_dim=self.args.ff_dim,
            target_slice=data_loader.target_slice,
        )
        
        # Set up optimizer
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.args.learning_rate)
        # True compilation
        model.compile(optimizer=optimizer, loss=self.args.loss, metrics=['mae'])
        checkpoint_path = os.path.join(self.args.checkpoints, f'{self.setting}_best')
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
        )
        early_stop_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=self.args.patience
        )
        start_training_time = time.time()
        
        # Fit the model
        history = model.fit(
            self.train_data,
            epochs=self.args.train_epochs,
            validation_data=self.val_data,
            callbacks=[checkpoint_callback, early_stop_callback],
            )
        end_training_time = time.time()
        elasped_training_time = end_training_time - start_training_time
        print(f'Training finished in {elasped_training_time} secconds')

        # evaluate best model on the val set
        # Load weights from the checkpoint
        best_epoch = np.argmin(history.history['val_loss'])
        model.load_weights(checkpoint_path)
        self.model = model # Save as self to move on to .predict()

        #return self.model ## NOTE why are we not returning the best model?
    
    def predict(self):
        # Generate predictions
        self.preds = self.model.predict(self.test_data, batch_size=self.args.batch_size)

        # Extract y_trues from DataLoader
        trues_list = []
        for _, targets in self.test_data:
            trues_list.append(targets.numpy())
        self.trues = np.concatenate(trues_list, axis=0)
                
        if self.args.delete_checkpoint:
            for f in glob.glob(self.args.checkpoint_path + '*'):
                os.remove(f)
        
        # Save results
        folder_path = './results/' + self.setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        preds = self.preds.reshape(-1, self.preds.shape[-2], self.preds.shape[-1])
        trues = self.trues.reshape(-1, self.trues.shape[-2], self.trues.shape[-1])        

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        all_metrics = np.array([mae, mse, rmse, mape, mspe])
        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        np.save(folder_path+'pred.npy', preds)
        np.save(folder_path+'true.npy', trues)      
                
        return self.preds