import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

### Implementation of the energy data synthesizer ammended from Pyraformer
class SynthesisTS:
    '''
    The class is an adaptation of the original energy time series generator developed
    by the authors of Pyraformer. 

    The original script was restructured into a class for better readability and 
    to reflect the order of operations in the synthesis process.
    Further documentation and comments were added.
    '''
    def __init__(self, cycle_periods=[24, 168, 720], series_amount=60, seq_len=720*20):
        '''
        Args:
            cycle_periods (list[int1, int2, int3]): Cycle periods to model dependency. For hourly data = [24, 168, 720] which == [day, week, month].
            series_amount (int): How many synthetic series to generate. (default = 60)
            seq_len (int): The length of the synthetic series to be generated. (default = 20 months)  
        '''
        self.cycle_periods = cycle_periods
        self.series_amount = series_amount
        self.seq_len = seq_len

    # Generates the actual energy time series
    def _generate_sin(self, time_points, sin_coefficients):
        '''
        Generates a mixed sinusoidal sequence given the amount of cycle periods, time points, and the amount of coefficients for individual sine functions.            
        '''
        # Empty array corresponding to time points
        y = np.zeros(len(time_points))
        # Generate individual sine functions to sum up
        for i in range(len(self.cycle_periods)):
            y += sin_coefficients[i] * np.sin(2 * np.pi / self.cycle_periods[i] * time_points)
            
        return y
        
    # Generates covariates (day of week, hour of day, month of year) for hourly data
    # will need amendments if we go with 15-minute frequency data
    def _gen_covariates(self, time_points, index):
        '''
        Generates covariates for each hourly time point t - day of week, hour of the day and month of year.
        '''
        # Set up empty array with same length as time points and 4 columns
        covariates = np.zeros((time_points.shape[0], 4))
        # Generate day of week for each time point t
        covariates[:, 0] = (time_points // 24) % 7
        # Generate our of the day for each time point t
        covariates[:, 1] = time_points % 24
        # Generate month for each time point t
        covariates[:, 2] = (time_points // (24 * 30)) % 12
        # Normalize day of week on MinMax (0, 1)
        covariates[:, 0] = covariates[:, 0] / 6
        # Normalize hour of the day on MinMax (0, 1)
        covariates[:, 1] = covariates[:, 1] / 23
        # Normalize month of year on MinMax (0, 1)
        covariates[:, 2] = covariates[:, 2] / 11
        # Add new column that tracks which time series the covariates are for
        covariates[:, -1] = np.zeros(time_points.shape[0]) + index
        # Return
        return covariates

    # Defines a polynomially decaying covariance of B_0
    def _polynomial_decay_cov(self):
        '''
        Defines polynomially decaying covariance function for B_0, which will be then drawn from Gaussian distribution.
        '''
        # Mean at each time point t will be 0
        mean = np.zeros(self.seq_len)
        # Obtaining distance matrix
        x_axis = np.arange(self.seq_len)
        distance = x_axis[:, None] - x_axis[None, :]
        distance = np.abs(distance)
        # Apply polynomial decay 
        cov = 1 / (distance + 1)
        return mean, cov
        
    def _multivariate_normal(self, mean, cov):
        '''
        Generates (series_amount) of Gaussian distributions for drawing the noise terms.
        Takes mean and cov generated by _polynomial_decay_cov as argument. 
        '''             
        noise = np.random.multivariate_normal(mean, cov, (self.series_amount,), 'raise')
        return noise
        
    def synthesze_single_series(self):
        '''
        Generates a single time series with a date-time index.
        '''
        # Generate initial time stamp
        init_date_stamp = pd.Timestamp('2022-01-01 00:00:00')
        # Generate fake hour within month of january
        # "Start of each time series t_0 is uniformly sampled from [0, 720]" - comment from paper
        start = int(np.random.uniform(0, self.cycle_periods[-1]))
        # Generate time points for the _generate_sin ---- May be reduntand after the datestamp generation was added
        time_points = start + np.arange(self.seq_len)
        # Obtain 'real' start date of the series
        real_start_date = init_date_stamp + pd.to_timedelta(start, 'H')
        # Generate datetime index for entire sequence
        datetime_index = pd.date_range(start=real_start_date, periods=self.seq_len, freq='H')
        # "Coefficients of the three sine functions B_1, B_2, B_3 for each time series sampled uniformly from [5, 10]"
        sin_coefficients = np.random.uniform(5, 10, 3)
        # Generate time series
        y = self._generate_sin(time_points, sin_coefficients)
        # Define mean and covariance of the noise term B_0 - a Gaussian process with a polynomially decaying covariance function.
        mean, cov = self._polynomial_decay_cov()
        # Draw B_0 -s for each time point t for each time series from Gaussian distribution
        noise = self._multivariate_normal(mean, cov) 
        series = y + noise
        df = pd.DataFrame({'date': datetime_index, 'TARGET': series.squeeze()})

        return df  

    def synthesize_data(self, generate_covariates=True):
        '''
        Main method that synthesizes the time series and their covariates (optional).

        Returns: 
            Time series datasets in the amount of series_amount
        '''
        # Empty containers
        data = []
        if generate_covariates:
            covariates = []
        # Iterator
        i = 0
        # Loop
        # "We generate 60 time series"
        while i < self.series_amount:
            # "Start of each time series t_0 is uniformly sampled from [0, 720]" - comment from paper
            start = int(np.random.uniform(0, self.cycle_periods[-1]))
            # Generate time points 
            time_points = start + np.arange(self.seq_len)
            # "Coefficients of the three sine functions B_1, B_2, B_3 for each time series sampled uniformly from [5, 10]"
            sin_coefficients = np.random.uniform(5, 10, 3)
            # Generate time series
            y = self._generate_sin(time_points, sin_coefficients)
            # Append
            data.append(y)
            # If we want covariates
            if generate_covariates:
                # Generate and append covariates
                covariates.append(self._gen_covariates(time_points, i))
            # Update iterator
            i += 1
        # Turn results into array
        data = np.array(data)
        # Define mean and covariance of the noise term B_0 - a Gaussian process with a polynomially decaying covariance function.
        mean, cov = self._polynomial_decay_cov()
        # Draw B_0 -s for each time point t for each time series from Gaussian distribution
        noise = self._multivariate_normal(mean, cov)
        # Add B_0-s for each time point t
        data = data + noise
        # If we want covariates
        if generate_covariates:
            # Set up as np array
            covariates = np.array(covariates)
            # Concatenate with time series
            data = np.concatenate([data[:, :, None], covariates], axis=2)
            np.save('synthetic_series.npy', data)
        return data
    

