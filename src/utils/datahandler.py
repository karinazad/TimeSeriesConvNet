from collections import defaultdict
from typing import List, Optional, Dict, Tuple
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


from src.utils.technical_indicators import TECHNICAL_INDICATORS, INDICATOR_FUNCTIONS

DATA_PATH = '../data/data_stocks.csv'
SAVE_PATH = '../data/img/'


class TimeSeriesHandler:
    def __init__(self,
                 path: str = DATA_PATH,
                 stock_index: str = 'SP500',
                 time_column_name: str = 'DATE',
                 nsamples=10,
                 minute_window=30,
                 impute_and_scale=True):

        self.stock_index = stock_index
        self.data_technical = None

        self.df = pd.read_csv(path)
        self.df.index = pd.to_datetime(self.df[time_column_name], unit='s')
        self.df.drop([time_column_name], axis=1, inplace=True)
        self.df.sort_index(inplace=True)

        self.data = self._split_to_windows(n=nsamples, minute_window=minute_window)
        self.target = np.array((self.data.iloc[0, :] < self.data.iloc[minute_window-2, :]), dtype =bool)

        if impute_and_scale:
            self.data = self._impute_scale(self.data, scale=False)


    def _split_to_windows(self,
                          n: Optional = None,
                          minute_window=30):
        start = min(self.df.index)
        end = max(self.df.index)
        duration = end - start

        delta = pd.Timedelta(f"{minute_window} minutes")

        if n is None:
            n = int(duration / delta)

        assert n <= int(duration / delta)

        windows = [start + delta * i for i in range(n)]

        windows_dict = dict()
        skipped = []

        for i in range(len(windows) - 1):
            cond1 = (self.df.index > windows[i])
            cond2 = self.df.index < windows[i + 1]
            cond = cond1 & cond2

            vals = self.df[cond][self.stock_index].values

            # Check if there are data in this window.
            if len(vals) != 0:
                windows_dict[windows[i]] = self.df[cond][self.stock_index].values

            # If no data are logged in this time period (off-hours).
            else:
                skipped += [(windows[i], windows[i+1])]

        print(f"Processed {n} samples total. Passed: {n-len(skipped)}. Skipped: {len(skipped)}.")

        return pd.DataFrame(windows_dict)

    def _impute_scale(self, df, scale=False):
        if scale:
            pipe = Pipeline([('scaler', MinMaxScaler()), ('impute', SimpleImputer())])
        else:
            pipe = Pipeline([('impute', SimpleImputer())])

        cols = df.columns
        df[cols] = pipe.fit_transform(df[cols])

        return df

    def calculate_technical_indicators(self,
                                       indicator_functions: Optional = None):
        if self.data_technical is None:
            if indicator_functions is None:
                indicator_functions = INDICATOR_FUNCTIONS

            results = defaultdict(lambda: defaultdict(List))

            for i, col in enumerate(self.data.columns):
                res = dict()

                for indicator, function in indicator_functions.items():
                    res[indicator] = function(self.data[col])

                results[i] = res

            # Save the result
            self.data_technical = results

        return self.data_technical

    def generate_images(self,
                        save_dir: Optional[str] = SAVE_PATH):
        pipe = Pipeline([('scaler', MinMaxScaler()), ('impute', SimpleImputer())])

        if self.data_technical is None:
            self.calculate_technical_indicators()

        count = 0
        for key, sample in self.data_technical.items():
            sample = pd.DataFrame(sample)

            sample[sample.columns] = pipe.fit_transform(sample[sample.columns])

            fig = plt.figure(figsize=(16, 16))
            sample.plot()
            plt.legend().set_visible(False)
            plt.axis('off')

            if save_dir:
                save_fname = os.path.join(save_dir, f"sp500_{count}")
                plt.savefig(save_fname, bbox_inches='tight', pad_inches=0)
                count += 1
            else:
                plt.show()

            plt.close()

    def plot_preview(self):
        pass
