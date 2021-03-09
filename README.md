# TimeSeriesConvNet
Convolutional neural network for analysis of time series converted to images. Applied for stock price predictions.  Project for CS496 Advanced Deep Learning class (Northwestern University 2021 Winter). 

This project is an implementation of the approach described by Sim et al. 2019 (https://www.hindawi.com/journals/complexity/2019/4324878/#data-availability). The authors use convolutional neural networks to predict stock market prices.

## Table of contents
- [TimeSeriesConvNet](#timeseriesconvnet)
  * [Problem description](#problem-description)
  * [Approach](#approach)
  * [Dataset](#dataset)
  * [Data processing](#data-processing)
    + [Technical indicators](#technical-indicators)
- [Usage](#usage)
    + [Setup](#setup)
      - [Installation](#installation)
      - [Image generation](#image-generation)
      - [Run the model](#run-the-model)
  * [Example](#example)

## Problem description
Stock market prediction is a very complex task. The price of a stock is influenced by many factors starting from the actual fundamental underlying characteristics of a company to stock market volatility or even news reports. 

Given the complexity of the task, simple models such as support vector machines or feedforward neural networks cannot fully capture the movement of the stock prices well enough and achieved only limited performance. 

## Approach
To overcome the problem of feedforward nets and the support vector machines, the authors proposed using convolutional neural networks. CNNs are based on the convolution operation which enables recognition of similar patterns along the whole image and thus is more suitable for complex pattern identification is stock prices. 

Time series data were converted to image data (64x64x3) by selecting 30 minute windows. Technical indicators related to stock price predictions were calculated from the stock price per minute data and plotted on the the same image.

![alt text](https://raw.githubusercontent.com/karinazad/TimeSeriesConvNet/main/approach.png)

## CNN
CNNs are based on the convolution operation which enables recognition of similar patterns along the whole image and thus is more suitable for complex pattern identification is stock prices. A convolution operation is an elementwise matrix multiplication operation, where one of the matrices is the image and the other is the filter or kernel that turns the image into something else.

![alt text](https://raw.githubusercontent.com/karinazad/TimeSeriesConvNet/main/cnn.png)

In our implementation, the CNN is defined as follow.
```python3
class CNN(tf.Keras.Model):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv2d_1 = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same', activation='relu',
                                     input_shape=(64, 64, 3))
        self.maxpool_1 = layers.MaxPool2D()
        self.conv2d_2 = layers.Conv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')
        self.maxpool_2 = layers.MaxPool2D()
        self.flatten = layers.Flatten()
        self.dense_1 = layers.Dense(units=512, activation='relu')
        self.dense_2 = layers.Dense(units=512, activation='relu')
        self.dense_3 = layers.Dense(units=1, activation='sigmoid')


    def call(self, x):
        x = self.maxpool_1(self.conv2d_1(x))
        x = self.maxpool_2(self.conv2d_2(x))
        x = self.flatten(x)
        x = self.dense_1(x)
        x = self.dense_2(x)
        return self.dense_3(x)
```

## Dataset and Data Processing
### Dataset
Dataset we are using is S&P 500 minute prices. It can be found at https://www.kesci.com/home/dataset/5bbdc2513631bc00109c29a4/files. The dataset consists of rows of minute data for the each individual stocks in S&P 500 as well as the aggregate price under the column S&P 500. This is the one we are interested in. The entire dataset covers over 41,000 minutes

![alt text](https://github.com/karinazad/TimeSeriesConvNet/blob/main/sp500.png)

### Data Processing
To get the time series data into images, we have to perform data processing. To generate the images, we break the whole data into 30 minute long windows and we move in these 30 minutes increments. 33,000 minutes from the dataset are left for the training data (80%), and 8,250 minutes are kept aside for the testing data (20%). After processing, this results in 1,100 input images for training, and the testing data consist of 275 input images. The target is a binary variable that indicates whether the stock price decreased or increased.


### Technical indicators
Technical indicators such as simple moving average, exponential moving average, moving average convergence divergence and others are calculated for each of the windows.  

```python3
N = 20

INDICATOR_FUNCTIONS = {
    "Closing Price": lambda df: df,
    "SMA": lambda df: df.rolling(window=N).mean(),
    "EMA": lambda df: df.ewm(span=N).mean(),
    "MACD": lambda df: df.ewm(span=12, adjust=False).mean() - df.ewm(span=26, adjust=False).mean(),
    "ROC": lambda df: df.pct_change(periods=1),
}
```

### Putting it all together

To go from the provided dataset into final images, we have to 1) split the dataset into windows, 2) calculate technical indicators for each window, and 3) generate images from the curves. We can define the main data processing class called ```TimeSeriesHandler``` in ```utils``` that takes care of this. This class takes in a data path and other parameters, performs splitting of the dataset, and calculates the technical indicators. For details on each function, please click on the code.

```python3
class TimeSeriesHandler:
    def __init__(self,
                 path: str = DATA_PATH,
                 stock_index: str = 'SP500',
                 time_column_name: str = 'DATE',
                 nsamples: Optional[int] = None,
                 minute_window: int = 30,
                 impute_and_scale: bool = True,
                 ):
        """
        Parameters
        ----------
        path: str 
            Path to the csv file with data.
        stock_index: str
            Name of the column of interest.
        time_column_name: str
            Column with time indication, serves as index to the dataset.
        nsamples: int
            Number of samples to be processed.
        minute_window: int
            Time window to divide the dataset in.
        impute_and_scale: bool
            If true, performs imputation for missing data and standard scaling.
        """

        self.stock_index = stock_index

        self.df = pd.read_csv(path)
        self.df.index = pd.to_datetime(self.df[time_column_name], unit='s')
        self.df.drop([time_column_name], axis=1, inplace=True)
        self.df.sort_index(inplace=True)

        self.data = self._split_to_windows(n=nsamples, minute_window=minute_window)
        self.target = np.array((self.data.iloc[0, :] < self.data.iloc[minute_window - 2, :]), dtype=bool)

        if impute_and_scale:
            self.data = self._impute_scale(self.data, scale=False)

        self.data_technical = self._calculate_technical_indicators()
```

To use this class, simply call the function to generate images and indicate the directory where images should be saved.

```python3
handler = TimeSeriesHandler(path=args.data_path,
                                nsamples=args.samples)

handler.generate_images(save_dir=os.path.join(args.save_path, "images"))
```

# Implementation

To run the whole project, we first have to  setup the repo and follow these instructions:

### Installation
To get the code clone the repository. The necessary packages can be installed by running the following commands in the same directory.

    git clone https://github.com/karinazad/TimeSeriesConvNet.git
    cd TimeSeriesConvNet
    pip install -r requirements.txt
    
To add the script path:

    export PYTHONPATH="${PYTHONPATH}<absolute path to the folder>"

### Image generation

After downloading the dataset from  https://www.kesci.com/home/dataset/5bbdc2513631bc00109c29a4/files, upload it to the project folder. 
To generate images from the raw data run the following script from the root folder. You can specify the datapath where your data is stored and the number of samples to generate.

    src/main/generate_images.py --samples 1100 --data-path <path_to_your_dataset>
    
This script will also save targets (price going up or down) for each image in a separate folder (default is data/images and data/targets). 

    
### Run the model

To train and evaluate the model (CNN, ANN, or SVM) on the generated images, run the following script from the root folder:
Note: right now, only CNN is supported.

    src/main/experiments.py --model CNN --show-example True

Running this script will also return performance evaluation. For example:

```
    Model performance on test data:
          hit ratio = 0.68
          specificity = 0.71
          sensitivity = 0.67
```
 
 The script also lets us preview training history for training loss and accuracy.
<img src=https://github.com/karinazad/TimeSeriesConvNet/blob/main/CNN-100epochs.png>
    
## Examples of generated images

Examples of input images with closing price, SMA and EMA:
<img src="https://raw.githubusercontent.com/karinazad/TimeSeriesConvNet/main/CNN2.png" width="650" >

Input images with 5 input variables: closing price, SMA, EMA, ROC, MACD:
<img src="https://raw.githubusercontent.com/karinazad/TimeSeriesConvNet/main/CNN3.png" width="650" >

## Concluding remarks
This leads us to the future work or points that could be further improved. Provided access to GPU, the model should be run for full 2500 epochs. Since the closing price itself achieved the best accuracy, it would be necessary to compare it on its own as well as to include CNN4 model.

In order to improve on the published results, further hyperparameter tuning and more data inputs could further increase model’s accuracy. Moreover, other factors that move in the opposite direction of the stock price, such as interest rate or gold price, could be included in the analysis.


