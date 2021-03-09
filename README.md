# TimeSeriesConvNet
Convolutional neural network for analysis of time series converted to images. Applied for stock price predictions. 
Project for CS496 Advanced Deep Learning class (Northwestern University 2021 Winter). 

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
To overcome the problem of feedforward nets and the support vector machines, the authors proposed using convolutional neural networks. 

CNNs are based on the convolution operation which enables recognition of similar patterns along the whole image and thus is more suitable for complex pattern identification is stock prices. A convolution operation is an elementwise matrix multiplication operation, where one of the matrices is the image and the other is the filter or kernel that turns the image into something else.


## Dataset
Dataset of S&P 500 minute prices can be found at https://www.kesci.com/home/dataset/5bbdc2513631bc00109c29a4/files. 

## Data processing
To get the time series data into images, we have to perform data processing. The dataset used in the paper which is available at this website consists of rows of minute data for the each individual stocks in S&P 500 as well as the aggregate price under the column S&P 500. This is the one we are interested in. 

The dataset consists of minute data of the S&P 500 index from 2017. The entire dataset covers over 41,000 minutes. To generate the images, we break the whole data into 30 minute long windows and we move in these 30 minutes increments.

33,000 minutes from the dataset are left for the training data (80%), and 8,250 minutes are kept aside for the testing data (20%). After processing, this results in 1,100 input images for training, and the testing data consist of 275 input images. The target is a binary variable that indicates whether the stock price decreased or increased.

### Technical indicators
To get the time series data into images, we have to perform data processing. The dataset used in the paper which is available at this website consists of rows of minute data for the each individual stocks in S&P 500 as well as the aggregate price under the column S&P 500. This is the one we are interested in. 


# Usage

To setup the repo and follow these instructions:

### Setup 

The setup consists of installing all the necessary packages, as well as optional but recommended steps to stratify the 
work flow.

#### Installation
To get the code, run:

    git clone https://github.com/karinazad/TimeSeriesConvNet.git

The necessary packages can be installed by running the following command in the same directory:

    pip install -r requirements.txt
    
To add the script path:

    export PYTHONPATH="${PYTHONPATH}<absolute path to the folder>"

#### Image generation

After downloading the dataset from  https://www.kesci.com/home/dataset/5bbdc2513631bc00109c29a4/files, upload it to the project folder. 
To generate images from the raw data run the following script from the root folder. You can specify the datapath where your data is stored and the number of samples to generate.

    src/main/generate_images.py --samples 1100 --data-path <path_to_your_dataset>
    
This script will also save targets (price going up or down) for each image in a separate folder (default is data/images and data/targets). 

    
#### Run the model

To train and evaluate the model (CNN, ANN, or SVM) on the generated images, run the following script from the root folder:
Note: right now, only CNN is supported.

    src/main/experiments.py --model CNN --show-example True
   
    
## Example of generated images

Examples of input images with closing price, SMA and EMA:
<img src="https://raw.githubusercontent.com/karinazad/TimeSeriesConvNet/main/CNN2.png" width="700" >

Input images with 5 input variables: closing price, SMA, EMA, ROC, MACD:
<img src="https://raw.githubusercontent.com/karinazad/TimeSeriesConvNet/main/CNN3.png" width="700" >

## Concluding remarks
This leads us to the future work or points that could be further improved. Provided access to GPU, the model should be run for full 2500 epochs. Since the closing price itself achieved the best accuracy, it would be necessary to compare it on its own as well as to include CNN4 model.

In order to improve on the published results, further hyperparameter tuning and more data inputs could further increase model’s accuracy. Moreover, other factors that move in the opposite direction of the stock price, such as interest rate or gold price, could be included in the analysis.


