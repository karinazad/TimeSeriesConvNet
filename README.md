# TimeSeriesConvNet
Convolutional neural network for analysis of time series converted to images. Applied for stock price predictions. 
Project for CS496 Advanced Deep Learning class (Northwestern University 2021 Winter).

## Dataset

Dataset of S&P 500 minute prices can be found at https://www.kesci.com/home/dataset/5bbdc2513631bc00109c29a4/files. 

## Usage

To setup the repo and run the experiment:

### Setup 

The setup consists of installing all the necessary packages, as well as optional but recommended steps to stratify the 
work flow.

#### Installation

The necessary packages can be installed using:

    pip install -r requirements.txt
    
    
#### Image generation

To generate images from the raw data run the following script from the src folder. You can specify the datapath where your data is stored and the number of samples to generate.

    generate_images.py --data-path ../../data/data_stocks.csv --samples 1100
    
This script will also save targets (price up or down) for each image in a separate folder.   
    
#### : Run the experiment

To train and evaluate the model (CNN, ANN, or SVM) on the generated images, run the following script from the src/experiments folder:

    experiments.py --model CNN --show_example True
    
    
