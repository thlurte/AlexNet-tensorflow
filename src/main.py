import argparse

from src.utils import Loader
from utils import Loader
from utils import Visualize
from alexnet import AlexNet

import tensoflow as tf
import tensorflow.keras as tfk
import matplotlib.pyplot as plt

parser=argparse.ArgumentParser(description="AlexNet")

parser.add_argument(
    "--data_dir"
    ,type=str
    ,required=True
    ,help="Path to the dataset directory"
    )

parser.add_argument(
    "--epochs"
    ,type=int
    ,default=10
    ,help="Number of training epochs"
    )

args=parser.parse_args()

data_dir=args.data_dir
epochs=args.epochs


# Image size
IMG_SIZE=(256,256,3)
# Batch Size
BATCH_SIZE=128

# Load the dataset
loader=Loader(data_dir=data_dir,img_size=IMG_SIZE,batch_size=BATCH_SIZE)
trainds,testds=loader.load()


# Initialize AlexNet class
alexnet=AlexNet(input_size=IMG_SIZE,batch_size=BATCH_SIZE)

# Load model
input,x=alexnet.model()

# Build the model
model=alexnet.build(model=input,x=x)

# Fit the model
history=alexnet.fit(model=model,trainds=trainds,testds=testds,epochs=epochs)

# Plot the results
plot=Visualize(history=history)




