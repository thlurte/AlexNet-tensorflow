import argparse
# from utils import Loader
# from alexnet import AlexNet

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

