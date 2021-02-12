import argparse
import os
from typing import Optional

from src.utils.datahandler import TimeSeriesHandler

DATA_PATH = '../data/data_stocks.csv'
SAVE_PATH = '../data/img'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-path",
        type=str,
        nargs="+",
        default=DATA_PATH,
    )
    parser.add_argument(
        "--save-path",
        type=str,
        default=SAVE_PATH,
    )

    parser.add_argument(
        "--samples",
        type=Optional[int],
        default=None,
    )
    parser.add_argument(
        "--plot-preview",
        type=Optional[int],
        default=None,
    )
    args = parser.parse_args()

    save_dir = args.save_path

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    handler = TimeSeriesHandler(path=args.data_path,
                                nsamples=args.samples)

    handler.generate_images(save_dir=args.save_path)

    if args.plot_preview:
        handler.plot_preview()