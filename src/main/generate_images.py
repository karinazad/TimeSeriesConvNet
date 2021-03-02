import argparse
import os
from typing import Optional

import numpy as np

from src.utils.datahandler import TimeSeriesHandler

DATA_PATH = 'data/data_stocks.csv'
SAVE_PATH = 'data'

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
        type=Optional[bool],
        default=False,
    )

    args = parser.parse_args()

    save_dir = args.save_path

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    handler = TimeSeriesHandler(path=args.data_path,
                                nsamples=args.samples)

    targets = handler.generate_images(save_dir=os.path.join(args.save_path, "images"))
    np.save(os.path.join(args.save_path, "targets/targets.npy"), targets)
