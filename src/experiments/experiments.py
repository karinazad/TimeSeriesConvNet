import argparse
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from src.models.ann import ANN
from src.models.cnn import CNN
from src.models.svm import SVM
from src.utils.dataloader import load_images

IMAGE_PATH = '../data/img'
TARGET_PATH = '../data/targets/targets.npy'
RESULTS_PATH = '../../results'
SAVED_WEIGHTS = '../weights'

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default='CNN',
        choices=["CNN", "ANN", "SVM"],
    )
    parser.add_argument(
        "--image-path",
        type=str,
        nargs="+",
        default=IMAGE_PATH,
    )
    parser.add_argument(
        "--target-path",
        type=str,
        nargs="+",
        default=TARGET_PATH,
    )
    parser.add_argument(
        "--results-path",
        type=str,
        nargs="+",
        default=RESULTS_PATH,
    )
    parser.add_argument(
        "--load-saved-weights",
        type=Optional[str],
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--show-example",
        type=bool,
        nargs="+",
        default=True,
    )
    parser.add_argument(
        "--save-weights",
        type=bool,
        nargs="+",
        default=False,
    )
    args = parser.parse_args()

    # Load the generated data and get train and test values
    print("Loading images...")
    X = load_images(path=args.image_path, nsamples=None)
    y = np.load(args.target_path)
    print(f"...succesfully loaded {X.shape[0]} images.\n")

    X_train, X_test, y_train, y_test = train_test_split(X, y, )

    # Create the model
    if args.model == "CNN":
        model = CNN()
    elif args.model == "ANN":
        model = ANN()
    elif args.model == "SVM":
        model = SVM()
    else:
        raise Warning(f"Sorry, your selection {args.model} is not available.")

    print(f"Starting to train the model...")
    model.train(X_train,
                y_train,
                epochs=2,
                steps_per_epoch=20)

    print(f"...finished training.\n")
    model.plot_history()

    # Evaluate the model
    y_pred = np.round(model.model.predict(X_test).flatten())

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    cm = confusion_matrix(y_test, y_pred)
    hit_ratio = np.round((tn + tp) / (tn + fp + fn + tp), 2)
    specificity = np.round(tp / (fn + tp), 2)
    sensitivity = np.round(tn / (tn + fp), 2)

    print(f"Model performance on test data:"
          f"\n\thit ratio = {hit_ratio}"
          f"\n\tspecificity = {specificity}"
          f"\n\tsensitivity = {sensitivity}")

    # Plot an example of the input sample
    if args.show_example:
        predictions = np.round(model.model.predict(X_test).flatten())
        correct = np.where(predictions == y_test.flatten())[0]

        if len(correct) == 0:
            raise Warning("Ups, the model did not predict any sample right.")

        trends = ["down", "up"]
        example = X_test[correct[0]]
        plt.imshow(example)
        plt.text(65, 6, s=f'Prediction: {trends[int(predictions[correct[0]])]}')
        plt.text(65, 1, s=f'Trend: {trends[int(y_test[correct[0]])]}')
        plt.title(f"Example of a prediction with {args.model} model")
        plt.show()
        plt.close()
