import argparse
import os
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from src.utils.dataloader import load_images
from src.models.cnn import CNN

IMAGE_PATH = '../data/img'
RESULTS_PATH = '../results'
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

    X = load_images(path=args.image_path,
                    nsamples=50)
    y = np.random.randint(2, size=X.shape[0])

    X_train, X_test, y_train, y_test = train_test_split(X, y, )

    cnn = CNN()
    if args.save_weights:
        cnn.model.save_weights(SAVED_WEIGHTS)

    if args.load_saved_weights:
        cnn.model.load_weights(args.load_saved_weights)

    history = cnn.train(X_train, y_train, epochs=5)
    fig, axes = plt.subplots(1,2, figsize=(8,4))
    axes[0].plot(history.history['loss'])
    axes[0].set_title("Loss")
    axes[1].plot(history.history['accuracy'])
    axes[1].set_title("Accuracy")
    plt.suptitle(f"{args.model}")
    plt.show()

    eval = cnn.model.evaluate(X_test, y_test)

    print(f"Model performance on test data: loss = {eval[0]}, accuracy = {eval[1]}")


    if args.show_example:
        predictions = np.round(cnn.model.predict(X_test).flatten())
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





