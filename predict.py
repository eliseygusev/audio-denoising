import argparse

from src.config import FITTED_MODEL_PATH
from src.model import Model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mel-path')
    args = parser.parse_args()
    print('Loading model')
    model = Model(FITTED_MODEL_PATH)
    print('Successfully loaded model')
    preds = model.predict(args.mel_path)
    print(preds)


if __name__ == '__main__':
    main()
