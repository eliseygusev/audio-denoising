import numpy as np
import torch
from torch.nn.functional import sigmoid

from .config import SAMPLE_SIZE
from .unet import SpectralUNet
from .utils import preprocess, postprocess


class Model():
    def __init__(self, model_path):
        self._init_model(model_path)

    def _init_model(self, model_path):
        self.model = self._create_model()
        self.model.load_state_dict(torch.load(model_path, map_location=self._device()))
        self.model.eval()  # Enable evaluation mode
        self.model.to(self._device())

    def _device(self):
        return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    def _create_model(self):
        model = SpectralUNet()
        return model

    def predict(self, filepath):
        sample = np.load(filepath)
        spectrogram = preprocess(sample, SAMPLE_SIZE).to(self._device())
        output = self.model(spectrogram)
        if self._device().type == 'cuda':
            output = output.cpu()

        mask_preds, class_preds = output

        denoised = postprocess(mask_preds.detach().numpy(), sample.shape)
        class_preds = sigmoid(class_preds).detach().numpy()
        is_clean = np.mean(class_preds) > 0.5

        return {
            'denoised': denoised,
            'is_clean': int(is_clean)
        }
