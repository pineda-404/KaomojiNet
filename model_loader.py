import io
import os
import zipfile
import h5py
import numpy as np

CATEGORIAS = ["uwu", "owo", ">.<", "u.u", "T.T"]

class KaomojiNet:
    def __init__(self, model_path="kaomojinet.keras"):
        self.model_path = model_path
        self.use_keras = False
        self.keras_model = None

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"No se encontro el archivo del modelo: {model_path}")

        try:
            import keras
            self.keras_model = keras.models.load_model(model_path)
            self.use_keras = True
            print(f"[KaomojiNet] Cargado exitosamente con Keras desde {model_path}")
        except Exception as e:
            self._load_weights_h5(model_path)
            print(f"[KaomojiNet] Cargado exitosamente con motor NumPy desde {model_path}")

    def _load_weights_h5(self, model_path):
        with zipfile.ZipFile(model_path, 'r') as z:
            weights_data = z.read('model.weights.h5')
        f = h5py.File(io.BytesIO(weights_data), 'r')
        self.w_c1 = f['layers/conv2d/vars/0'][:]
        self.b_c1 = f['layers/conv2d/vars/1'][:]
        self.w_c2 = f['layers/conv2d_1/vars/0'][:]
        self.b_c2 = f['layers/conv2d_1/vars/1'][:]
        self.w_c3 = f['layers/conv2d_2/vars/0'][:]
        self.b_c3 = f['layers/conv2d_2/vars/1'][:]
        self.w_d1 = f['layers/dense/vars/0'][:]
        self.b_d1 = f['layers/dense/vars/1'][:]
        self.w_d2 = f['layers/dense_1/vars/0'][:]
        self.b_d2 = f['layers/dense_1/vars/1'][:]

    @staticmethod
    def _conv2d_same(x, w, b):
        H, W, C_in = x.shape
        kH, kW, _, C_out = w.shape
        x_pad = np.pad(x, ((kH // 2, kH // 2), (kW // 2, kW // 2), (0, 0)), mode='constant')
        shape = (H, W, kH, kW, C_in)
        strides = (x_pad.strides[0], x_pad.strides[1], x_pad.strides[0], x_pad.strides[1], x_pad.strides[2])
        cols = np.lib.stride_tricks.as_strided(x_pad, shape=shape, strides=strides).reshape(H * W, -1)
        w_mat = w.reshape(-1, C_out)
        return np.maximum(0, (np.dot(cols, w_mat) + b).reshape(H, W, C_out))

    @staticmethod
    def _maxpool2d(x):
        H, W, C = x.shape
        return x.reshape(H // 2, 2, W // 2, 2, C).max(axis=(1, 3))

    def predict(self, img64):
        """
        img64: array numpy de forma (64, 64) o (64, 64, 1), normalizado [0, 1]
        """
        if img64.ndim == 2:
            img64 = img64[..., np.newaxis]

        if self.use_keras:
            inp = img64[np.newaxis, ...]
            preds = self.keras_model.predict(inp, verbose=0)[0]
        else:
            x = self._conv2d_same(img64, self.w_c1, self.b_c1)
            x = self._maxpool2d(x)
            x = self._conv2d_same(x, self.w_c2, self.b_c2)
            x = self._maxpool2d(x)
            x = self._conv2d_same(x, self.w_c3, self.b_c3)
            x = self._maxpool2d(x)
            x = x.reshape(-1)
            x = np.maximum(0, np.dot(x, self.w_d1) + self.b_d1)
            logits = np.dot(x, self.w_d2) + self.b_d2
            exp = np.exp(logits - np.max(logits))
            preds = exp / exp.sum()

        top_idx = int(np.argmax(preds))
        conf = float(preds[top_idx]) * 100.0

        prob_dict = {
            cat: round(float(preds[i]) * 100.0, 2)
            for i, cat in enumerate(CATEGORIAS)
        }

        return {
            "clase": CATEGORIAS[top_idx],
            "confianza": round(conf, 2),
            "probabilidades": prob_dict
        }
