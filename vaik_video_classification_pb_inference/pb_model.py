from typing import List, Dict, Tuple
import tensorflow as tf
from PIL import Image
import numpy as np
from numpy import ndarray


class PbModel:
    def __init__(self, input_saved_model_dir_path: str = None, classes: Tuple = None):
        self.model = tf.saved_model.load(input_saved_model_dir_path)
        self.model_input_shape = self.model.signatures["serving_default"].inputs[0].shape
        self.model_input_dtype = self.model.signatures["serving_default"].inputs[0].dtype
        self.model_output_shape = self.model.signatures["serving_default"].outputs[0].shape
        self.model_output_dtype = self.model.signatures["serving_default"].outputs[0].dtype
        self.classes = classes

    def inference(self, input_image_list: List[np.ndarray], batch_size: int = 8) -> Tuple[List[Dict], ndarray]:
        resized_image_array = self.__preprocess_image_list(input_image_list, self.model_input_shape[1:])
        raw_pred = self.__inference(resized_image_array, batch_size)
        output = self.__output_parse(raw_pred, len(input_image_list)-1)
        return output, raw_pred

    def __inference(self, resize_input_tensor: np.ndarray, batch_size: int) -> np.ndarray:
        if len(resize_input_tensor.shape) != 5:
            raise ValueError('dimension mismatch')
        if not np.issubdtype(resize_input_tensor.dtype, np.uint8):
            raise ValueError(f'dtype mismatch expected: {np.uint8}, actual: {resize_input_tensor.dtype}')

        output_tensor = tf.zeros((resize_input_tensor.shape[0], self.model_output_shape[-1]),
                                 self.model_output_dtype).numpy()
        for index in range(0, resize_input_tensor.shape[0], batch_size):
            batch = resize_input_tensor[index:index + batch_size, :, :, :]
            batch_pad = tf.zeros(((batch_size, ) + self.model_input_shape[1:]), self.model_input_dtype).numpy()
            batch_pad[:batch.shape[0], :, :, :] = batch
            raw_pred = self.model(batch_pad)
            output_tensor[index:index + batch.shape[0], :] = raw_pred[:batch.shape[0]]
        return output_tensor

    def __preprocess_image_list(self, input_image_list: List[np.ndarray],
                                resize_input_shape: Tuple[int, int, int, int]) -> np.ndarray:
        resized_image_list = []
        for input_image in input_image_list:
            resized_image = self.__preprocess_image(input_image, resize_input_shape[1:3])
            resized_image_list.append(resized_image)
        for _ in range(resize_input_shape[0] - (len(input_image_list) % resize_input_shape[0])):
            resized_image_list.append(np.zeros(resize_input_shape[1:], dtype=np.uint8))
        resized_image_array = np.split(np.stack(resized_image_list), len(resized_image_list)//resize_input_shape[0])
        return np.stack(resized_image_array)

    def __preprocess_image(self, input_image: np.ndarray, resize_input_shape: Tuple[int, int]) -> np.ndarray:
        if len(input_image.shape) != 3:
            raise ValueError('dimension mismatch')
        if not np.issubdtype(input_image.dtype, np.uint8):
            raise ValueError(f'dtype mismatch expected: {np.uint8}, actual: {input_image.dtype}')

        output_image = np.zeros((*resize_input_shape, input_image.shape[2]),
                                dtype=input_image.dtype)
        pil_image = Image.fromarray(input_image)
        x_ratio, y_ratio = resize_input_shape[1] / pil_image.width, resize_input_shape[0] / pil_image.height
        if x_ratio < y_ratio:
            resize_size = (resize_input_shape[1], round(pil_image.height * x_ratio))
        else:
            resize_size = (round(pil_image.width * y_ratio), resize_input_shape[0])
        resize_pil_image = pil_image.resize(resize_size)
        resize_image = np.array(resize_pil_image)
        output_image[:resize_image.shape[0], :resize_image.shape[1], :] = resize_image
        return output_image

    def __output_parse(self, pred: np.ndarray, max_frame_index: int) -> List[Dict]:
        output_dict_list = []
        pred_index = np.argsort(-pred, axis=-1)
        for index in range(pred.shape[0]):
            output_dict = {'score': pred[index][pred_index[index]].tolist(),
                           'label': [self.classes[class_index] for class_index in pred_index[index]],
                           'start_frame': index*self.model_input_shape[1],
                            'end_frame': min(index*self.model_input_shape[1]+self.model_input_shape[1], max_frame_index)}
            output_dict_list.append(output_dict)
        return output_dict_list