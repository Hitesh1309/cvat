import os
import shutil
import cv2
import numpy as np
import json
import tensorflow as tf
import keras
import gdown

def to_cvat_mask(box: list, mask):
    xtl, ytl, xbr, ybr = box
    flattened = mask[ytl:ybr + 1, xtl:xbr + 1].flat[:].tolist()
    flattened.extend([xtl, ytl, xbr, ybr])
    return flattened

def copy_file_to_current_directory(full_path):
    # Get the filename from the full path
    filename = os.path.basename(full_path)
    
    # Copy the file to the current directory
    shutil.copy(full_path, filename)

class ModelHandler:
    def __init__(self, labels):

        file_path = "temp/model_new.keras"

        if os.path.isfile(file_path):
            print(f"{file_path} exists and is a file.")
        else:
            print(f"{file_path} either does not exist or is not a file.")

        self.model = keras.models.load_model('temp/model_new.keras')
        self.labels = labels

    def infer(self, image, threshold):
        output = self.model.predict(np.expand_dims(image, axis=0))

        results = []
        mask = (output[0, :, :, 0] > threshold).astype(np.uint8)  # Assuming channel 0 is the mask
        width, height = mask.shape

        for i in range(len(self.labels)):
            mask_by_label = np.zeros((width, height), dtype=np.uint8)
            mask_by_label = ((mask == float(i)) * 255).astype(np.uint8)
            mask_by_label = cv2.resize(mask_by_label,
                dsize=(image.width, image.height),
                interpolation=cv2.INTER_NEAREST)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                contour = np.flip(contour, axis=1)
                if len(contour) < 3:
                    continue

                x_min = max(0, int(np.min(contour[:,:,0])))
                x_max = max(0, int(np.max(contour[:,:,0])))
                y_min = max(0, int(np.min(contour[:,:,1])))
                y_max = max(0, int(np.max(contour[:,:,1])))

                cvat_mask = to_cvat_mask((x_min, y_min, x_max, y_max), mask_by_label)

                # print(cvat_mask)

                results.append({
                    "confidence": None,
                    "label": self.labels.get(i, "unknown"),
                    "points": contour.ravel().tolist(),
                    "mask": cvat_mask,
                    "type": "mask",
                })

                # print(results)

        return results
