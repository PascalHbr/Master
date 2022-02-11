import tensorflow as tf
import tensorflow_hub as hub
import numpy as np


class I3D:
    def __init__(self, labels):
        self.i3d = hub.load("https://tfhub.dev/deepmind/i3d-kinetics-400/1").signatures['default']
        self.labels = labels

    def predict(self, video, targets):
        # Add a batch axis to the sample video.
        model_input = tf.constant(video, dtype=tf.float32)

        logits = self.i3d(model_input)['default'][0]
        probabilities = tf.nn.softmax(logits)
        sorted_probabilities = list(np.argsort(probabilities)[::-1])

        # See if correct
        top_1_bool = 0
        top_3_bool = 0
        top_5_bool = 0

        if any([target == sorted_probabilities[0] for target in targets]):
            top_1_bool = 1
        if any([target in sorted_probabilities[:3] for target in targets]):
            top_3_bool = 1
        if any([target in sorted_probabilities[:5] for target in targets]):
            top_5_bool = 1

        return top_1_bool, top_3_bool, top_5_bool