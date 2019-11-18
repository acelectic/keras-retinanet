"""
Copyright 2017-2018 Fizyr (https://fizyr.com)

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import keras
from ..utils.eval import evaluate


class Evaluate(keras.callbacks.Callback):
    """ Evaluation callback for arbitrary datasets.
    """

    def __init__(
        self,
        generator,
        iou_threshold=0.5,
        score_threshold=0.05,
        max_detections=100,
        save_path=None,
        tensorboard=None,
        weighted_average=False,
        verbose=1
    ):
        """ Evaluate a given dataset using a given model at the end of every epoch during training.

        # Arguments
            generator        : The generator that represents the dataset to evaluate.
            iou_threshold    : The threshold used to consider when a detection is positive or negative.
            score_threshold  : The score confidence threshold to use for detections.
            max_detections   : The maximum number of detections to use per image.
            save_path        : The path to save images with visualized detections to.
            tensorboard      : Instance of keras.callbacks.TensorBoard used to log the mAP value.
            weighted_average : Compute the mAP using the weighted average of precisions among classes.
            verbose          : Set the verbosity level, by default this is set to 1.
        """
        self.generator       = generator
        self.iou_threshold   = iou_threshold
        self.score_threshold = score_threshold
        self.max_detections  = max_detections
        self.save_path       = save_path
        self.tensorboard     = tensorboard
        self.weighted_average = weighted_average
        self.verbose         = verbose

        super(Evaluate, self).__init__()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}

        # run evaluation
        average_precisions, temp_confusion = evaluate(
            self.generator,
            self.model,
            iou_threshold=self.iou_threshold,
            score_threshold=self.score_threshold,
            max_detections=self.max_detections,
            save_path=self.save_path
        )

        # compute per class average precision
        total_instances = []
        precisions = []
        recalls = []
        precisionts = []
        for label, (average_precision, num_annotations ) in average_precisions.items():
            if self.verbose == 1:
                print('class:{}\t{:.0f} instances of class'.format(label, num_annotations),
                    self.generator.label_to_name(label), 'with average precision: {:.4f}'.format(average_precision))
            total_instances.append(num_annotations)
            precisions.append(average_precision)
            
        for label, tmp_conf in temp_confusion.items():
            recall = tmp_conf['recall'] 
            precision_t = tmp_conf['precision']
            if self.verbose == 1:
                print('class:{}\t{:.0f} instances of class'.format(label, num_annotations),
                      self.generator.label_to_name(label), 'with recall: {:.4f}'.format(recall))
                print('class:{}\t{:.0f} instances of class'.format(label, num_annotations),
                      self.generator.label_to_name(label), 'with precision: {:.4f}'.format(precision_t))
            recalls.append(recall)
            precisionts.append(precision_t)

        
        if self.weighted_average:
            self.mean_ap = sum([a * b for a, b in zip(total_instances, precisions)]) / sum(total_instances)
            self.mean_recall = sum([a * b for a, b in zip(total_instances, recalls)]) / sum(total_instances)
            self.mean_precision = sum([a * b for a, b in zip(total_instances, precisionts)]) / sum(total_instances)
        else:
            self.mean_ap = sum(precisions) / sum(x > 0 for x in total_instances)
            self.mean_recall = sum(recalls) / sum(x > 0 for x in total_instances)
            self.mean_precision = sum(precisionts) / sum(x > 0 for x in total_instances)

        if self.tensorboard:
            import tensorflow as tf
            if tf.version.VERSION < '2.0.0' and self.tensorboard.writer:
                summary = tf.compat.v1.Summary()
                summary_value = summary.value.add()
                summary_value.simple_value = self.mean_ap
                summary_value.tag = "mAP"
                self.tensorboard.writer.add_summary(summary, epoch)

               
        self.temp_confusion = temp_confusion

        logs['mAP'] = self.mean_ap
        logs['mean_recall'] = self.mean_recall
        logs['mean_precision'] = self.mean_precision
        
        # logs['true_positives'] =    self.temp_confusion['true_positives']
        # logs['false_positives'] =   self.temp_confusion['false_positives']
        # logs['num_annotations'] =   self.temp_confusion['num_annotations']
        # logs['num_detects'] =       self.temp_confusion['num_detects']
        
        if self.verbose == 1:
            print('mAP: {:.4f}'.format(self.mean_ap))
            print('mean_recall: {:.4f}'.format(self.mean_recall))
            print('mean_precision: {:.4f}'.format(self.mean_precision))

