import tensorflow as tf

import images

class Summary:
    """
        Handle tensorflow summaries.
    """

    def __init__(self, options, session, summary_path):
        self._options = options
        self._session = session
        self._summary_writer = tf.summary.FileWriter(summary_path, session.graph)
        self.summary_ops = []

    def flush(self):
        """

        """
        self._summary_writer.flush()

    def add(self, summary_str, global_step=None):
        """

        """
        self._summary_writer.add_summary(summary_str, global_step)

    def get_summary_op(self, scalars):
        """

        """
        for key, value in scalars.items():
            self.summary_ops.append(tf.summary.scalar(key, value))

        return tf.summary.merge(self.summary_ops)

    def initialize_train_summary(self):
        self._train_predictions = tf.placeholder(tf.float32, name="train_predictions")
        self._train_labels = tf.placeholder(tf.float32, name="train_labels")

        predictions = self._train_predictions
        labels = self._train_labels

        accuracy, recall, precision, f1_score = self.get_prediction_metrics(labels, predictions)

        self._train_summary = [tf.summary.scalar("train_accuracy", accuracy)]
        self._train_summary.append(tf.summary.scalar("train_recall", recall))
        self._train_summary.append(tf.summary.scalar("train_precision", precision))
        self._train_summary.append(tf.summary.scalar("train_f1_score", f1_score))
        self._train_summary = tf.summary.merge(self._train_summary)

    def initialize_eval_summary(self):
        opts = self._options
        num_eval_images = 2
        self._groundtruth_to_display = tf.placeholder(tf.float32, name="groundtruth_display")
        self._image_summary = [
            tf.summary.image('eval_groundtruth', self._groundtruth_to_display, max_outputs=num_eval_images)]

        self._images_to_display = tf.placeholder(tf.float32, name="image_display")
        self._image_summary.append(
            tf.summary.image('eval_images', self._images_to_display, max_outputs=num_eval_images))

        self._labels_to_display = tf.placeholder(tf.float32, name="label_display")
        self._image_summary.append(
            tf.summary.image('eval_labels', self._labels_to_display, max_outputs=num_eval_images))

        self._image_summary = tf.summary.merge(self._image_summary)


    def initialize_snr_summary(self):
        self._snr = tf.placeholder(tf.float64, name='snr')
        self._snr_summary = tf.summary.scalar('SNR', self._snr)

    def add_to_training_summary(self, predictions, labels, global_step):
        train_predictions = predictions
        train_labels = labels

        feed_dict_train = {
            self._train_predictions: train_predictions,
            self._train_labels: train_labels
        }

        train_sum, step = self._session.run([self._train_summary, global_step],
                                            feed_dict=feed_dict_train)
        self._summary_writer.add_summary(train_sum, global_step=step)

    def add_to_snr_summary(self, snr, global_step):
        snr, step = self._session.run(
                [self._snr_summary, global_step],
                feed_dict={self._snr: snr}
        )
        self._summary_writer.add_summary(snr, global_step=step)

    def add_to_eval_summary(self, groundtruth, images_to_predict, predictions, global_step):
        opts = self._options

        img = images.img_float_to_uint8(groundtruth)
        disp_img = images.img_float_to_uint8(images_to_predict)
        labl_img = images.img_float_to_uint8(predictions)

        feed_dict_eval = {
            self._groundtruth_to_display: img,
            self._images_to_display: disp_img,
            self._labels_to_display: labl_img
        }

        image_sum, step = self._session.run([self._image_summary, global_step],
                                            feed_dict=feed_dict_eval)
        self._summary_writer.add_summary(image_sum, global_step=step)

    def get_prediction_metrics(self, labels, predictions):
        accuracy = tf.metrics.accuracy(labels=labels, predictions=predictions)[1]
        recall = tf.metrics.recall(labels=labels, predictions=predictions)[1]
        precision = tf.metrics.precision(labels=labels, predictions=predictions)[1]
        f1_score = 2 / (1 / recall + 1 / precision)

        return accuracy, recall, precision, f1_score
