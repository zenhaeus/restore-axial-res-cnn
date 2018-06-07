"""

"""
import tensorflow as tf
import numpy as np

import unet
import images

from summary import Summary
from datetime import datetime

import os

import logging
logging.getLogger('tensorflow').disabled = True

# Set logging level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
tf.logging.set_verbosity(tf.logging.WARN)

# TODO: implement full volume prediction
tf.app.flags.DEFINE_boolean('full_prediction', False, "Whether or not to run a full volume prediction after training")

tf.app.flags.DEFINE_integer('gpu', 0, "GPU to run the model on")
tf.app.flags.DEFINE_integer('batch_size', 20, "Batch size of training instances")
tf.app.flags.DEFINE_integer('patch_size', 120, "Size of the prediction image")
tf.app.flags.DEFINE_integer('stride', 60, "Sliding delta for patches")
tf.app.flags.DEFINE_integer('seed', 2018, "Random seed for reproducibility")
tf.app.flags.DEFINE_integer('root_size', 16, "Number of filters of the first U-Net layer")
tf.app.flags.DEFINE_integer('num_epoch', 40, "Number of pass on the dataset during training")
tf.app.flags.DEFINE_integer('num_layers', 3, "Number of layers of the U-Net")
tf.app.flags.DEFINE_integer('train_score_every', 1000, "Compute training score after the given number of iterations")
tf.app.flags.DEFINE_integer('k_factor', 3, "Determines the factor by which training images are downsampled for training")
tf.app.flags.DEFINE_integer('dilation_size', 3, "Filter size of dilation layer")

tf.app.flags.DEFINE_float('learning_rate', 0.001, "Initial learning rate")
tf.app.flags.DEFINE_float('momentum', 0.9, "Momentum")
tf.app.flags.DEFINE_float('dropout', 0.9, "Probability to keep an input")

tf.app.flags.DEFINE_string('logdir', os.path.abspath("./logdir"), "Directory where to write logfiles")
tf.app.flags.DEFINE_string('save_path', os.path.abspath("./runs"),
                           "Directory where to write checkpoints, overlays and submissions")
tf.app.flags.DEFINE_string('data', os.path.abspath("../data/Membrane_.tif"), "Data to learn on")
tf.app.flags.DEFINE_string('log_suffix', os.path.abspath(""), "suffix to attach to log folder")

FLAGS = tf.app.flags.FLAGS

class Options(object):
    """
        Options used by model
    """
    def __init__(self):
        self.batch_size = FLAGS.batch_size
        self.gpu = FLAGS.gpu
        self.learning_rate = FLAGS.learning_rate
        self.logdir = FLAGS.logdir
        self.momentum = FLAGS.momentum
        self.patch_size = FLAGS.patch_size
        self.save_path = FLAGS.save_path
        self.seed = FLAGS.seed
        self.stride = FLAGS.stride
        self.dropout = FLAGS.dropout
        self.dilation_size = FLAGS.dilation_size
        self.root_size = FLAGS.root_size
        self.num_epoch = FLAGS.num_epoch
        self.num_layers = FLAGS.num_layers
        self.train_score_every = FLAGS.train_score_every
        self.downsample_factor = FLAGS.k_factor
        self.data = FLAGS.data
        self.log_suffix = FLAGS.log_suffix

class ConvolutionalModel:
    def __init__(self, options, session):
        self._options = options
        self._session = session

        self.train_images_shape = None

        np.random.seed(options.seed)
        tf.set_random_seed(options.seed)
        self.input_size = self._options.patch_size

        self.experiment_name = datetime.now().strftime("%Y-%m-%dT%Hh%Mm%Ss")
        experiment_path = os.path.abspath(os.path.join(options.save_path, self.experiment_name))
        self.summary_path = os.path.join(options.logdir, self.experiment_name + options.log_suffix)

        self._summary = Summary(options, session)
        self.build_graph()

    def calculate_loss_abs(self, labels, pred_logits):
        """Calculate absolute difference loss

        """
        loss = tf.losses.absolute_difference(
            labels,
            pred_logits
        )

        return loss

    def calculate_loss_mse(self, labels, pred_logits):
        """Calculate mean squared error loss

        """
        loss = tf.losses.mean_squared_error(
            labels,
            pred_logits
        )

        return loss

    def calculate_loss_snr(self, labels, pred_logits):
        """Calculate loss based on signal to noise
        """
        loss = tf.negative(
            tf.log(tf.divide(
                tf.reduce_sum(tf.square(labels)),
                tf.reduce_sum(tf.square(labels - pred_logits)))),
            name="snr")
        return loss

    def optimize(self, loss):
        """optimize with MomentumOptimizer

        """
        learning_rate = tf.train.exponential_decay(
            self._options.learning_rate,
            self._global_step,
            100,
            0.99,
            staircase=True
        )

        optimizer = tf.train.MomentumOptimizer(learning_rate, self._options.momentum)
        train = optimizer.minimize(loss, global_step=self._global_step)

        return train, learning_rate

    def adam_optimize(self, loss):
        """ optimize with AdamOptimizer
        """
        optimizer = tf.train.AdamOptimizer(self._options.learning_rate, epsilon=10e-3)
        train = optimizer.minimize(loss, global_step=self._global_step)

        return train, optimizer._lr

    def build_graph(self):
        """

        """
        opts = self._options

        global_step = tf.Variable(0, name="global_step")
        self._global_step = global_step

        # data placeholders
        patches_node = tf.placeholder(
            tf.float32,
            shape=(
                self._options.batch_size,
                self.input_size,
                self.input_size,
                1
            ),
            name="patches"
        )

        labels_node = tf.placeholder(
            tf.float32,
            shape=(
                self._options.batch_size,
                self._options.patch_size,
                self._options.patch_size,
                1
            ),
            name="labels"
        )

        dropout_keep = tf.placeholder_with_default(1.0, shape=(), name="dropout_keep")
        self._dropout_keep = dropout_keep

        print("Patches node: {}".format(patches_node))
        predict_logits = unet.forward(
            patches_node,
            root_size=opts.root_size,
            num_layers=opts.num_layers,
            dropout_keep=dropout_keep,
            dilation_size=opts.dilation_size
        )

        #predictions = tf.nn.softmax(predict_logits)
        predictions = predict_logits

        print("Predicted logits: {}".format(predict_logits))
        loss = self.calculate_loss_snr(labels_node, predict_logits)

        self._train, self._learning_rate = self.adam_optimize(loss)

        self._loss = loss
        self._predictions = predictions
        self._patches_node = patches_node
        self._labels_node = labels_node
        self._predict_logits = predict_logits

        self._summary.create_writer(self.summary_path)
        self._summary.initialize_train_summary()
        self._summary.initialize_snr_summary()
        self._summary.initialize_eval_summary()

        summary_scalars = {"loss": loss, "learning_rate": self._learning_rate}
        self.summary_op = self._summary.get_summary_op(summary_scalars)

        # Properly initialize all variables.
        tf.global_variables_initializer().run()
        tf.local_variables_initializer().run()

        self.saver = tf.train.Saver(max_to_keep=100)

    def train(self, patches, labels_patches, eval_images, downsampled_eval_images):
        """Train the model for one epoch

        params:
            patches: [num_patches, patch_height, patch_width]
            imgs: [num_images, img_height, img_width]
        """
        opts = self._options

        # Fix negative values from downsampling
        # TODO: check if this changes network performance
        # downsampled_eval_images[downsampled_eval_images < 0] = 0
        # patches[patches < 0] = 0

        num_train_patches = patches.shape[0]

        indices = np.arange(0, num_train_patches)
        np.random.shuffle(indices)

        for batch_i, offset in enumerate(range(0, num_train_patches - opts.batch_size, opts.batch_size)):
            batch_indices = indices[offset:offset + opts.batch_size]

            feed_dict = {
                self._patches_node: patches[batch_indices, :, :, :],
                self._labels_node: labels_patches[batch_indices, :, :, :],
                self._dropout_keep: opts.dropout,
            }

            summary_str, _, l, predictions, predictions, step = self._session.run([
                self.summary_op,
                self._train,
                self._loss,
                self._predict_logits,
                self._predictions,
                self._global_step
            ],feed_dict=feed_dict)

            print("Batch {} Step {}".format(batch_i, step), end="\r")
            self._summary.add(summary_str, global_step=step)

            snr = images.psnr(labels_patches[batch_indices], predictions)
            self._summary.add_to_snr_summary(snr, self._global_step)

            # Do evaluation once per epoch
            if step > 0 and step % int(patches.shape[0] / opts.batch_size) == 0:
                predictions = self.predict(downsampled_eval_images)

                self._summary.add_to_eval_summary(
                    eval_images,
                    downsampled_eval_images,
                    predictions,
                    self._global_step
                )

        self._summary.flush()

    def predict(self, imgs):
        """Run inference on `imgs` and return predictions

        imgs: [num_images, image_height, image_width, num_channel]
        returns: predictions [num_images, images_height, image_width, num_channel]
        """
        opts = self._options

        num_images = imgs.shape[0]

        print()
        print("Running prediction on {} images with shape {}... ".format(num_images, imgs.shape))

        #patches = images.extract_patches(imgs, self.input_size, opts.stride)
        patches = tf.extract_image_patches(
            imgs,
            [1, self.input_size, self.input_size, 1],
            [1, opts.stride, opts.stride, 1],
            [1, 1, 1, 1],
            'VALID'
        ).eval()
        patches = patches.reshape((-1, self.input_size, self.input_size, 1))

        num_patches = patches.shape[0]

        # patches padding to have full batches
        if num_patches % opts.batch_size != 0:
            num_extra_patches = opts.batch_size - (num_patches % opts.batch_size)
            extra_patches = np.zeros((num_extra_patches, self.input_size, self.input_size, 1))
            patches = np.concatenate([patches, extra_patches], axis=0)

        num_patches = patches.shape[0]

        num_batches = int(num_patches / opts.batch_size)
        eval_predictions = np.ndarray(shape=(num_patches, opts.patch_size, opts.patch_size, 1))
        print("Patches to predict: ", num_patches)
        print("Shape eval predictions: ", eval_predictions.shape)

        for batch in range(num_batches):
            offset = batch * opts.batch_size

            feed_dict = {
                self._patches_node: patches[offset:offset + opts.batch_size, :, :, :],
            }
            eval_predictions[offset:offset + opts.batch_size, :, :, :] = self._session.run(self._predictions, feed_dict)

        # remove padding
        eval_predictions = eval_predictions[0:num_patches]


        # construct predicted images
        predictions = images.images_from_patches(eval_predictions, imgs.shape, stride=opts.stride)

        # Clipping for display in tensorboard
        predictions[predictions < 0] = 0
        predictions[predictions > 1] = 1

        return predictions


    def save(self, epoch=0):
        opts = self._options
        model_data_dir = os.path.abspath(
            os.path.join(opts.save_path, self.experiment_name, 'model-epoch-{:03d}.chkpt'.format(epoch)))
        saved_path = self.saver.save(self._session, model_data_dir)
        # create checkpoint
        print("Model saved in file: {}".format(saved_path))


def main(_):
    """

    """
    opts = Options()

    if opts.gpu == -1:
        config = tf.ConfigProto()
    else:
        config = tf.ConfigProto(allow_soft_placement=True)

    with tf.Graph().as_default(), tf.Session(config=config) as session:
        device = '/device:CPU:0' if opts.gpu == -1 else '/device:GPU:{}'.format(opts.gpu)
        print("Running on device {}".format(device))
        with tf.device(device):
            model = ConvolutionalModel(opts, session)

        if opts.num_epoch > 0:
            # train model
            train_images = images.load_data(os.path.abspath(opts.data))
            model.train_images_shape = train_images.shape

            #labels_patches = images.extract_patches(train_images, model.input_size, opts.stride)
            labels_patches = tf.extract_image_patches(
                train_images,
                [1, model.input_size, model.input_size, 1],
                [1, opts.stride, opts.stride, 1],
                [1, 1, 1, 1],
                'VALID'
            ).eval()
            labels_patches = labels_patches.reshape((-1, model.input_size, model.input_size, 1))
            print("Shape labels_patches: ", labels_patches.shape)

            patches = images.downsample(labels_patches, opts.downsample_factor)
            patches = tf.image.resize_images(
                patches,
                np.array([model.input_size, model.input_size]),
                method=tf.image.ResizeMethod.BICUBIC,
                align_corners=True
            )

            # resize labels_patches to compensate for downsampling
            labels_patches = np.tile(labels_patches, (opts.downsample_factor, 1, 1, 1))

            print("Train on {} labels_patches of size {}x{}".format(
                labels_patches.shape[0],
                labels_patches.shape[1],
                labels_patches.shape[2]
            ))

            print("Train on {} patches of size {}x{}".format(
                patches.shape[0],
                patches.shape[1],
                patches.shape[2]
            ))

            # Eval images
            eval_images = np.swapaxes(train_images, 0, 1)[180:181]
            downsampled_eval_images = images.downsample(eval_images, opts.downsample_factor, get_all_patches=False)
            downsampled_eval_images = tf.image.resize_bicubic(
                downsampled_eval_images,
                np.array([model.train_images_shape[1], model.train_images_shape[2]]),
                align_corners=True
            )

            for i in range(opts.num_epoch):
                print("==== Train epoch: {} ====".format(i))
                # Reset scores
                tf.local_variables_initializer().run()
                # Process one epoch
                model.train(tf.Tensor.eval(patches), labels_patches, eval_images, downsampled_eval_images.eval())
                memop = tf.contrib.memory_stats.MaxBytesInUse()
                print("Memory in use {:.2f} GB".format(memop.eval()/10**9))
                # TODO: Save model to disk
                # model.save(i)

if __name__ == '__main__':
    tf.app.run()
