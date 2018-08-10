# Restoring axial resolution using a 2D/3D deep convolution neuronal network

We built a convolutional network which attempts to restore the axial resolution in anisotropic 3D microscopy data.


# Setup
The software is implemented in python using the following libraries:

- Tensorflow
- Numpy
- Pillow

If you want to run the application you will have to make sure that these are installed. To visualize the results you might also want to install `Tensorboard` software.

# Run
Use the following command within the `src` directory to run the application

```bash
    python tf_restore_axial_res.py
```

# Flags
When running `./tf_restore_axial_res.py` there are a number of flags to influence the behaviour of the training procedure:

| Flag              |      Description |
|-------------------|------------------|
| full\_prediction  |      Wheter or not to run a prediction on the full image volume after training |
| gpu               |      GPU ID on which to run the training procedure |
| batch\_size       |      Batch size of training instances |
| patch\_size       |      Size of the prediction image |
| stride            |      Sliding delta for patches |
| seed              |      Random seed for reproducibility |
| root\_size        |      Number of filters of the first U-Net layer |
| num\_epoch        |      Number of pass on the dataset during training |
| num\_layers       |      Number of layers of the U-Net |
| k\_factor         |      Determines the factor by which training images are downsampled for training |
| dilation\_size    |      Filter size of dilated convolution layer |
| conv\_size        |      Filter size of convolution layer |
| larning\_rate     |      Initial learning rate for Adam Optimizer |
| dropout           |      Probability to keep an input |
|                   |                                   | 
| logdir            |      Directory where to write logfiles |
| save\_path        |      Directory where to write checkpoints |
| data              |      Path to data to learn on |
| log\_suffix       |      suffix to attach to log folder |

Default values for each parameter can be seen in the source code of `tf_restore_axial_res.py`

To pass a flag use the following syntax:

```bash
    python tf_restore_axial_res.py --flag=value
```

The training data needs to be supplied in form of a greyscale multilayer tiff file.
