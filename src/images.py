import numpy as np
import PIL

from PIL import Image

def load_data(path, channels=1):
    """ Load the input data into a python array

    path: path to input data
    channels: number of channels in input data

    returns: images [num_images, width, height, channels] as np.float32
    """
    img = Image.open(path)

    if type(img) is PIL.TiffImagePlugin.TiffImageFile:
        n_frames = img.n_frames
        images = np.empty(shape=(n_frames, img.size[1], img.size[0], channels), dtype=np.float32)

        # iterate frames in multipage tiff
        for i in range(n_frames):
            img.seek(i)
            # save images in numpy array as float64
            images[i,:,:,:] = np.array(img)[:,:,np.newaxis]

        return img_uint8_to_float(images)
    else:
        print("Image Format not supported")
        return None

def save_array_as_tif(array, path):
    array = img_float_to_uint8(array)
    images = []
    for layer in np.squeeze(array):
        images.append(Image.fromarray(layer))

    images[0].save(path, compression="tiff_deflate", save_all=True, append_images=images[1:])


def extract_patches(images, patch_size, stride=4, channels=1):
    """Generate patches from an array of images

    images:
        array of images
        shape: [num_images, image_width, image_height, num_channel]
    """

    z, x, y, _ = images.shape
    x_starts = range(0, x - patch_size + stride, stride)
    y_starts = range(0, y - patch_size + stride, stride)
    num_patches = len(x_starts)*len(y_starts)*z

    patches = np.zeros((num_patches, patch_size, patch_size, channels), dtype=np.float32)
    num_patch = 0
    for iz in range(0, z):
        for ix in x_starts:
            for iy in y_starts:
                patches[num_patch] = images[iz, ix:ix + patch_size, iy:iy + patch_size, :]
                num_patch += 1

    return patches

def downsample(patches, downsample_factor, get_all_patches=True):
    """Generate downsampled version of each patch in patches

    patches: patches to be downsampled
    downsample_factor: 
    """
    if not get_all_patches:
        return patches[:, ::downsample_factor, :]

    downsampled_patches = np.zeros_like(patches)
    new_shape = list(downsampled_patches.shape)
    new_shape[0] = downsample_factor * new_shape[0]
    new_shape[1] = int(new_shape[1] / downsample_factor)
    downsampled_patches = np.resize(downsampled_patches, tuple(new_shape))

    for i in range(downsample_factor):
        temp = patches[:, i::downsample_factor, :]
        if temp.shape[1] != new_shape[1]:
            temp = np.delete(temp, new_shape[1], 1)
        downsampled_patches[i * patches.shape[0]: (i + 1) * patches.shape[0]] = temp

    return downsampled_patches

def img_float_to_uint8(img):
    """Transform an array of float images into uint8 images"""
    return (img * 255).round().astype(np.uint8)

def img_uint8_to_float(img):
    """Convert uint8 image to float image
    """
    img = np.array(img, dtype=np.float32)
    img -= np.min(img)
    img *= 1.0 / np.max(img)
    return img

def images_from_patches(patches, image_shape, stride=None):
    """Generate image from patches
    """
    num_patches, patch_size, _, num_channel = patches.shape

    if stride is None:
        stride = patch_size

    x_starts = range(0, image_shape[1] - patch_size + stride, stride)
    y_starts = range(0, image_shape[2] - patch_size + stride, stride)
    patches_per_image = int(num_patches / image_shape[0])

    images = np.zeros(shape=image_shape, dtype=patches.dtype)
    count_hits = np.zeros(shape=image_shape, dtype=np.uint64)

    for n in range(0, image_shape[0]):
        i = 0
        for ix in x_starts:
            for iy in y_starts:
                images[n, ix:ix + patch_size, iy:iy + patch_size] += patches[n*patches_per_image+i]
                count_hits[n, ix:ix + patch_size, iy:iy + patch_size] += 1
                i += 1

    # replace zero counts with 1 to avoid division by 0
    count_hits[count_hits == 0] = 1
    images = images / count_hits

    return images

def psnr(original, degraded):
    """ Compute the peak signal-to-noise ratio
    """
    assert original.shape == degraded.shape, "Shapes of original ({}) and degraded ({}) must be identical to calculate PSNR".format(original.shape, degraded.shape)


    numerator = np.amax(original)
    denominator = ((original - degraded) ** 2).mean()

    return 20 * (np.log10(numerator) - np.log10(np.sqrt(denominator)))
