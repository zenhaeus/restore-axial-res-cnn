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
    """ Save numpy array as

    array: [num_images, image_height, image_width, num_channel]
    path: where to save the result as a tif image

    """
    array = img_float_to_uint8(array)
    images = []
    for layer in np.squeeze(array):
        images.append(Image.fromarray(layer))

    images[0].save(path, compression="tiff_deflate", save_all=True, append_images=images[1:])


def extract_patches(images, patch_size, stride=4):
    """Generate patches from an array of images

    images:
        array of images
        shape: [num_images, image_width, image_height, num_channel]

    patch_size:
        extracted patches have dimensions [patch_size, patch_size]

    stride:
        distance between overlapping extracted patches

    """

    z, x, y, channels = images.shape
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
    To downsample the images this function will keep every downsample_factor-th line
    and drop all other lines.

    patches: patches to be downsampled
    downsample_factor: int value indicating by which factor the images should be subsapmled
    get_all_patches: boolean value indicating whether or not to return all possible downsampled
        versions of every image.

    returns:
        if get_all_patches:
            downsampled images [num_images, int(image_height / downsample_factor), image_width, num_channel]
        else:
            downsampled images [downsample_factor * num_images, int(image_height / downsample_factor), image_width, num_channel]
    """

    # if we don't want all patches we just return the first set of subsampled
    # lines
    if not get_all_patches:
        return patches[:, ::downsample_factor, :]

    downsampled_patches = np.zeros_like(patches)
    new_shape = list(downsampled_patches.shape)
    new_shape[0] = downsample_factor * new_shape[0]
    new_shape[1] = int(new_shape[1] / downsample_factor)

    # placeholder array for downsampled patches / images
    downsampled_patches = np.resize(downsampled_patches, tuple(new_shape))

    for i in range(downsample_factor):
        temp = patches[:, i::downsample_factor, :]
        if temp.shape[1] != new_shape[1]:
            temp = np.delete(temp, new_shape[1], 1)
        downsampled_patches[i * patches.shape[0]: (i + 1) * patches.shape[0]] = temp

    return downsampled_patches

def img_float_to_uint8(img):
    """Transform an array of float images into uint8 images

    img: image to convert to uint8
    """
    return (img * 255).round().astype(np.uint8)

def img_uint8_to_float(img):
    """Convert uint8 image to float image and normalize
    Don't use this on individual images of a 3D image volume

    imgs: [num_images, image_height, image_width, num_channel]
    returns: img [num_images, image_height, image_width, num_channel] converted and normalized images
    """
    img = np.array(img, dtype=np.float32)
    img -= np.min(img)
    img *= 1.0 / np.max(img)
    return img

def images_from_patches(patches, image_shape, stride=None):
    """Reassemble image from patches

    patches: patches to transform back into images
    image_shape: shape of the final image [num_images, image_height, image_width, num_channel]
    stride: stride that was used to extract patches

    returns: [num_images, images_height, images_width, num_channel] reassembled images
    """
    num_patches, patch_size, _, _ = patches.shape

    if stride is None:
        stride = patch_size

    x_starts = range(0, image_shape[1] - patch_size + stride, stride)
    y_starts = range(0, image_shape[2] - patch_size + stride, stride)
    patches_per_image = int(num_patches / image_shape[0])

    images = np.zeros(shape=image_shape, dtype=patches.dtype)
    #keep track of number of patches that overlap any region in image
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
    # normalize image
    images = images / count_hits

    return images

def psnr(original, degraded):
    """ Compute the peak signal-to-noise ratio

    The shapes of the 'original' and the 'degraded' array need to be identical.

    original: original image
    degraded: degraded image

    returns: (float) peak signal to noise ratio
    """
    assert original.shape == degraded.shape, "Shapes of original ({}) and degraded ({}) must be identical to calculate PSNR".format(original.shape, degraded.shape)


    numerator = np.amax(original)
    denominator = ((original - degraded) ** 2).mean()

    return 20 * (np.log10(numerator) - np.log10(np.sqrt(denominator)))
