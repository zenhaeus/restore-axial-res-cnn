import numpy as np
import PIL

from PIL import Image

def load_data(path):
    """
    """
    img = Image.open(path)

    if type(img) is PIL.TiffImagePlugin.TiffImageFile:
        n_frames = img.n_frames
        images = np.empty(shape=(n_frames, img.size[1], img.size[0]))

        # iterate frames in multipage tiff
        for i in range(n_frames):
            img.seek(i)
            images[i,:,:] = np.array(img)

        return images
    else:
        print("Image Format not supported")
        return None


def extract_patches(images, patch_size, stride=4):
    # order dimensions by size
    # assumes z is always smallest dimension

    # z, y, x = sorted(enumerate(images.shape), key=lambda x:x[1])
    x, y, z = images.shape
    x_starts = range(0, x - patch_size, stride)
    y_starts = range(0, y - patch_size, stride)
    num_patches = len(x_starts)*len(y_starts)*z

    patches = np.zeros((num_patches, patch_size, patch_size))
    num_patch = 0
    for iz in range(0, z):
        for ix in x_starts:
            for iy in y_starts:
                patches[num_patch] = images[iz, ix:ix + patch_size, iy:iy + patch_size]
                num_patch += 1

    return patches

def downsample_patches(patches, downsample_factor=3):
    """Generate downsampled version of each patch in patches

    patches: patches to be downsampled
    downsample_factor: 
    """

    # TODO: extend this to generate y axis downsampling as well as 1 and 2 offset downsampling
    samples = patches.copy()
    for i in range(downsample_factor-1):
        samples[:, i::downsample_factor, :] = 0

    return samples
