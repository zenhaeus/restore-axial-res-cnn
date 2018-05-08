import numpy as np
import PIL

from PIL import Image

def load_data(path, channels=1):
    """
    """
    img = Image.open(path)

    if type(img) is PIL.TiffImagePlugin.TiffImageFile:
        n_frames = img.n_frames
        images = np.empty(shape=(n_frames, img.size[1], img.size[0], channels))

        # iterate frames in multipage tiff
        for i in range(n_frames):
            img.seek(i)
            # save images in numpy array as float64
            images[i,:,:,:] = img_uint8_to_float(img)[:,:,np.newaxis]

        return images
    else:
        print("Image Format not supported")
        return None


def extract_patches(images, patch_size, stride=4, channels=1):
    """Generate patches from an array of images

    images:
        array of images
        shape: [num_images, image_width, image_height, num_channel]
    """

    z, x, y, _ = images.shape
    x_starts = range(0, x - patch_size, stride)
    y_starts = range(0, y - patch_size, stride)
    num_patches = len(x_starts)*len(y_starts)*z

    patches = np.zeros((num_patches, patch_size, patch_size, channels))
    num_patch = 0
    for iz in range(0, z):
        for ix in x_starts:
            for iy in y_starts:
                patches[num_patch] = images[iz, ix:ix + patch_size, iy:iy + patch_size, :]
                num_patch += 1

    return patches

def downsample_patches(patches, downsample_factor=10, keep_lines_between=True):
    """Generate downsampled version of each patch in patches

    patches: patches to be downsampled
    downsample_factor: 
    """

    # TODO: extend this to generate y axis downsampling as well as 1 and 2 offset downsampling
    #samples = patches.copy()
    #if keep_lines_between:
    #    for i in range(downsample_factor-1):
    #        samples[:, i::downsample_factor, :] = 0
    #else:
    #    return samples[:, 2::downsample_factor, :]

    #return samples
    return patches[:, ::downsample_factor, :]

def img_float_to_uint8(img):
    """Transform an array of float images into uint8 images"""
    return (img * 255).round().astype(np.uint8)

def img_uint8_to_float(img):
    img = np.array(img, dtype=np.float32)
    img -= np.min(img)
    img *= 1.0 / np.max(img)
    return img

def images_from_patches(patches, image_shape, stride=None):
    num_images, num_patches, patch_size, _, num_channel = patches.shape

    if stride is None:
        stride = patch_size

    # TODO: pass input size and replace 108 with it
    num_x_patches = len(range(0, image_shape[1] - 108, stride))
    num_y_patches = len(range(0, image_shape[2] - 108, stride))
    patches_per_image = int(num_patches / image_shape[0])

    images = np.zeros(shape=image_shape, dtype=patches.dtype)
    count_hits = np.zeros(shape=images.shape, dtype=np.uint64)

    #for n in range(0, image_shape[0]):
    #    patch_idx = 0
    #    for x in range(0, image_shape[1] - patch_size + 1, stride):
    #        for y in range(0, image_shape[2] - patch_size + 1, stride):
    #            #print("Images shape: ", images.shape)
    #            #print("Image area coords : ", n, x, x + patch_size, y, y + patch_size)
    #            #print("Image area shape : ", images[n, x:x + patch_size, y:y + patch_size].shape)
    #            print("Patches shape: ", patches.shape)
    #            print("Patches idx: ", patch_idx, " n: ", n)
    #            images[n, x:x + patch_size, y:y + patch_size] += patches[n, patch_idx]
    #            count_hits[n, x:x + patch_size, y:y + patch_size] += 1
    #            patch_idx += 1

    for n in range(0, image_shape[0]):
        patch_idx = 0
        for x in range(0, num_x_patches):
            for y in range(0, num_y_patches):
                x_pos = x * stride
                y_pos = y * stride
                images[n, x_pos:x_pos+patch_size, y_pos:y_pos+patch_size] += patches[n, patch_idx]
                count_hits[n, x_pos:x_pos+patch_size, y_pos:y_pos+patch_size] += 1
                patch_idx += 1

    images = images / count_hits

    return images

