import numpy as np
import scipy
from skimage import io, measure, morphology
from skimage.filters import rank, roberts, threshold_li
from skimage.measure import label
from skimage.morphology import erosion, closing, dilation, skeletonize
from skimage.segmentation import watershed
from skimage.transform import resize

################################################################################
def pruning(im, outline_val = 1.0):
    im = np.array(im, np.int)
    elem = np.array([
        [16, 32,  64], 
        [ 8,  1, 128], 
        [ 4,  2, 256]
    ])
    val = np.array([1, 3, 5, 9, 17, 33, 65, 129, 257, 7, 13, 25, 49, 97, 193, 259, 385])
    count = True
    while count == True:
        count = False
        diff = scipy.ndimage.convolve(im, elem, mode='constant', cval=outline_val)
        diff[~np.array(im, np.bool)] = 0
        diff = np.isin(diff, val) 
        if np.any(diff>0):
            im = np.subtract(im, diff)
            count = True
    return im

def create_markers(im, min_size = 3, connectivity=2):
    im = rank.mean(im, np.ones((3,3)))
    im = im > threshold_li(im)
    im = erosion(im, np.ones((3,3)))
    labels = measure.label(im, connectivity=connectivity)
    
    labels_flat = labels.ravel()
    labels_count = np.bincount(labels_flat)
    index = np.argsort(labels_flat)[labels_count[0]:]
    coordinates = np.column_stack(np.unravel_index(index, im.shape))
    lab = np.cumsum(labels_count[1:])

    im = np.zeros(im.shape, np.uint8)
    it = dict(enumerate(np.split(coordinates, lab), start=1))
    for _, indexes in it.items():
        if len(indexes) < min_size:
            continue
        center = np.mean(indexes, axis=0)
        y, x = int(center[0]), int(center[1])
        im[y,x] = 255
    return im

def local_maximum_shift(im, markers, radius = 1):
    assert(radius >= 1)
    y_lim, x_lim = im.shape[0], im.shape[1]
    new_markers = np.zeros(markers.shape)
    for (y,x) in np.argwhere(markers>0):
        indexes = []
        for n in range(-radius, radius+1):
            for m in range(-radius, radius+1):
                if (x+m)>0 and (y+n)>0 and (y+n)<y_lim and (x+m)<x_lim:
                    indexes.append((y+n, x+m))
        y_min, x_min = indexes[0]
        for (y,x) in indexes:
            if im[y_min, x_min] < im[y,x]:
                y_min, x_min = y,x
        new_markers[y_min, x_min] = 1
    return new_markers
def unpad2D(array, pad_width):
    width, height = array.shape
    assert(width > 2*pad_width and height > 2*pad_width)
    return array[pad_width:width-pad_width, pad_width:height-pad_width]
################################################################################

def postprocess_watershed(im, markers, roi, create_markers = True, pad_width = 5):
    if create_markers == True:
        markers = create_markers(markers)
    markers = local_maximum_shift(im, markers, radius=1)
    im = np.pad(im, pad_width, 'constant', constant_values=0)
    roi = np.pad(roi, pad_width, 'constant', constant_values=0)
    markers = np.pad(markers, pad_width, 'constant', constant_values=0)

    markers = label(markers, connectivity=1)
    im = watershed(im,markers, watershed_line=False)
    im[roi == 0] = 0
    im = roberts(im) > 0
    im = closing(im, morphology.disk(3))
    im = skeletonize(im)
    im = pruning(im, 1)
    im = unpad2D(im, pad_width)
    return np.array(255*im, np.uint8, copy=False)

#Use this version of watershed algorithm if you do not have ROI mask
def postprocess_watershed_no_roi(im, markers, create_markers = True, pad_width=5):
    if create_markers == True:
        markers = create_markers(markers)
    markers = local_maximum_shift(im, markers, radius=1)
    im = np.pad(im, pad_width, 'constant', constant_values=0)
    markers = np.pad(markers, pad_width, 'constant', constant_values=0)

    markers = label(markers, connectivity=1)
    im = watershed(im,markers, watershed_line=True) == 0
    im = closing(im, morphology.disk(3))
    im = skeletonize(im)
    im = pruning(im, 1)
    im = unpad2D(im, pad_width)
    return np.array(255*im, np.uint8, copy=False)

################################################################################


if __name__ == '__main__':
    pred = io.imread('pred.png', as_gray=False)
    roi = io.imread('roi.png', as_gray=True)

    im = pred[:,:,1]
    markers = pred[:,:,2]

    result = postprocess_watershed(im, markers, roi)

    io.imsave('postprocessed.png', result)
