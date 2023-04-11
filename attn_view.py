import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import skimage
import cv2

from PIL import Image

cmap = matplotlib.cm.get_cmap('jet')
cmap.set_bad(color="k", alpha=0.0)

def attention_bbox_interpolation(im, bboxes, att):
    softmax = att
    assert len(softmax) == len(bboxes)

    img_h, img_w = im.shape[:2]
    opacity = np.zeros((img_h, img_w), np.float32)
    for bbox, weight in zip(bboxes, softmax):
        x1, y1, x2, y2 = bbox
        # opacity[int(y1):int(y2), int(x1):int(x2)] += weight
        opacity[int(y1*img_h):int(y2*img_h), int(x1*img_w):int(x2*img_w)]+= weight
    opacity = np.minimum(opacity, 1)

    opacity = opacity[..., np.newaxis]
    
    vis_im = np.array(Image.fromarray(cmap(opacity, bytes=True), 'RGBA'))
    vis_im = vis_im.astype(im.dtype)
    vis_im = cv2.addWeighted(im, 0.7, vis_im, 0.5, 0)
    vis_im = vis_im.astype(im.dtype)
    
    return vis_im


def attention_grid_interpolation(im, att):
    softmax = np.reshape(att, (14, 14))
    opacity = skimage.transform.resize(softmax, im.shape[:2], order=3)
    opacity = opacity[..., np.newaxis]
    opacity = opacity*0.95+0.05

    vis_im = opacity*im + (1-opacity)*255
    vis_im = vis_im.astype(im.dtype)
    return vis_im

def visualize_pred(im_path, boxes, att_weights):
    im = cv2.imread(im_path)
    im = cv2.cvtColor(im, cv2.COLOR_RGB2RGBA)
    b,g,r,a = cv2.split(im)           # get b, g, r
    im = cv2.merge([r,g,b,a])

    M = min(len(boxes), len(att_weights))
    im_ocr_att = attention_bbox_interpolation(im, boxes[:M], att_weights[:M])
    plt.imshow(im_ocr_att)

def get_blend_map(self,img, att_map, blur=True, overlap=True):
        att_map -= att_map.min()
        if att_map.max() > 0:
            att_map /= att_map.max()
        att_map = skimage.transform.resize(att_map, (img.shape[:2]), order = 3)
        if blur:
            att_map = skimage.filters.gaussian(att_map, 0.02*max(img.shape[:2]))
            att_map -= att_map.min()
            att_map /= att_map.max()
        cmap = plt.get_cmap('jet')
        att_map_v = cmap(att_map)
        att_map_v = np.delete(att_map_v, 3, 2)
        plt.imshow(att_map_v)
        #plt.imshow(img,alpha=0.2)    
        plt.show()
        # cv2.imwrite("attentionmap.jpg",att_map_v*255)
        return att_map_v
