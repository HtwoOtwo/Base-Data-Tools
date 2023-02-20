import cv2
import numpy as np
from enum import Enum
import matplotlib.pyplot as plt
# def draw_boxes(image, boxes, labels = None, color=(0, 0, 255), thickness=2):
#     font_scale = image.shape[0] / 500
#     font = cv2.FONT_HERSHEY_SIMPLEX
#     for box, label in zip(boxes, labels):
#         x1, y1, x2, y2 = box
#         cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
#         text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
#         text_x = x1 + (x2 - x1 - text_size[0]) / 2
#         text_y = y1 + (y2 - y1 + text_size[1]) / 2
#         cv2.putText(image, label, (int(text_x), int(text_y)), font, font_scale, color, thickness)
#     return image
class Color(Enum):
    """An enum that defines common colors.

    Contains red, green, blue, cyan, yellow, magenta, white and black.
    """
    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    cyan = (255, 255, 0)
    yellow = (0, 255, 255)
    magenta = (255, 0, 255)
    white = (255, 255, 255)
    black = (0, 0, 0)


def color_val(color):
    """Convert various input to color tuples.

    Args:
        color (:obj:`Color`/str/tuple/int/ndarray): Color inputs

    Returns:
        tuple[int]: A tuple of 3 integers indicating BGR channels.
    """
    if isinstance(color, str):
        return Color[color].value
    elif isinstance(color, Color):
        return color.value
    elif isinstance(color, tuple):
        assert len(color) == 3
        for channel in color:
            assert 0 <= channel <= 255
        return color
    elif isinstance(color, int):
        assert 0 <= color <= 255
        return color, color, color
    elif isinstance(color, np.ndarray):
        assert color.ndim == 1 and color.size == 3
        assert np.all((color >= 0) & (color <= 255))
        color = color.astype(np.uint8)
        return tuple(color)
    else:
        raise TypeError(f'Invalid type for color: {type(color)}')

def imshow(img, need_win = False, win_name='', wait_time=0):
    """Show an image.

    Args:
        img (str or ndarray): The image to be displayed.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
    """
    # 转RGB
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if need_win:
        cv2.imshow(win_name, img)
        if wait_time == 0:  # prevent from hanging if windows was closed
            while True:
                ret = cv2.waitKey(1)

                closed = cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1
                # if user closed window or if some key pressed
                if closed or ret != -1:
                    break
        else:
            ret = cv2.waitKey(wait_time)
    else:
        plt.imshow(img)
        plt.show()


def imshow_det_bboxes(img,
                      bboxes,
                      labels,
                      class_names=None,
                      score_thr=0,
                      bbox_color=None,
                      text_color=None,
                      thickness=1,
                      font_scale=0.5,
                      show=True,
                      win_name='',
                      wait_time=0):
    """Draw bboxes and class labels (with scores) on an image.
    Args:
        img (str or ndarray): The image to be displayed.
        bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
            (n, 5).
        labels (ndarray): Labels of bboxes.
        class_names (list[str]): Names of each classes.
        score_thr (float): Minimum score of bboxes to be shown.
        bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
        text_color (str or tuple or :obj:`Color`): Color of texts.
        thickness (int): Thickness of lines.
        font_scale (float): Font scales of texts.
        show (bool): Whether to show the image.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
    Returns:
        ndarray: The image with bboxes drawn on it.
    """
    bboxes = np.array(bboxes)
    labels = np.array(labels)
    assert bboxes.ndim == 2
    assert labels.ndim == 1
    assert bboxes.shape[0] == labels.shape[0]
    assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
    img = cv2.imread(img)
    img = np.ascontiguousarray(img)

    if score_thr > 0:
        assert bboxes.shape[1] == 5
        scores = bboxes[:, -1]
        inds = scores > score_thr
        bboxes = bboxes[inds, :]
        labels = labels[inds]

    if bbox_color or text_color is None:
        bbox_color = []
        text_color = []
        for c in Color:
            bbox_color.append(color_val(c))
            text_color.append(color_val(c))

    for bbox, label in zip(bboxes, labels):
        bbox_int = bbox.astype(np.int32)
        left_top = (bbox_int[0], bbox_int[1])
        right_bottom = (bbox_int[2], bbox_int[3])
        idx = label % len(bbox_color)
        cv2.rectangle(
            img, left_top, right_bottom, bbox_color[idx], thickness=thickness)
        label_text = class_names[
            label] if class_names is not None else f'cls {label}'
        if len(bbox) > 4:
            label_text += f'|{bbox[-1]:.02f}'
        cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                    cv2.FONT_HERSHEY_COMPLEX, font_scale, text_color[idx])

    if show:
        imshow(img, win_name, wait_time)
    return img