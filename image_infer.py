import re
import cv2
import numpy
import numpy as np
import onnxruntime

copy_by = "https://github.com/Meituan-Dianping/vision-ui"  # 此处核心代码从vision-ui仓库copy而来
OP_NUM_THREADS = 4
IMAGE_INFER_MODEL_PATH = "model/ui_det_v2.onnx"


def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = numpy.maximum(x1[i], x1[order[1:]])
        yy1 = numpy.maximum(y1[i], y1[order[1:]])
        xx2 = numpy.minimum(x2[i], x2[order[1:]])
        yy2 = numpy.minimum(y2[i], y2[order[1:]])

        w = numpy.maximum(0.0, xx2 - xx1 + 1)
        h = numpy.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = numpy.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def yolox_preprocess(img, input_size, swap=(2, 0, 1)):
    if len(img.shape) == 3:
        padded_img = numpy.ones((input_size[0], input_size[1], 3), dtype=numpy.uint8) * 114
    else:
        padded_img = numpy.ones(input_size, dtype=numpy.uint8) * 114
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR,
    ).astype(numpy.uint8)
    padded_img[: int(img.shape[0] * r), : int(img.shape[1] * r)] = resized_img
    padded_img = padded_img.transpose(swap)
    padded_img = numpy.ascontiguousarray(padded_img, dtype=numpy.float32)
    return padded_img, r


def yolox_postprocess(outputs, img_size, p6=False):
    grids = []
    expanded_strides = []
    if not p6:
        strides = [8, 16, 32]
    else:
        strides = [8, 16, 32, 64]
    hsizes = [img_size[0] // stride for stride in strides]
    wsizes = [img_size[1] // stride for stride in strides]
    for hsize, wsize, stride in zip(hsizes, wsizes, strides):
        xv, yv = numpy.meshgrid(numpy.arange(wsize), numpy.arange(hsize))
        grid = numpy.stack((xv, yv), 2).reshape(1, -1, 2)
        grids.append(grid)
        shape = grid.shape[:2]
        expanded_strides.append(numpy.full((*shape, 1), stride))
    grids = numpy.concatenate(grids, 1)
    expanded_strides = numpy.concatenate(expanded_strides, 1)
    outputs[..., :2] = (outputs[..., :2] + grids) * expanded_strides
    outputs[..., 2:4] = numpy.exp(outputs[..., 2:4]) * expanded_strides
    return outputs


def multiclass_nms(boxes, scores, nms_thr, score_thr, class_agnostic=True):
    """Multiclass NMS implemented in Numpy"""
    if class_agnostic:
        nms_method = multiclass_nms_class_agnostic
    else:
        nms_method = multiclass_nms_class_aware
    return nms_method(boxes, scores, nms_thr, score_thr)


def multiclass_nms_class_agnostic(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-agnostic version."""
    cls_inds = scores.argmax(1)
    cls_scores = scores[numpy.arange(len(cls_inds)), cls_inds]

    valid_score_mask = cls_scores > score_thr
    if valid_score_mask.sum() == 0:
        return None
    valid_scores = cls_scores[valid_score_mask]
    valid_boxes = boxes[valid_score_mask]
    valid_cls_inds = cls_inds[valid_score_mask]
    keep = nms(valid_boxes, valid_scores, nms_thr)
    if keep:
        dets = numpy.concatenate(
            [valid_boxes[keep], valid_scores[keep, None], valid_cls_inds[keep, None]], 1
        )
    return dets


def multiclass_nms_class_aware(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy. Class-aware version."""
    final_dets = []
    num_classes = scores.shape[1]
    for cls_ind in range(num_classes):
        cls_scores = scores[:, cls_ind]
        valid_score_mask = cls_scores > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = boxes[valid_score_mask]
            keep = nms(valid_boxes, valid_scores, nms_thr)
            if len(keep) > 0:
                cls_inds = numpy.ones((len(keep), 1)) * cls_ind
                dets = numpy.concatenate(
                    [valid_boxes[keep], valid_scores[keep, None], cls_inds], 1
                )
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return numpy.concatenate(final_dets, 0)


def img_show(img, boxes, scores, cls_ids, conf=0.5, class_names=None):
    _COLORS = numpy.array([255, 0, 0,
                           195, 123, 40,
                           110, 176, 23]).astype(numpy.float32).reshape(-1, 3)
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        color = _COLORS[cls_id].astype(numpy.uint8).tolist()
        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        txt_color = (0, 0, 0) if numpy.mean(_COLORS[cls_id]) > 128 else (255, 255, 255)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 3)

        txt_bk_color = (_COLORS[cls_id] * 0.7).astype(numpy.uint8).tolist()
        cv2.rectangle(
            img,
            (x0, y0 + 1),
            (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
            txt_bk_color,
            -1
        )
        cv2.putText(img, text, (x0, y0 + txt_size[1]), font, 0.4, txt_color, thickness=1)

    return img


class ImageInfer(object):
    def __init__(self, model_path):
        self.UI_CLASSES = ("bg", "icon", "pic")
        self.input_shape = [640, 640]
        self.cls_thresh = 0.5
        self.nms_thresh = 0.2
        self.model_path = model_path
        so = onnxruntime.SessionOptions()
        so.intra_op_num_threads = OP_NUM_THREADS
        self.model_session = onnxruntime.InferenceSession(self.model_path, sess_options=so,
                                                          providers=['CPUExecutionProvider'])

    def ui_infer(self, image):
        origin_img = cv2.imread(image) if isinstance(image, str) else image
        img, ratio = yolox_preprocess(origin_img, self.input_shape)
        ort_inputs = {self.model_session.get_inputs()[0].name: img[None, :, :, :]}
        output = self.model_session.run(None, ort_inputs)
        predictions = yolox_postprocess(output[0], self.input_shape)[0]
        boxes = predictions[:, :4]
        scores = predictions[:, 4:5] * predictions[:, 5:]
        boxes_xyxy = np.ones_like(boxes)
        boxes_xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.
        boxes_xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.
        boxes_xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.
        boxes_xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.
        boxes_xyxy /= ratio
        dets = multiclass_nms(boxes_xyxy, scores, nms_thr=self.nms_thresh, score_thr=self.cls_thresh)
        if dets is not None:
            # 兼容不同版本模型返回结果中UI classes index起始位置
            offset = 0
            match_obj = re.match(r'.*o(\d+)\.onnx$', self.model_path)
            if match_obj:
                offset = int(match_obj.group(1))
            dets[:, 5] += offset
        return dets

    def show_infer(self, dets, origin_img, infer_result_path):
        if dets is not None:
            boxes, scores, cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
            origin_img = img_show(origin_img, boxes, scores, cls_inds, conf=self.cls_thresh,
                                  class_names=self.UI_CLASSES)
        cv2.imwrite(infer_result_path, origin_img)


image_infer = ImageInfer(IMAGE_INFER_MODEL_PATH)


def get_ui_infer(image, cls_thresh=0.3):
    """
    elem_det_region x1,y1,x2,y2
    """
    data = []
    image_infer.cls_thresh = cls_thresh if isinstance(cls_thresh, float) else image_infer.cls_thresh
    dets = image_infer.ui_infer(image)
    if isinstance(dets, np.ndarray):
        boxes, scores, cls_inds = dets[:, :4], dets[:, 4], dets[:, 5]
        for i in range(len(boxes)):
            box = boxes[i]
            box[box < 0] = 0
            box = box.tolist() if isinstance(box, (np.ndarray,)) else box
            elem_type = image_infer.UI_CLASSES[int(cls_inds[i])]
            score = scores[i]
            data.append(
                {
                    "elem_det_type": "image" if elem_type == 'pic' else elem_type,
                    "elem_det_region": box,
                    "probability": score
                }
            )
    return data
