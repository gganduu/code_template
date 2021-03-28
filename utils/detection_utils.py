import numpy as np
import collections

def iou(bbox1, bbox2):
    '''
    this function is to calculate the iou
    :param bbox1: boarding box1
    :param bbox2: boarding box2
    :return: iou if two bboxes intersect else 0
    '''
    x1min, y1min, x1max, y1max = bbox1
    x2min, y2min, x2max, y2max = bbox2
    # use two boarding boxes' four edges to judge intersection
    if x1min >= x2max or x1max <= x2min or y1min >= y2max or y1max <= y2min:
        return 0
    intersection = (min(x1max, x2max) - max(x1min, x2min)) * (min(y1max, y2max) - max(y1min, y2min))
    union = (x1max - x1min) * (y1max - y1min) + (x2max - x2min) * (y2max - y2min) - intersection
    ret = intersection / union
    return ret

def nms(pred, iou_threshold=0.4, confidence_threshold=0.5):
    '''
    this function is non-maximum-suppression, which is filtering the overlapping bbox by given iou and confidence threshold
    :param pred: a object detection model predictions for one image, it contains boxes, labels and scores
    :param iou_threshold: a given iou threshold
    :param confidence_threshold: a given confidence threshold
    :return: a dictionary which key is classification labels
    '''
    # use confidence threshold to filter boxes, labels and scores
    scores = pred.get('scores')
    mask = scores > confidence_threshold
    boxes = pred.get('boxes')[mask]
    labels = pred.get('labels')[mask]
    scores = scores[mask]
    # ret should be organized by each label, eg. {'label': {'boxes':[[x1,y1,x2,y2]], 'scores':[0.88]}}
    ret = dict()
    unique_labels = set(labels.numpy())
    for u_l in unique_labels:
        ret[u_l] = {'boxes': [], 'scores': []}
        # to filter scores and boxes from each label
        label_mask = labels == u_l
        label_scores = scores[label_mask]
        label_boxes = boxes[label_mask]
        # get the descending scores indexes
        loc = list(np.argsort(-label_scores.detach().numpy()))
        while len(loc) > 0:
            ret[u_l].get('boxes').append(label_boxes[loc[0]])
            ret[u_l].get('scores').append(label_scores[loc[0]])
            if len(loc) == 1:
                loc.clear()
            else:
                # calculate iou of max score bbox and other bboxes one by one
                # if iou is great than or equal to iou_threshold, then delete it
                # after loop one round, do the same thing on the rest of bboxes
                for l in loc[1:]:
                    result = iou(label_boxes[loc[0]], label_boxes[l])
                    if result >= iou_threshold:
                        loc.remove(l)
                loc = loc[1:]
    return ret

def smooth_calc_ap(precisions, recalls):
    '''
    this function is to smooth PR curve and calculate the ap
    :param precisions: a precision list
    :param recalls: a recall list
    :return: average precision
    '''
    # first append sentinel values at the end
    mrec = np.concatenate(([0.], recalls, [1.]))
    mpre = np.concatenate(([0.], precisions, [0.]))
    # compute the precision, use interpolation algorithm to achieve a smooth precision
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]
    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap

def calc_loc(value, value_list, start, end):
    '''
    this function is to calculate the value location in the given descending list.
    the given value may not be equal with elements of value_list
    this is just getting the location if put the value in the list
    :param value: a given value
    :param value_list: a descending list
    :param start: value_list start location
    :param end: the length of value_list
    :return: location of given value, if list length is less than 1, return -1
    '''
    if value > value_list[0]:
        return -1
    elif value == value_list[0]:
        return 0
    elif value <= value_list[-1]:
        return len(value_list) - 1
    else:
        mid = int(start + (end - start) / 2)
        if value == value_list[mid]:
            return mid
        elif value > value_list[mid] and value <= value_list[mid-1]:
            return mid-1
        elif value < value_list[mid] and value >= value_list[mid+1]:
            return mid
        elif value > value_list[mid]:
            return calc_loc(value, value_list, start, mid-1)
        else:
            return calc_loc(value, value_list, mid+1, end)

def calc_map(preds, truths, iou_threshold=0.4):
    '''
    this function is to calculate mean average precision
    :param preds: a batch/list of predictions, it contains boxes, labels and scores
    :param truths: a batch/list of ground truths, the same format with preds
    like: {'labels': tensor([12, 15]), 'boxes': tensor([[ 48., 240., 195., 371.],
        [  8.,  12., 352., 498.]]), 'image_id': tensor(0)}
    :param iou_threshold: a threshold of iou
    :param confidence_threshold: a threshold of confidence
    :return: a map of this batch and a dictionary which contains class ap values
    '''
    # get a batch statistic of ground truth for each class
    counter = collections.Counter([int(l) for t in truths for l in t.get('labels')])
    mAP = 0.0
    cls_ap = dict()
    for label, counts in counter.items():
        # get predict boxes, scores and ground truth for given label
        p_loc = [True if l == label else False for p in preds for l in p.get('labels')]
        t_loc = [True if l == label else False for t in truths for l in t.get('labels')]
        p_boxes = np.array([b.numpy() for p in preds for b in p.get('boxes')])[p_loc]
        t_boxes = np.array([b.numpy() for t in truths for b in t.get('boxes')])[t_loc]
        p_scores = np.array([s for p in preds for s in p.get('scores')])[p_loc]
        # descend the scores
        sort_loc = np.argsort(-p_scores)
        sort_p_boxes = p_boxes[sort_loc]
        label_TP = np.zeros_like(p_scores)
        label_FP = np.ones_like(p_scores)
        if len(p_scores) != 0:
            for t_box in t_boxes:
                ious = np.zeros_like(p_scores)
                for j, p_box in enumerate(sort_p_boxes):
                    res = iou(t_box, p_box)
                    ious[j] = res
                # get the max iou and its location
                max_loc = np.argmax(ious)
                max_iou = ious[max_loc]
                # mark max iou's bbox as TP, others are marked as FP
                if max_iou >= iou_threshold:
                    label_TP[max_loc] = 1
                    label_FP[max_loc] = 0
        # accumulate TP and FP
        cum_TP = np.cumsum(label_TP)
        cum_FP = np.cumsum(label_FP)
        if cum_TP is not None and cum_FP is not None:
            precisions = cum_TP/(cum_TP+cum_FP+np.finfo(np.float).eps)
            recalls = cum_TP/counts
        else:
            raise Exception('cum_TP or cum_FP calculation has errors')
        label_ap = smooth_calc_ap(precisions, recalls)
        cls_ap[label] = label_ap
        mAP += label_ap
    return mAP/len(counter.keys()), cls_ap

if __name__ == '__main__':
    # pred = {'boxes': torch.Tensor(
    #     [[1894.8378, 1496.1295, 1954.3877, 1532.4030], [1883.7123, 1485.4585, 1959.0841, 1543.7224]]),
    #         'labels': torch.IntTensor([1, 1]),
    #         'scores': torch.Tensor([0.881, 0.9468, ])}
    # print(iou(pred.get('boxes')[0], pred.get('boxes')[1]))
    # nms(pred)
    pre = [1.0, 1.0, 0.66, 0.5, 0.4, 0.5, 0.43, 0.38, 0.44, 0.5]
    rec = [0.14, 0.29, 0.29, 0.29, 0.29, 0.43, 0.43, 0.43, 0.57, 0.71]
    ap = smooth_calc_ap(pre, rec)
    print(ap)

    # # l1 = [9,8,7,6,5,4,3,2,1]
    # l1 = [3,1]
    # loc = calc_loc(2, l1, 0, len(l1))
    # print(loc)
