COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def get_prediction(model, img, threshold):
    pred = model(img)  # Pass the image to the model
    pred_class = [[COCO_INSTANCE_CATEGORY_NAMES[i] for i in list(p['labels'].cpu().numpy())] for p in pred]  # Get the Prediction Score
    pred_boxes = [[[(int(i[0]), int(i[1])), (int(i[2]), int(i[3]))] for i in list(p['boxes'].detach().cpu().numpy())] for p in pred]  # Bounding boxes
    pred_score = [list(p['scores'].detach().cpu().numpy()) for p in pred]

    try:
        if any([max(score) < threshold for score in pred_score]):
            return [], []

        pred_t = [[p_score.index(x) for x in p_score if x > threshold][-1] for p_score in pred_score]  # Get list of index with score greater than threshold.

        pred_boxes = [pred_boxes[i][:p_t+1] for i, p_t in enumerate(pred_t)]
        pred_class = [pred_class[i][:p_t+1] for i, p_t in enumerate(pred_t)]

        return pred_boxes, pred_class

    except:
        return [], []