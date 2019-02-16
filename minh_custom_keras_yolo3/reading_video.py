import cv2

cap = cv2.VideoCapture('test-video.mp4')

if not cap.isOpened():
    print("Error opening video stream or file")

frame_width = int(cap.get(3))
frame_height = int(cap.get(4))
out = cv2.VideoWriter('output-test-video.mp4',cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 30, (frame_width, frame_height))

bs = cv2.createBackgroundSubtractorMOG2(detectShadows=True)

yolo_labels = open("custom-keras-yolo3/output_labels.txt", 'r')


def get_iou(x1, y1, x2, y2, x3, y3, x4, y4):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert x1 < x2
    assert y1 < y2
    assert x3 < x4
    assert y3 < y4

    # determine the coordinates of the intersection rectangle
    x_left = max(x1, x3)
    y_top = max(y1, y3)
    x_right = min(x2, x4)
    y_bottom = min(y2, y4)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (x2 - x1) * (y2 - y1)
    bb2_area = (x4 - x3) * (y4 - y3)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou


def get_iou_mapping(boxes1, boxes2, thresh=0.3):
    mapped_list = []
    _boxes2 = boxes2.copy()

    for a in boxes1:
        max_iou = 0
        max_box = None

        for b in _boxes2:
            iou = get_iou(a[0], a[1], a[2], a[3], b[0], b[1], b[2], b[3])
            if iou > max_iou:
                max_iou = iou
                max_box = b

        if max_iou >= thresh:
            mapped_list.append(max_box)
            _boxes2.remove(max_box)
        else:
            mapped_list.append(None)

    return mapped_list


def get_motion_boxes(fg):
    motion_boxes = []
    th = cv2.threshold(fg.copy(), 244, 255, cv2.THRESH_BINARY)[1]
    dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
    image, contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Getting motion_boxes using Algorithm
    for c in contours:
        if cv2.contourArea(c) > 3000:
            (x, y, w, h) = cv2.boundingRect(c)
            motion_boxes.append((x, y, x+w, y+h))

    return motion_boxes


saved_boxes = []

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        yolo_boxes = []

        # Reading yolo_boxes from output_labels.txt
        try:
            num_obj = int(yolo_labels.readline())

            for obj in range(num_obj):
                yolo_boxes.append([int(x) for x in yolo_labels.readline().replace("\n", "").strip().split(" ")[2:]])

        except Exception as e:
            print(e)
            pass

        fg = bs.apply(frame)
        if fg is not None:
            motion_boxes = get_motion_boxes(fg)

            mapped_list = get_iou_mapping(yolo_boxes, motion_boxes, 0.3)

            person_boxes = []

            for i in range(len(mapped_list)):
                if mapped_list[i] is not None:
                    person_boxes.append(yolo_boxes[i])

            unmapped_motion_boxes = motion_boxes.copy()
            for mbox in mapped_list:
                if mbox is not None:
                    unmapped_motion_boxes.remove(mbox)

            saved_boxes_mapping = get_iou_mapping(unmapped_motion_boxes, saved_boxes)

            for sbox in saved_boxes_mapping:
                if sbox is not None:
                    # box nay la box co motion, khong co yolo, dc save o frame trc do
                    person_boxes.append(sbox)

            # update saved_boxes
            saved_boxes = person_boxes.copy()


            for p in person_boxes:
                cv2.rectangle(frame, (p[0], p[1]), (p[2], p[3]), (255, 0, 255), 2)

            cv2.putText(frame, 'Persons: ' + str(len(person_boxes)), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2, cv2.LINE_AA)

            cv2.imshow("detection", frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    else:
        break

cap.release()
out.release()

cv2.destroyAllWindows()