from deep_sort.application_util import preprocessing as prep
from deep_sort.application_util import visualization
from deep_sort.deep_sort.detection import Detection
from deep_sort.deep_sort import nn_matching
from deep_sort.deep_sort.tracker import Tracker
from deep_sort import generate_detections
import numpy as np
import cv2

metric = nn_matching.NearestNeighborDistanceMetric(
    "cosine", 0.2, 100)
tracker = Tracker(metric)
encoder = generate_detections.create_box_encoder(
        "deepsort/resources/networks/mars-small128.ckpt-68577")

def sort(boxResults, imgcv):
	detections = []
	scores = []
	nms_max_overlap = 0.1
	if type(imgcv) is not np.ndarray:
		imgcv = cv2.imread(imgcv)
	h, w, _ = imgcv.shape
	thick = int((h + w) // 300)
	for boxresult in boxResults:
		# left, right, top, bot, mess, max_indx, confidence = boxResults
		max_indx = 20
		print (boxresult)
		top, left, bot, right, mess, confidence  = boxresult['topleft']['y'],boxresult['topleft']['x'],boxresult['bottomright']['y'],boxresult['bottomright']['x'], boxresult['label'], boxresult['confidence']
		mess = boxresult['label']
		# if mess not in self.FLAGS.trackObj :
		# 			continue

		# detections.append(np.array([right-left,bot-top,left,top]).astype(np.float64))
		detections.append(np.array([left,top,right-left,bot-top]).astype(np.float64))
		scores.append(confidence)
		print (np.array([left,top,right-left,bot-top]).astype(np.float64))
		print (imgcv.shape)

	detections = np.array(detections)
	scores = np.array(scores)

	features = encoder(imgcv, detections.copy())
	detections = [
	            Detection(bbox, score, feature) for bbox,score, feature in
	            zip(detections,scores, features)]
	# Run non-maxima suppression.
	boxes = np.array([d.tlwh for d in detections])
	scores = np.array([d.confidence for d in detections])
	indices = prep.non_max_suppression(boxes, nms_max_overlap, scores)
	detections = [detections[i] for i in indices]
	tracker.predict()
	tracker.update(detections)
	trackers = tracker.tracks

	for track in trackers:

		if not track.is_confirmed() or track.time_since_update > 1:
			continue
		bbox = track.to_tlbr()
		id_num = str(track.track_id)


		cv2.rectangle(imgcv, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),
				        (255,255,255), thick//3)
		cv2.putText(imgcv, id_num,(int(bbox[0]), int(bbox[1]) - 12),0, 1e-3 * h, (255,255,255),thick//6)
	
	return imgcv