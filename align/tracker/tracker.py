import cv2

from .deepsort.deep_sort import nn_matching
from .deepsort.deep_sort.tracker import Tracker
from .deepsort.deep_sort.detection import Detection


class DeepSORT():
    def __init__(self, max_dist=0.2, nn_budget=50, max_age=20):
        self.metric = nn_matching.NearestNeighborDistanceMetric(
            "cosine", max_dist, nn_budget)
        self.tracker = Tracker(self.metric, max_age=max_age)

    def update(self, detections, features):
        self.tracker.predict()
        self.tracker.update(self.create_detection(detections, features))

    def create_detection(self, detections, features):
        detectionList = []
        for detect, feature in zip(detections, features):
            bbox = detect[0]  # [left, top, w, h]
            detec_confid = detect[1]
            detectionList.append(Detection(bbox, detec_confid, feature))
        return detectionList

    def draw_tracks(self, image, draw_detection=False, detections=[]):
        colors = [(0, 0, 255), (214, 15, 60)]
        for track in self.tracker.tracks:
            color = colors[track.temp_check]
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            label = str(track.track_id)
            tc1 = tuple(map(int, track.to_tlwh()[:2]))
            tc2 = tuple(
                map(int,
                    [tc1[0] + track.to_tlwh()[2], tc1[1] + track.to_tlwh()[3]
                     ]))
            cv2.rectangle(image, tc1, tc2, color, 1)
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                                     1)[0]
            tc2 = tc1[0] + t_size[0] + 3, tc1[1] + t_size[1] + 10
            cv2.rectangle(image, tc1, tc2, color, -1)
            cv2.putText(image, label, (tc1[0], tc1[1] + t_size[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, [255, 255, 255], 1)
        if draw_detection:
            for detection in detections:
                c1 = tuple(map(int, detection[0][:2]))
                c2 = tuple(
                    map(int,
                        [c1[0] + detection[0][2], c1[1] + detection[0][3]]))
                cv2.rectangle(image, c1, c2, color[1], 1)

    def update_temp_status(self, thermometer_pos):
        for track in self.tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 0:
                continue
            tc1 = tuple(map(int, track.to_tlwh()[:2]))
            center_x = int(tc1[1] + track.to_tlwh()[3]/2)
            center_y = int(tc1[0] + track.to_tlwh()[2]/2)
            target_x, target_y = thermometer_pos
            dist = ((target_x-center_x)**2 + (target_y-center_y)**2)
            print(center_x, center_y)
            print(target_x, target_y)
            print(dist)
            if dist < 8000:
                track.temp_check = True
