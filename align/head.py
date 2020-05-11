from PIL import Image
import argparse
import os
import cv2
import time

from detector import detect_faces
from visualization_utils import show_results
from tracker.tracker import DeepSORT
from reID.extractor import AlignedReID, AlignedReID_Plus, StrongBaseline


'''
example for run the script
python head.py --video /home/aicenter/Documents/hsu/hteam/20200506-140039CH04.avi \
--export /home/aicenter/Documents/hsu/hteam/ \
--nn_budget=100 --maxage 15
'''


class Config(object):
    def __init__(self):
        parser = argparse.ArgumentParser(
            description='head tracking'
        )

        parser.add_argument(
            "--export",
            dest='export',
            help="the file name to export with frame having something detected",
            default="output",
            type=str)

        parser.add_argument(
            "--videofolder",
            dest='video_folder',
            help="Run all videos in this folder",
            default=None,
            type=str)
        parser.add_argument(
            "--video",
            dest='video',
            help="Video to run detection upon",
            default=None,
            type=str)

        #############
        # deep sort #
        #############

        parser.add_argument(
            "--maxdist",
            dest='maxdist',
            help="Max cosine distance between deep feature vectors",
            default="0.2",
            type=float)
        parser.add_argument(
            "--maxage",
            dest='max_age',
            help="last time for a tracker",
            default="10",
            type=int)
        parser.add_argument(
            "--budget",
            dest='nn_budget',
            help="features amount in a tracker",
            default="20",
            type=int)

        args = parser.parse_known_args()[0]

        # I/O
        self.export = args.export

        # for yolov3 detection usage
        self.video = args.video
        self.video_folder = args.video_folder

        # deep sort
        self.maxdist = args.maxdist
        self.max_age = args.max_age
        self.nn_budget = args.nn_budget


if __name__ == "__main__":

    cfg = Config()

    # parameters of I/O
    video_file = cfg.video
    file_folder = cfg.video_folder
    video_sequences = []
    if file_folder != None:
        video_sequences = os.listdir(file_folder)
        video_sequences = [
            file_folder + "/" + filename for filename in video_sequences
            if filename.endswith(".avi")
        ]
        video_sequences = sorted(video_sequences)
        assert len(video_sequences) != 0, 'the folder is empty'
    else:
        video_sequences = [video_file]

    # export to video
    tmp_video = ''
    if len(video_sequences) != 0:
        tmp_video = video_sequences[0]
    else:
        tmp_video = video_file
    cap = cv2.VideoCapture(tmp_video)
    assert cap.isOpened(), 'Cannot capture source'
    # parameters for recording video
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    video_width, video_height = int(cap.get(3)), int(cap.get(4))
    output_pos = cfg.export
    export_path = output_pos + 'head_tracking'
    out = cv2.VideoWriter(
        '{0}.avi'.format(export_path), fourcc, cap.get(5),
        (video_width, video_height))  # file, fourcc, fps, (w, h)

    # parameters of tracker
    max_cosine_distance = cfg.maxdist
    nn_budget = cfg.nn_budget
    max_age = cfg.max_age
    tracker = DeepSORT(
        max_dist=max_cosine_distance, nn_budget=nn_budget, max_age=max_age)
    extractor = AlignedReID_Plus(visualize=False)

    # Single Object Tracking
    # extract the OpenCV version info
    (major, minor) = cv2.__version__.split(".")[:2]

    # if we are using OpenCV 3.2 OR BEFORE, we can use a special factory
    # function to create our object tracker
    if int(major) == 3 and int(minor) < 3:
        tracker = cv2.Tracker_create(args["tracker"].upper())
    # otherwise, for OpenCV 3.3 OR NEWER, we need to explicity call the
    # approrpiate object tracker constructor:
    else:
        # initialize a dictionary that maps strings to their corresponding
        # OpenCV object tracker implementations
        OPENCV_OBJECT_TRACKERS = {
            "csrt": cv2.TrackerCSRT_create,
            "kcf": cv2.TrackerKCF_create,
            "boosting": cv2.TrackerBoosting_create,
            "mil": cv2.TrackerMIL_create,
            "tld": cv2.TrackerTLD_create,
            "medianflow": cv2.TrackerMedianFlow_create,
            "mosse": cv2.TrackerMOSSE_create
        }
        # grab the appropriate object tracker using our dictionary of
        # OpenCV object tracker objects
        temp_tracker = OPENCV_OBJECT_TRACKERS['kcf']()
    # initialize the bounding box coordinates of the object we are going
    # to track
    initBB = None
    monitor_pos = (0, 0)

    frames = 0
    start = time.time()
    for file in video_sequences:
        # start running video
        cap = cv2.VideoCapture(file)
        assert cap.isOpened(), 'Cannot capture source'

        while cap.isOpened():
            ret, frame = cap.read()
            if ret:
                # Sinlge Object Tracking
                if initBB is not None:
                    # grab the new bounding box coordinates of the object
                    (success, box) = temp_tracker.update(frame)
                    # check to see if the tracking was a success
                    if success:
                        (x, y, w, h) = [int(v) for v in box]
                        cv2.rectangle(
                            frame, (x, y), (x + w, y + h), (255, 90, 143), 2)
                        monitor_pos = (y+h/2, x+w/2)
                else:
                    # select the bounding box of the object we want to track (make
                    # sure you press ENTER or SPACE after selecting the ROI)
                    initBB = cv2.selectROI("Frame", frame, fromCenter=False,
                                           showCrosshair=True)
                    cv2.destroyAllWindows()
                    # start OpenCV object tracker using the supplied bounding box
                    # coordinates, then start the FPS throughput estimator as well
                    temp_tracker.init(frame, initBB)

                crop_top, crop_left = (220, 900)
                image = frame[crop_top:(crop_top + 600),
                              crop_left:(crop_left + 500), :]
                # cv2.imshow('crop', image)
                # key = cv2.waitKey(1)
                # if key & 0xFF == ord('q'):
                #     break
                # continue
                # detect head
                # detect bboxes and landmarks for all faces in the image
                image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                head_bboxes, _ = detect_faces(image)

                detections = []
                for bbox in head_bboxes:
                    left = int(bbox[0]) + crop_left
                    top = int(bbox[1]) + crop_top
                    right = int(bbox[2]) + crop_left
                    bottom = int(bbox[3]) + crop_top
                    w = right - left
                    h = bottom - top
                    detection, detec_confid = [left, top, w, h], 1.0
                    detections.append((detection, detec_confid))

                # run the following code only when frame contains person
                if len(detections) != 0:
                    global_features = extractor.extract_feature(
                        detections, frame)
                    tracker.update(detections, global_features)
                    tracker.update_temp_status(monitor_pos)
                    tracker.draw_tracks(
                        frame, draw_detection=True, detections=detections)
                else:
                    tracker.update([], [])

                out.write(frame)
                cv2.imshow('camera', frame)

                key = cv2.waitKey(1)
                if key & 0xFF == ord('q'):
                    break
                frames += 1
                print("FPS of the video is {:5.2f}".format(
                    frames / (time.time() - start)))
            else:
                break

    # 釋放所有資源
    cap.release()
    out.release()
    cv2.destroyAllWindows()
