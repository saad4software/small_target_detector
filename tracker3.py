import numpy as np
from scipy.optimize import linear_sum_assignment
from datetime import datetime
from util import *
from k_means import *
from datetime import datetime

class Track(object):

    def __init__(self, box, trackIdCount):

        self.track_id = trackIdCount  # identification of each track object
        self.skipped_frames = 0  # number of frames skipped undetected
        self.trace = [box]  # trace path
        self.distances = [0]
        self.start_time = datetime.utcnow()
        self.passed = False

    def is_target(self):
        m = np.mean(self.trace)
        for t in self.trace:
            if np.mean(t) != m:
                return True
        return False



class Tracker(object):


    def __init__(self, dist_thresh, max_frames_to_skip, max_trace_length,
                 trackIdCount, model):
        """Initialize variable used by Tracker class
        Args:
            dist_thresh: distance threshold. When exceeds the threshold,
                         track will be deleted and new track is created
            max_frames_to_skip: maximum allowed frames to be skipped for
                                the track object undetected
            max_trace_lenght: trace path history length
            trackIdCount: identification of each track object
        Return:
            None
        """
        self.dist_thresh = dist_thresh
        self.max_frames_to_skip = max_frames_to_skip
        self.max_trace_length = max_trace_length
        self.tracks = []
        self.trackIdCount = trackIdCount

        self.model = model


    def is_target_cnn(self, frame, box):
        x, y, w, h = box

        crop1 = cv2.resize(imcrop(frame, (x, y, x + w, y + h)), (20, 20), interpolation = cv2.INTER_LINEAR)

        kernel = [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]

        crop1 = cv2.filter2D(crop1,-1,np.array(kernel))

        prediction = self.model.predict(np.array([crop1]))
        prediction = prediction[0] == max(prediction[0])
        target_class = "target" if prediction[0] else "noise"
        print(target_class)
        return target_class == "target", box


    def update(self, bboxes, frame):
        if len(self.tracks) == 0:
            for i in range(len(bboxes)):
                track = Track(bboxes[i], self.trackIdCount)
                self.trackIdCount += 1
                self.tracks.append(track)

        # Calculate cost using sum of square distance between
        # predicted vs detected centroids
        N = len(self.tracks)
        M = len(bboxes)
        cost = np.zeros(shape=(N, M))   # Cost matrix

        new_centers = boxes2centers2(bboxes)

        for i in range(len(self.tracks)):
            for j in range(len(bboxes)):
                center = box2center(self.tracks[i].trace[-1])
                diff = np.array([center]) - np.array(new_centers[j])
                distance = np.sqrt(diff[0][0]*diff[0][0] +
                                    diff[0][1]*diff[0][1])
                cost[i][j] = distance

        # Let's average the squared ERROR
        cost = (0.5) * cost

        # Using Hungarian Algorithm assign the correct detected measurements
        # to predicted tracks
        assignment = [-1 for _ in range(N)]
        # for _ in range(N):
        #     assignment.append(-1)
        row_ind, col_ind = linear_sum_assignment(cost)

        for i in range(len(row_ind)):
            assignment[row_ind[i]] = col_ind[i]

        # Identify tracks with no assignment, if any
        un_assigned_tracks = []
        for i in range(len(assignment)):
            if assignment[i] != -1:
                # check for cost distance threshold.
                # If cost is very high then un_assign (delete) the track

                if cost[i][assignment[i]] > self.dist_thresh:
                    assignment[i] = -1
                    un_assigned_tracks.append(i)
                pass
            else:
                self.tracks[i].skipped_frames += 1

        # If tracks are not detected for long time, remove them
        del_tracks = [i for i in range(len(self.tracks)) if self.tracks[i].skipped_frames > self.max_frames_to_skip]

        for i in range(len(self.tracks)):
            box = self.tracks[i].trace[-1]
            # is_target, new_box = kmeans_is_target(frame, box)
            is_target, new_box = self.is_target_cnn(frame, box)
            
            
            if is_target:
                print("new self feed")
                self.tracks[i].trace.append(new_box)

                last_box = self.tracks[i].trace[-1]
                new_center = box2center(new_box)
                last_center = box2center(last_box)
                new_distance = centers_distance(new_center, last_center)

                self.tracks[i].distances.append(new_distance)      


        if len(del_tracks) > 0:  # only when skipped frame exceeds max

            for id in del_tracks:
                # is_target1(box, frame)

                if id < len(self.tracks):
                    if len(self.tracks[id].trace) > 0:
                        box = self.tracks[id].trace[-1]
                        
                        x, y, w, h = box
                        # cv2.imwrite(f"img_{id}_{datetime.now()}.png", imcrop(frame, (x, y, x + w, y + h)))

                        # is_target, new_box = kmeans_is_target(frame, box)
                        is_target, new_box = self.is_target_cnn(frame, box)

                        if is_target:
                            print("self feed")
                            self.tracks[id].trace.append(new_box)


                            last_box = self.tracks[id].trace[-1]
                            new_center = box2center(new_box)
                            last_center = box2center(last_box)
                            new_distance = centers_distance(new_center, last_center)

                            self.tracks[id].distances.append(new_distance)      

                            # x, y, w, h = box
                            # x1, y1, w1, h1 = new_box
                            # cv2.imwrite(f"img_{box}1.png", imcrop(frame, (x, y, x + w, y + h)))
                            # cv2.imwrite(f"img_{box}2.png", imcrop(frame, (x1, y1, x1 + w1, y1 + h1)))

                            continue 

                    del self.tracks[id]
                    del assignment[id]
                else:
                    print("ERROR: id is greater than length of tracks")


        # Now look for un_assigned detects
        un_assigned_detects = [i for i in range(len(bboxes)) if i not in assignment]

        # Start new tracks
        if len(un_assigned_detects) != 0:
            for i in range(len(un_assigned_detects)):
                track = Track(bboxes[un_assigned_detects[i]],
                              self.trackIdCount)
                self.trackIdCount += 1
                self.tracks.append(track)


        for i in range(len(assignment)):

            if assignment[i] != -1:
                self.tracks[i].skipped_frames = 0
                new_box = bboxes[assignment[i]]
                last_box = self.tracks[i].trace[-1]
                
                new_center = box2center(new_box)
                last_center = box2center(last_box)

                new_distance = centers_distance(new_center, last_center)

                m = (np.average(self.tracks[i].distances))
                std = (np.std(self.tracks[i].distances))

                # print(f"new distance = {new_distance}")
                # print(f"mean = {m}")
                # print(f"std = {std}")

                if True or len(self.tracks[i].trace) < self.max_trace_length or new_distance < m + 4*std:
                    self.tracks[i].distances.append(new_distance)                
                    self.tracks[i].trace.append(bboxes[assignment[i]])


            if len(self.tracks[i].trace) > self.max_trace_length:
                for j in range(len(self.tracks[i].trace) -
                               self.max_trace_length):
                    del self.tracks[i].trace[j]

