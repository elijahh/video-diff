#!/usr/bin/env python

# Compute asymmetric dissimilarity between two videos

from scipy.fftpack import dct
from scipy.signal import correlate

import cv2
import math
import matplotlib.pyplot as plt
import numpy as np
import sys
import time

DISTANCE_TAU = 16 # threshold for binarizing distance matrix

BLOCK_SIZE = 64 # size of block to partition and compare frames


def get_block(V, block):
    if block == []:
        for i in range(64):
            flag, frame = V.read()
            if flag == False:
                return []
            frame = cv2.cvtColor(frame, cv2.cv.CV_BGR2YCrCb)
            frame = cv2.resize(frame, (32, 32))
            block += [np.array(frame, dtype='d')[:,:,0]]
    else:
        flag, frame = V.read()
        if flag == False:
            return []
        frame = cv2.cvtColor(frame, cv2.cv.CV_BGR2YCrCb)
        frame = cv2.resize(frame, (32, 32))
        block = block[1:] + [np.array(frame, dtype='d')[:,:,0]]
    return block


def dct_hash(block):
    transform = dct(dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho'), axis=2, norm='ortho')
    coeffs = [transform[u][v][w] for u in range(1,5) for v in range(1,5) for w in range(1,5)]
    median = np.median(coeffs)
    return [1 if c > median else 0 for c in coeffs]


def hashes_of_blocks(V):
    hashes = []
    block = []
    while True:
        block = get_block(V, block)
        if block == []:
            break
        hashes += [dct_hash(block)]
    return hashes


def distance(blockA, blockB):
    dist = 0
    for i in range(64):
        dist += blockA[i] ^ blockB[i]
    return dist


def align(Va, Vb):
    starttime = time.time()
    # split into blocks of 64 frames and get 3D DCT hash of each block
    blocks_Va = hashes_of_blocks(Va)
    blocks_Vb = hashes_of_blocks(Vb)
    # compare pair-wise Hamming distance between each block
    x = len(blocks_Va)
    y = len(blocks_Vb)
    distances = np.zeros((x, y))
    distances_binarized = np.zeros((x, y))
    for i in range(x):
        for j in range(y):
            distances[i][j] = distance(blocks_Va[i], blocks_Vb[j])
            # binarize using threshold
            if distances[i][j] <= DISTANCE_TAU:
                distances_binarized[i][j] = 1
            else:
                distances_binarized[i][j] = 0
    # apply morphological opening
    distances_binarized = cv2.erode(distances_binarized, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    distances_binarized = cv2.dilate(distances_binarized, cv2.getStructuringElement(cv2.MORPH_RECT, (30, 30)))
    print "Distance matrix computed in ", (time.time() - starttime), " seconds"
#    imgplot = plt.imshow(distances / 64.0)
#    imgplot.set_cmap('spectral')
#    plt.colorbar()
#    plt.show()
#    imgplot = plt.imshow(distances_binarized)
#    plt.show()

    # label connected components
    starttime = time.time()
    current_label = 1
    pixel_queue = []
    distances_labeled = np.zeros((x, y))
    for i in range(x):
        for j in range(y):
            if distances_labeled[i][j] != 0:
                continue
            if distances_binarized[i][j] == 1:
                distances_labeled[i][j] = current_label
                pixel_queue.append((i, j))
                while len(pixel_queue) > 0:
                    (k, l) = pixel_queue.pop()
                    for m in range(k-1, k+2):
                        for n in range(l-1, l+2):
                            if m >= 0 and m < x and n >= 0 and n < y:
                                if distances_labeled[m][n] == 0 and distances_binarized[m][n] == 1:
                                    distances_labeled[m][n] = current_label
                                    pixel_queue.append((m, n))
                current_label += 1
    print "Connected components labeled in ", (time.time() - starttime), " seconds"

    # find distance minima for rows and columns of components
    starttime = time.time()
    minima = []
    boundaries = []
    for label in range(1, current_label):
        minima_label = []
        boundaries_label = (x, 0, y, 0) # left, right, upper, lower
        for i in range(x):
            min_column = DISTANCE_TAU
            min_coords = (-1, -1)
            for j in range(y):
                if distances_labeled[i][j] == label:
                    boundaries_label = (min(i, boundaries_label[0]), max(i, boundaries_label[1]), min(j, boundaries_label[2]), max(j, boundaries_label[3]))
                    dist = distances[i][j]
                    if dist <= min_column:
                        min_column = dist
                        min_coords = (i, j)
            if min_coords != (-1, -1):
                minima_label += [(min_coords)]
        for j in range(y):
            min_row = DISTANCE_TAU
            min_coords = (-1, -1)
            for i in range(x):
                if distances_labeled[i][j] == label:
                    dist = distances[i][j]
                    if dist <= min_row:
                        min_row = dist
                        min_coords = (i, j)
            if min_coords != (-1, -1):
                minima_label += [(min_coords)]
        minima += [minima_label]
        boundaries += [boundaries_label]
    print "Component minima found in ", (time.time() - starttime), " seconds"

    # fit line through minima for each component
    matching_segments = []
    for label in range(current_label-1):
        if len(minima[label]) == 0:
            continue
        [vx, vy, px, py] = cv2.fitLine(np.array(minima[label]), cv2.cv.CV_DIST_L2, 0, 0.01, 0.01)
        vx, vy, px, py = vx[0], vy[0], px[0], py[0]
        m = vy/vx
        b = py-m*px
        upper_left = (int(m*boundaries[label][0]+b), int(boundaries[label][0]))
        if m != 0:
            upper = (int(boundaries[label][2]), int((boundaries[label][2]-b)/m))
            if upper[1] > boundaries[label][0]:
                upper_left = upper
        lower_right = (int(m*boundaries[label][1]+b), int(boundaries[label][1]))
        if m != 0:
            lower = (int(boundaries[label][3]), int((boundaries[label][3]-b)/m))
            if lower[1] < boundaries[label][1]:
                lower_right = lower
        cv2.line(distances, upper_left, lower_right, (255, 255, 255))
        matching_segments += [(upper_left, lower_right, m, b)]
        print "Line segment: ", upper_left, ", ", lower_right, " ", m, " ", b
#    imgplot = plt.imshow(distances / 64.0)
#    imgplot.set_cmap('spectral')
#    plt.colorbar()
#    plt.show()

    # fine-tune alignment by correlating signals of average luminance over frames
    starttime = time.time()
    frame_pairs = []
    for segment in matching_segments:
        frame_x = []
        frame_y = []
        avgluma_a = []
        avgluma_b = []
        previous_py = -1
        for px in range(segment[0][1], min(segment[1][1]+64, x)):
            py = int(segment[2]*px+segment[3])
            if py < 0 or py == previous_py:
                continue
            if py >= y:
                break
            previous_py = py
            print "(", px, ",", py, ")"
            Va.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, px)
            flag1, frame1 = Va.read()
            Vb.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, py)
            flag2, frame2 = Vb.read()
            if flag1 and flag2:
                frame_x += [px]
                frame_y += [py]
                frame1 = cv2.cvtColor(frame1, cv2.cv.CV_BGR2YCrCb)
                avgluma_a += [np.average(frame1[:,:,0])]
                frame2 = cv2.cvtColor(frame2, cv2.cv.CV_BGR2YCrCb)
                avgluma_b += [np.average(frame2[:,:,0])]
        s1 = []
        s2 = []
        for i in range(1, len(avgluma_a)):
            s1 += [avgluma_a[i] - avgluma_a[i-1]]
            s2 += [avgluma_b[i] - avgluma_b[i-1]]
        s1 = np.array(s1)
        s2 = np.array(s2)
        correl1 = correlate(s1, s2)
        correl2 = correlate(s2, s1)
        shift_a = np.argmax(correl1)
        shift_b = np.argmax(correl2)
        shift = (shift_b - shift_a) / 2
        if shift != 0:
            if shift > 0:
                frame_y = np.roll(frame_y, -shift)
            else:
                frame_x = np.roll(frame_x, -shift)
            frame_x = frame_x[:-shift]
            frame_y = frame_y[:-shift]
        frame_pairs += zip(frame_x, frame_y)
    print "Sequences aligned in ", (time.time() - starttime), " seconds"
    return frame_pairs


def transform(img1, img2, detector):
    # find the keypoints and descriptors
    kp1, des1 = detector.detectAndCompute(img1,None)
    kp2, des2 = detector.detectAndCompute(img2,None)

    # match using FLANN
    FLANN_INDEX_LSH = 6
    index_params = dict(algorithm=FLANN_INDEX_LSH, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params,search_params)
    matches = flann.knnMatch(des1,des2,k=2)
    matches = [s[0] for s in matches if len(s) == 2 and s[0].distance <= 0.7*s[1].distance]

    #print "#matches=", len(matches)

    # bounding boxes around keypoints
    rect1 = [(np.inf, np.inf), (0, 0)]
    rect2 = [(np.inf, np.inf), (0, 0)]
    for m in matches:
        point1 = kp1[m.queryIdx].pt
        rect1[0] = (min(point1[0], rect1[0][0]), min(point1[1], rect1[0][1]))
        rect1[1] = (max(point1[0], rect1[1][0]), max(point1[1], rect1[1][1]))
        point2 = kp2[m.trainIdx].pt
        rect2[0] = (min(point2[0], rect2[0][0]), min(point2[1], rect2[0][1]))
        rect2[1] = (max(point2[0], rect2[1][0]), max(point2[1], rect2[1][1]))

    return (rect1, rect2)


def dissimilarity(Va, Vb, frame_pairs):
    # initialize detector
    detector = cv2.ORB()

    # try to transform frames in Va to corresponding frames in Vb
    for x, y in frame_pairs:
        Va.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, x)
        flag1, frame1 = Va.read()
        Vb.set(cv2.cv.CV_CAP_PROP_POS_FRAMES, y)
        flag2, frame2 = Vb.read()
        if flag1 and flag2:
            # compare matching region
            (rect1, rect2) = transform(frame1, frame2, detector)
            frame1_cropped = frame1[rect1[0][1]:rect1[1][1], rect1[0][0]:rect1[1][0]]
            frame2_cropped = frame2[rect2[0][1]:rect2[1][1], rect2[0][0]:rect2[1][0]]
            frame1_cropped = cv2.cvtColor(frame1_cropped, cv2.cv.CV_BGR2YCrCb)
            frame2_cropped = cv2.cvtColor(frame2_cropped, cv2.cv.CV_BGR2YCrCb)
            # scale down to smaller frame for comparison
            if frame1_cropped.shape[0] < frame2_cropped.shape[0] or frame1_cropped.shape[1] < frame2_cropped.shape[1]:
                frame1_scaled = frame1_cropped
                frame2_scaled = cv2.resize(frame2_cropped, (frame1_cropped.shape[1], frame1_cropped.shape[0]))
            else:
                frame1_scaled = cv2.resize(frame1_cropped, (frame2_cropped.shape[1], frame2_cropped.shape[0]))
                frame2_scaled = frame2_cropped
            # compare average luminance values of matching blocks
            h, w = frame1_scaled.shape[:2]
            max_diff = 0
            diff_i, diff_j = 0, 0
            for i in range(h / BLOCK_SIZE):
                for j in range(w / BLOCK_SIZE):
                    diff = np.abs(np.average(frame2_scaled[BLOCK_SIZE*i:BLOCK_SIZE*(i+1),BLOCK_SIZE*j:BLOCK_SIZE*(j+1),0])-np.average(frame1_scaled[BLOCK_SIZE*i:BLOCK_SIZE*(i+1),BLOCK_SIZE*j:BLOCK_SIZE*(j+1),0]))
                    if diff > max_diff:
                        max_diff = diff
                        diff_i, diff_j = i, j
            print x, y, max_diff
            if max_diff > 10:
                cv2.rectangle(frame1_scaled, (BLOCK_SIZE*diff_i, BLOCK_SIZE*diff_j), (BLOCK_SIZE*(diff_i+1), BLOCK_SIZE*(diff_j+1)), (0, 0, 255))
                cv2.rectangle(frame2_scaled, (BLOCK_SIZE*diff_i, BLOCK_SIZE*diff_j), (BLOCK_SIZE*(diff_i+1), BLOCK_SIZE*(diff_j+1)), (0, 0, 255))
            h1, w1 = frame1_scaled.shape[:2]
            h2, w2 = frame2_scaled.shape[:2]
            view = np.zeros((max(h1, h2), w1 + w2, 3), np.uint8)
            view[:h1, :w1, 0] = frame1_scaled[:, :, 0]
            view[:h2, w1:, 0] = frame2_scaled[:, :, 0]
            view[:, :, 1] = view[:, :, 0]
            view[:, :, 2] = view[:, :, 0]
            cv2.imshow("Frames", view)
            cv2.waitKey(30)


if __name__ == "__main__":
    Va = cv2.VideoCapture(sys.argv[1])
    Vb = cv2.VideoCapture(sys.argv[2])
    frame_pairs = align(Va, Vb)
    dissimilarity(Va, Vb, frame_pairs)

