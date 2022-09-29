import numpy as np
import cv2 as cv
import math
from tqdm import tqdm

def get_sift_correspondences(img1, img2, k, good_rate):
    '''
    Input:
        img1: numpy array of the first image
        img2: numpy array of the second image
    Return:
        points1: numpy array [N, 2], N is the number of correspondences
        points2: numpy array [N, 2], N is the number of correspondences
    '''
    #sift = cv.xfeatures2d.SIFT_create()# opencv-python and opencv-contrib-python version == 3.4.2.16 or enable nonfree
    sift = cv.SIFT_create()             # opencv-python==4.5.1.48
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    
    matcher = cv.BFMatcher()
    matches = matcher.knnMatch(des1, des2, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < good_rate * n.distance:
            good_matches.append(m)

    good_matches = sorted(good_matches, key=lambda x: x.distance)
    # print(good_matches[0].distance, good_matches[1].distance)
    good_matches_topk = good_matches[:k]
    points1 = np.array([kp1[m.queryIdx].pt for m in good_matches_topk])
    points2 = np.array([kp2[m.trainIdx].pt for m in good_matches_topk])
    
    img_draw_match = cv.drawMatches(img1, kp1, img2, kp2, good_matches_topk, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv.imshow('match', img_draw_match)
    cv.waitKey(0)
    return points1, points2
        
def normalized(points):
    mean = np.mean(points, 0)
    
    total_dist = 0
    dist_list = []
    for point in points:
        dist = (point - mean[:2])**2
        dist = np.sum(dist, axis=0)
        dist = np.sqrt(dist)
        dist_list.append(dist)
    total_dist += dist
    mean_dist = total_dist/points.shape[0]

    factor = np.sqrt(2)/mean_dist

    T = np.array([[factor, 0, -factor*mean[0]],
                [0, factor, -factor*mean[1]],
                [0, 0, 1]])
    
    points = np.insert(points, 2, 1, axis=1)
    mid_points = np.dot(T, points.T).T

    temp =[]
    for point in mid_points:
        temp.append([point[0]/point[2], point[1]/point[2]])
    trans_points = np.array(temp)
    
    return T, trans_points


def homography_estimation(points1, points2):
    '''
    Input:
        points1: numpy array [N, 2], N is the number of correspondences
        points2: numpy array [N, 2], N is the number of correspondences
    Return:
        H: the homography between anchor image and target images
    '''
    temp = []
    for i in range(points1.shape[0]):
        temp.append([points1[i][0], points1[i][1], 1, 0, 0, 0, -(points2[i][0]*points1[i][0]), -(points2[i][0]*points1[i][1]), -(points2[i][0])])
        temp.append([0, 0, 0, -(points1[i][0]), -(points1[i][1]), -1, points2[i][1]*points1[i][0], points2[i][1]*points1[i][1], points2[i][1]])

    A = np.array(temp)
    # print('A : \n', A, '\n')
    u, s, vh = np.linalg.svd(A)
    # print('u : \n', u, '\n')
    # print('s : \n', s, '\n')
    # print('vh : \n', vh, '\n')
    h = vh[-1] / vh[-1,-1]
    # h = vh[-1]
    # print('h : \n', h, '\n')

    temp = []
    for i in range(3):
        temp.append([h[i*3], h[i*3+1], h[i*3+2]])
    H = np.array(temp)
    return H

def reprojection(groundtruth, H):
    '''
    Input:
        groundtruth: groundtruth
        H: the homography between anchor image and target images
    Return:
        pt_hat: groundtruth after reprojection
    '''
    ps = np.insert(groundtruth, 2, 1, axis=1)
    pt_hat = np.dot(H, ps.T)
    pt_hat = np.divide(pt_hat, pt_hat[-1]).T
    pt_hat = np.delete(pt_hat, 2, axis=1)

    return pt_hat

def calculate_error(pt_hat, pt):
    '''
    Input:
        pt_hat: groundtruth after reprojection
        pt: groundtruth
    Return:
        error: error which need to divide by n (mean is what we want)
    '''
    error = 0
    n = pt_hat.shape[0]
    for i in range(n):
        dist = (pt_hat[i] - pt[i])**2
        dist = np.sum(dist, axis=0)
        error += np.sqrt(dist)
    return error/n
