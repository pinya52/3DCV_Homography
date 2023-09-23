import sys
import numpy as np
import cv2 as cv
import math
from tqdm import tqdm
from function import *

def main(img1, img2, gt_correspondences, good_rate, img2_name):
    # print('Getting SIFT correspondences of img1 and img2 \n')
    points1, points2 = get_sift_correspondences(img1, img2, k, good_rate, img2_name)

    # DLT
    # print('Start DLT')
    # print('Estimating homography')
    H = homography_estimation(points1, points2)
    M, mask = cv.findHomography(points1, points2)


    # print('Reprojecting')
    pt_hat = reprojection(gt_correspondences[0], H)
    pt_hat_cv = reprojection(gt_correspondences[0], M)
    # print('Calculating Error')
    error = calculate_error(pt_hat, gt_correspondences[1])
    error_cv = calculate_error(pt_hat_cv, gt_correspondences[1])


    # Normalized DLT
    # print('Start Normalized DLT')
    # print('Normalizing')
    T, trans_p1 = normalized(points1)
    T_prime, trans_p2 = normalized(points2)

    # print('Estimating homography')
    H_hat = homography_estimation(trans_p1, trans_p2)
    H_norm = np.dot(np.linalg.inv(T_prime), np.dot(H_hat, T))
    M, mask = cv.findHomography(points1, points2)

    # print('Reprojecting')
    pt_hat_norm = reprojection(gt_correspondences[0], H_norm)
    # print('Calculating Error \n')
    error_norm = calculate_error(pt_hat_norm, gt_correspondences[1])

    pt_hat_cv = reprojection(gt_correspondences[0], M)
    error_cv = calculate_error(pt_hat_cv, gt_correspondences[1])

    print('----------Result----------')
    print('Mine H : \n', H, '\n')
    print('OpenCV H : \n', M, '\n')
    print('Mine H after Normliazed DLT: \n', H_norm, '\n')
    # print('Mine Error : ', error, '\n')
    # print('OpenCV Error : ', error_cv, '\n')
    # print('Mine Error after Normliazed DLT: ', error_norm)
    return [error, error_cv, error_norm]

if __name__ == '__main__':
    # read
    # print('Reading img1 and img2')
    img1 = cv.imread('images/1-0.png')
    img2_name = sys.argv[1]
    img2 = cv.imread('images/1-%s.png'%(img2_name))
    # print('Reading groundtruth')
    gt_correspondences = np.load('groundtruth_correspondences/correspondence_0%s.npy'%(img2_name))
    good_rate = 0.75

    ks = [4, 8, 20]

    errors = []
    for k in ks:
        print("Now running k : %d \n"%(k))
        error_k = main(img1, img2, gt_correspondences, good_rate, img2_name)
        errors.append(error_k)
        print("\n========================================================== \n")
    for i, k in enumerate(ks):
        print("error for k : %d \n"%(k))
        print('Mine Error : ', errors[i][0], '\n')
        print('OpenCV Error : ', errors[i][1], '\n')
        print('Mine Error after Normliazed DLT: ', errors[i][2], '\n\n')
