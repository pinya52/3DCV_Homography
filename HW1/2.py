import sys
import numpy as np
import cv2 as cv
import math
from tqdm import tqdm
from copy import copy
from function import homography_estimation, reprojection
from  mouse_click_example import main
if __name__ == '__main__':
    # read
    img_bef = cv.imread(sys.argv[1])
    # cv.imshow('Original Image', img_bef)
    # cv.waitKey(0)

    # corners = [[169, 467],
    #             [673, 308],
    #             [427, 1206],
    #             [1038, 935]]

    corners = main(img_bef)
    print(corners)

    img_y,img_x = img_bef.shape[:2]

    points_bef = np.float32(corners)
    points_after = np.float32([[0,0], [img_x-1,0], [0,img_y-1], [img_x-1, img_y-1]])

    H_warp = homography_estimation(points_bef, points_after)
    H_warp_inv = np.linalg.inv(H_warp)

    Y,X = np.mgrid[0:img_y, 0:img_x]
    yx = np.vstack((X.flatten(), Y.flatten())).T
    yx

    pixel_trans = reprojection(yx, H_warp_inv)

    img_warp = copy(img_bef)
    for i, [y,x] in enumerate(tqdm(pixel_trans)):
        x1, x2, y1, y2 = math.floor(x), math.ceil(x), math.floor(y), math.ceil(y)
        temp = []
        for j in range(3):
            res = ((x2-x)*(y2-y)*img_bef[x1][y1][j]+(x2-x)*(y-y1)*img_bef[x1][y2][j]+(x-x1)*(y2-y)*img_bef[x2][y1][j]+(x-x1)*(y-y1)*img_bef[x2][y2][j])/((x2-x1)*(y2-y1))
            temp.append(int(np.round(res)))

        img_warp[i//img_x][i%img_x] = temp

    cv.imshow('Warpped Image', img_warp)
    cv.waitKey(0)