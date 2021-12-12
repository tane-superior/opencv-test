import time

import cv2 as cv
import numpy as np
import glob

niggers = glob.glob('*.jpeg')

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

pseudo_objp = np.zeros((7*7, 3), np.float32)
pseudo_objp[:,:2] = np.mgrid[0:7,0:7].T.reshape(-1,2)

object_points = []
image_points = []

for nig in niggers:
    img = cv.imread(nig)
    boomer_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ok, prototype_corners = cv.findChessboardCorners(boomer_img, (7, 7), None)

    if ok:
        object_points.append(pseudo_objp)

        corners = cv.cornerSubPix(boomer_img, prototype_corners, (11, 11), (-1, -1), criteria)
        image_points.append(corners)

        cv.drawChessboardCorners(img, (7, 7), corners, ok)
        cv.imshow("prodam garazh", img)
        cv.waitKey(500)

print("this part of the code did not crash\n")

ok, camera_matrix, distortion, rot_vecs, tran_vecs = cv.calibrateCamera(object_points, image_points, boomer_img.shape[::-1], None, None)

print("the camera coefficients matrix:\n", camera_matrix, '\n')
print("the distortion\n", distortion, '\n')

for nig in niggers:
    img = cv.imread(nig)
    height, width = img.shape[:2]
    upd_camera_matrix, rect = cv.getOptimalNewCameraMatrix(camera_matrix, distortion, (width, height), 1, (width, height))

    undistorted = cv.undistort(img, camera_matrix, distortion, None, upd_camera_matrix)
    x, y, w, h = rect
    undistorted = undistorted[y:y+h, x:x+w]
    cv.imwrite(nig.split('.')[0] + "und.png", undistorted)

print("this part did not crash neither\n")

mean_error = 0
for i in range(len(object_points)):
    pseudo_imgp, whatever = cv.projectPoints(object_points[i], rot_vecs[i], tran_vecs[i], camera_matrix, distortion)
    error = cv.norm(image_points[i], pseudo_imgp, cv.NORM_L2) / len(pseudo_imgp)
    mean_error += error

print("the reprojection error(the closer to zero the better):\n", mean_error)


