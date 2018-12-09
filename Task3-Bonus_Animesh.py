# -*- coding: utf-8 -*-
"""
Created on Tue Oct 7 11:42:45 2018
@author: Animesh
"""
import cv2 as cv
#import numpy as np
import os

#img = cv2.imread("pos_2.jpg", 0) #Read Image as Numpy Array
mytemplate1 = cv.imread('t1_animesh.png', 0)
mytemplate2 = cv.imread('t2_animesh.png', 0)
mytemplate3 = cv.imread('t3_animesh.png', 0)
laplacian_template1 = cv.Laplacian(mytemplate1, cv.CV_8U)
laplacian_template2 = cv.Laplacian(mytemplate2, cv.CV_8U)
laplacian_template3 = cv.Laplacian(mytemplate3, cv.CV_8U)
w1, h1 = laplacian_template1.shape[::-1]
w2, h2 = laplacian_template2.shape[::-1]
w3, h3 = laplacian_template3.shape[::-1]

techniques = ['cv.TM_CCOEFF', 'cv.TM_CCOEFF_NORMED', 'cv.TM_CCORR', 
           'cv.TM_CCORR_NORMED','cv.TM_SQDIFF', 'cv.TM_SQDIFF_NORMED']

#Stores all 6 techniques of comparison in a list

source_image1 = ['neg_1.jpg','neg_2.jpg','neg_3.jpg','neg_4.jpg','neg_5.jpg',
                 'neg_6.jpg','neg_8.jpg','neg_9.jpg','neg_10.jpg',
                 'neg_11.jpg','neg_12.jpg','t1_1.jpg','t1_2.jpg','t1_3.jpg',
                 't1_4.jpg','t1_5.jpg','t1_6.jpg']
source_image2 = ['neg_1.jpg','neg_2.jpg','neg_3.jpg','neg_4.jpg','neg_5.jpg',
                 'neg_6.jpg','neg_8.jpg','neg_9.jpg','neg_10.jpg',
                 'neg_11.jpg','neg_12.jpg','t2_1.jpg','t2_2.jpg','t2_3.jpg',
                 't2_4.jpg','t2_5.jpg','t2_6.jpg']
source_image3 = ['neg_1.jpg','neg_2.jpg','neg_3.jpg','neg_4.jpg','neg_5.jpg',
                 'neg_6.jpg','neg_8.jpg','neg_9.jpg','neg_10.jpg',
                 'neg_11.jpg','neg_12.jpg','t3_1.jpg','t3_2.jpg','t3_3.jpg',
                 't3_4.jpg','t3_5.jpg','t3_6.jpg']

for tech in techniques:
    if not os.path.exists(tech):
        os.makedirs('./outputs/template1/' + tech)
        os.makedirs('./outputs/template2/' + tech)
        os.makedirs('./outputs/template3/' + tech)
    
    for image in source_image1:
        img = cv.imread(image, 0)
        img_blur = cv.GaussianBlur(img, (3, 3), 0)
        img_blur_laplacian = cv.Laplacian(img_blur, cv.CV_8U)

        method = eval(tech)

        # Apply template Matching
        # res = cv.matchTemplate(img_blur_laplacian, laplacian_template, method)
        res = cv.matchTemplate(img, mytemplate1, method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc

        bottom_right = (top_left[0] + w1, top_left[1] + h1)
        cv.rectangle(img, top_left, bottom_right, 255, 2)
        cv.imwrite('./outputs/template1/' + tech + '/' + image, img)
        
    for image in source_image2:
        img = cv.imread(image, 0)
        img_blur = cv.GaussianBlur(img, (3, 3), 0)
        img_blur_laplacian = cv.Laplacian(img_blur, cv.CV_8U)

        method = eval(tech)

        # Apply template Matching
        # res = cv.matchTemplate(img_blur_laplacian, laplacian_template, method)
        res = cv.matchTemplate(img, mytemplate2, method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc

        bottom_right = (top_left[0] + w2, top_left[1] + h2)
        cv.rectangle(img, top_left, bottom_right, 255, 2)
        cv.imwrite('./outputs/template2/' + tech + '/' + image, img)
        
    for image in source_image3:
        img = cv.imread(image, 0)
        img_blur = cv.GaussianBlur(img, (3, 3), 0)
        img_blur_laplacian = cv.Laplacian(img_blur, cv.CV_8U)

        method = eval(tech)

        # Apply template Matching
        # res = cv.matchTemplate(img_blur_laplacian, laplacian_template, method)
        res = cv.matchTemplate(img, mytemplate3, method)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)
        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        if method in [cv.TM_SQDIFF, cv.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc

        bottom_right = (top_left[0] + w3, top_left[1] + h3)
        cv.rectangle(img, top_left, bottom_right, 255, 2)
        cv.imwrite('./outputs/template3/' + tech + '/' + image, img)
        


