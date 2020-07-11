"""
Program:   Cardiacvascural borderline detection.
Module:    cb
Language:  python3
Date:      2019-12-19 10:00:00
Version:   VER 1.0
Developer:  LEE GAEUN  (ggelee93@gmail.com)

Copyright (c) medical imaging and intelligent reality lab (MI2RL) in Asan Medical Center (AMC).
All rights reserved.
"""

import numpy as np
import cv2
import os
import colorsys
import random
import SimpleITK as sitk


class Calc(object):

    def Get_DSC(self, pred_m, gt_m):
        """Returns Dice Similarity Coefficient for ground truth and predicted masks."""

        if pred_m is None:
            """Absence of prediction :class"""
            return 0
        if not np.any(pred_m):
            """Empty mask"""
            return 0

        intersection = np.logical_and(gt_m, pred_m)

        true_sum = gt_m[:, :].sum()
        pred_sum = pred_m[:, :].sum()
        intersection_sum = intersection[:, :].sum()

        dsc = (2 * intersection_sum + 1.) / (true_sum + pred_sum + 1.)

        return dsc

    def Get_BBox(self, mask_binary, margin=0):
        """Returns bbox: [x1,y1,x2,y2]
        y2 and x2 do not include the object."""

        contours, _ = cv2.findContours(mask_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not (len(contours) > 0):
            return
        rects = []
        for cnt in contours:
            rects.append(cv2.boundingRect(cnt))

        top_x, top_y, bottom_x, bottom_y = 0, 0, 0, 0
        rects.sort()

        top_x = min([x for (x, y, w, h) in rects]) - margin
        top_y = min([y for (x, y, w, h) in rects]) - margin
        bottom_x = max([x + w for (x, y, w, h) in rects]) + margin
        bottom_y = max([y + h for (x, y, w, h) in rects]) + margin

        mask_size = mask_binary.shape
        if (top_x < 0): top_x = 0
        if (top_y < 0): top_y = 0
        if (bottom_x >= mask_size[1]): bottom_x = mask_size[1] - 1
        if (bottom_y >= mask_size[0]): bottom_y = mask_size[0] - 1

        return (top_x, top_y, bottom_x, bottom_y)

    def Get_MiddlePoint(self, LineMask_binary, y=None):
        if y is None:
            bbox_line = self.Get_BBox(LineMask_binary)
            if bbox_line is None:
                return
            y = bbox_line[1] + (bbox_line[3] - bbox_line[1]) / 2
            y = int(y)

        LineMask_binary = np.asarray(LineMask_binary)
        x_candidates1 = []
        y_start = y;
        while np.asarray(x_candidates1).size == 0:
            x_candidates1 = np.where(LineMask_binary[y_start, :] > 0)
            y_start = y_start - 1
            if (y_start < 0):
                break;
        x_candidates2 = []
        y_start = y
        while np.asarray(x_candidates2).size == 0:
            x_candidates2 = np.where(LineMask_binary[y_start, :] > 0)
            y_start = y_start + 1
            if (y_start == LineMask_binary.shape[0]):
                break;

        x_candidates = np.append(x_candidates1, x_candidates2)

        if np.asarray(x_candidates).size > 0:
            x = np.min(x_candidates) + (np.max(x_candidates) - np.min(x_candidates)) / 2
            return (x, y)
        else:
            return

    def Get_Axis_withMask(self, lineMask_binary):

        bbox_Line = self.Get_BBox(lineMask_binary)
        w = bbox_Line[2] - bbox_Line[0]
        h = bbox_Line[3] - bbox_Line[1]

        axis, range1, range2 = 0, 0, 0
        if (w > h):
            """horizontal axis"""
            axis = bbox_Line[1] + h / 2
            range1 = bbox_Line[0]
            range2 = bbox_Line[2]
        else:
            """vertical axis"""
            axis = bbox_Line[0] + w / 2
            range1 = bbox_Line[1]
            range2 = bbox_Line[3]

        return axis, np.array([range1, range2])

    def Get_Distance_fromMidline(self, lineMask_binary, midline_x, spacing=None):
        midpoint_line = self.Get_MiddlePoint(lineMask_binary)
        if midpoint_line is None:
            return
        dist = abs(midpoint_line[0] - midline_x)

        if spacing:
            dist *= spacing[0]
        return dist

    def Get_Cardiac_Area(self, carina_binary, midline_x,
                         rtLowerCB_binary, rtUpperCB_binary,
                         aorticKnob_binary, pulmonaryConus_binary, laa_binary, ltLowerCB_binary,
                         spacing=None):

        """Top boundary"""
        bbox = self.Get_BBox(carina_binary)
        if bbox is None:
            return
        top_y = bbox[1]
        mask_size = np.array(carina_binary).shape

        """Left boundary"""
        mask_left = np.zeros(mask_size, dtype="uint8")
        for mask in ([rtLowerCB_binary, rtUpperCB_binary]):
            if mask is not None:
                mask_left = np.bitwise_or(mask_left, mask);

        """Right boundary"""
        mask_right = np.zeros(mask_size, dtype="uint8")
        for mask in ([aorticKnob_binary, pulmonaryConus_binary, laa_binary, ltLowerCB_binary]):
            if mask is not None:
                mask_right = np.bitwise_or(mask_right, mask)

        """Bottom boundary & Count pixels"""
        count_pixels = 0
        # Left ~ Mid
        bbox = self.Get_BBox(mask_left)
        if bbox is None:
            return
        bottom_y1 = bbox[3]
        for y in range(top_y, bottom_y1 + 1):
            left_point = self.Get_MiddlePoint(mask_left, y)
            if left_point is None:
                return
            count_pixels += (midline_x - left_point[0])
        # Right ~ Mid
        bbox = self.Get_BBox(mask_right)
        if bbox is None:
            return
        bottom_y2 = bbox[3]
        for y in range(top_y, bottom_y2 + 1):
            right_point = self.Get_MiddlePoint(mask_right, y)
            if right_point is None:
                return
            count_pixels += (right_point[0] - midline_x)
        count_pixels += (np.max([bottom_y1, bottom_y2]) - top_y)

        if spacing:
            return count_pixels * spacing[0] * spacing[1]

        return count_pixels

    def Get_Carina_Angle(self, carina_binary, spacing=None):
        """spacing: [x,y]"""

        p_candidates = np.where(carina_binary > 0)
        if len(p_candidates) == 0:
            return

        # Point1 (top): y가 가장 작은 Point들 중 x가 중간인 Point
        p_top_y = p_candidates[0].min()
        p_top_x_candidates = np.where(carina_binary[p_top_y, :] > 0)[0]
        p_top_x = p_top_x_candidates[int((len(p_top_x_candidates) - 1) / 2)]
        p_top = np.array([p_top_x, p_top_y])

        # Point2 (left): x가 가장 작은 Point들 중 y가 중간인 Point
        p_left_x = p_candidates[1].min()
        p_left_y_candidates = np.where(carina_binary[:, p_left_x] > 0)[0]
        p_left_y = p_left_y_candidates[int((len(p_left_y_candidates) - 1) / 2)]
        p_left = np.array([p_left_x, p_left_y])

        # Point3 (left): x가 가장 큰 Point들 중 y가 중간인 Point
        p_right_x = p_candidates[1].max()
        p_right_y_candidates = np.where(carina_binary[:, p_right_x] > 0)[0]
        p_right_y = p_right_y_candidates[int((len(p_right_y_candidates) - 1) / 2)]
        p_right = np.array([p_right_x, p_right_y])

        if spacing:
            p_top = np.multiply(p_top, spacing)
            p_left = np.multiply(p_left, spacing)
            p_right = np.multiply(p_right, spacing)

        v_top2left = p_left - p_top
        v_top2right = p_right - p_top
        cosine_angle = np.dot(v_top2left, v_top2right) / (np.linalg.norm(v_top2left) * np.linalg.norm(v_top2right))
        angle = np.arccos(cosine_angle)

        return np.degrees(angle)




