import cv2
import numpy as np
from utils import *

class CheckerboardGenerator():
    def __init__(self, cb_w, cb_h, win_name = "screen"):
        self.win_name = win_name
        self.win_size = self.get_window_shape()
        self.roi_list = []
        self.roi = [] # visible area
        self.cbb = [] # checkerboard boundary list

        self.checkerboard = self.get_checkerboard(cb_w + 1, cb_h + 1) # generate checkerboard

        focal = 500 # vitual focal length
        self.K_in_inv = np.linalg.inv(np.array([[focal, 0, self.checkerboard.shape[0] * 0.5],[0, focal, self.checkerboard.shape[1] * 0.5], [0, 0, 1]]))
        self.K_out = np.array([[focal, 0, self.win_size[1] * 0.5],[0, focal, self.win_size[0] * 0.5], [0, 0, 1]])        

    def get_window_shape(self):
        win_name = self.win_name
        cv2.namedWindow(win_name, cv2.WND_PROP_FULLSCREEN)
        cv2.imshow(win_name, np.array([[0.0, 1.0]]).T)
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.waitKey(10)
        window_vertical = cv2.getWindowImageRect(win_name)
        cv2.imshow(win_name, np.array([[0.0, 1.0]]))
        cv2.setWindowProperty(win_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        cv2.waitKey(10)
        window_horizontal = cv2.getWindowImageRect(win_name)
        win_size = np.array([window_vertical[3], window_horizontal[2]])
        return win_size

    def get_visible_area(self, font = cv2.FONT_HERSHEY_PLAIN, font_scale = 2, thickness = 1):
        text_size = cv2.getTextSize(text = chr(65) + str(0), fontFace = font, fontScale = font_scale, thickness = thickness)
        text_offset = int(text_size[0][1] * 1.5)

        stride = 100
        img = np.ones(self.win_size, dtype = np.uint8) * 255
        for v in range(0, self.win_size[0], stride):
            v_ = int(v/stride)
            for u in range(0, self.win_size[1], stride):
                u_ = int(u/stride)
                if (v_ + u_) % 2:
                    img[v:v + stride, u:u + stride] = 0
                    cv2.putText(img, chr(65 + v_) + str(u_), (u, v + text_offset), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
                else:
                    cv2.putText(img, chr(65 + v_) + str(u_), (u, v + text_offset), font, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
        cv2.imshow(self.win_name, img)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        img_clear = img.copy()
        cv2.setMouseCallback(self.win_name, self.mouse_event, img)
        while True:
            key = cv2.waitKey()
            if key == ord('f'): # full screen 
                print("Full screen selected")
                print("Show calibration pattern")
                self.roi_list.append([0, 0])
                self.roi_list.append([self.win_size[1], 0])
                self.roi_list.append([self.win_size[1], self.win_size[0]])
                self.roi_list.append([0, self.win_size[0]])
                self.roi_list = np.array(self.roi_list)
                self.roi = self.roi_list
                return 
            if key == 27: # esc 
                print("Exit calibration")
                return 
            if key == 13: # enter key
                if len(self.roi_list) ==0:
                    print("No region selected")
                    print("Exit calibration")
                    return
                print("Show calibration pattern")
                self.roi_list = np.array(self.roi_list)
                self.roi = cv2.convexHull(self.roi_list)
                img = np.zeros(self.win_size)
                cv2.fillConvexPoly(img, self.roi, 255)
                cv2.imshow(self.win_name, img)
                cv2.waitKey(1)
                break
            else:
                img = img_clear.copy()
                self.roi_list = []
                cv2.imshow(self.win_name, img)
                cv2.setMouseCallback(self.win_name, self.mouse_event, img)
        cv2.waitKey()

    def get_checkerboard(self, cb_w, cb_h):
        square_size = np.round(np.min([self.win_size[0] / (cb_h + 1), self.win_size[1] / (cb_h + 1)])).astype(int)
        checkerboard = 255 * np.ones(np.array([cb_h + 1, cb_w + 1]) * square_size, dtype = np.uint8)
        offset = int(square_size / 2)
        colors = [0, 255] # Black and white
        for i in range(0, cb_w):
            for j in range(0, cb_h):
                x = i * square_size + offset
                y = j * square_size + offset
                color = colors[(i + j)%2]
                cv2.rectangle(checkerboard, (x, y), (x + square_size, y + square_size), color, -1)

        self.cbb.append([square_size, square_size])
        self.cbb.append([square_size * (cb_w) , square_size])
        self.cbb.append([square_size , square_size * (cb_h)])
        self.cbb.append([square_size * (cb_w) , square_size * (cb_h)])
        self.cbb = np.array(self.cbb)

        checkerboard = cv2.cvtColor(checkerboard, cv2.COLOR_GRAY2BGR)
        return checkerboard

    def mouse_event(self, event, x, y, flags, img):    
        if event == cv2.EVENT_LBUTTONDOWN:
            self.roi_list.append([x, y])
            color = (255, 0, 255)
            markerType = cv2.MARKER_CROSS
            markerSize = 15
            thickness = 2
            cv2.drawMarker(img, (x, y), color, markerType, markerSize, thickness)
            cv2.imshow(self.win_name, img)

    def homography(self, R, t):       
        H = R.copy()
        H[:3, 2] += t
        H = self.K_out @ H @ self.K_in_inv
        return H
    
    def show_checkerboard(self, z_range = None, r_range = None, repeat_num = 10):
        roi_img = np.zeros(self.win_size, dtype = np.uint8)
        cv2.fillConvexPoly(roi_img, self.roi, 1)

        if z_range is None:
            z_range = 10 * np.linspace(0.0, 1.0, 5)
        if r_range is None:
            r_range = np.zeros((3, 10))
            r_range[0, 0:5] = np.linspace(-1.0, 1.0, 5)
            r_range[1, 5:10] = np.linspace(-1.0, 1.0, 5)
            r_range = r_range.T

        # for z range
        for z in z_range:
            # for r range
            for r in r_range:
                R = SO3(r)
                H = self.homography(R, np.array([0, 0, z]))
                cbb_out = H @ np.block([[self.cbb.T], [np.ones((1, 4))]])
                cbb_out /= cbb_out[2, :]
                cbb_out = cbb_out.astype(int)

                # rotate and scale image
                warped_img = cv2.warpPerspective(self.checkerboard, H, (self.win_size[1], self.win_size[0]))
                # compute movable space
                roi_img_res = np.zeros(self.win_size, dtype = np.uint8)
                for cbb_iter in range(4):
                    (x, y) = cbb_out[:2, cbb_iter] - np.array([self.win_size[1]/2, self.win_size[0]/2]).astype(int)
                    roi_img_res += cv2.warpAffine(roi_img, np.float32([[1, 0, x], [0, 1, y]]), (self.win_size[1], self.win_size[0]), cv2.INTER_NEAREST, cv2.BORDER_ISOLATED)
                roi_img_res[int(self.win_size[0]/2), int(self.win_size[1]/2)] = 0
                # compute intersection area
                valid = roi_img_res >= 4
                # compute shift parameter in visible area
                valid_pts = np.array(np.where(valid))
                if valid_pts.shape[1] == 0:
                    continue
                valid_pts[[0, 1]] = valid_pts[[1, 0]]

                # randomly sampling shift parameter
                uv_shift_sample = np.random.permutation(valid_pts.T)[:repeat_num]
                (x, y) = valid_pts[:, -1].T
                for (x, y) in uv_shift_sample:
                    (x_shift, y_shift) = np.array([self.win_size[1]/2, self.win_size[0]/2]).astype(int) - (x, y)
                    warped_img2 = cv2.warpAffine(warped_img, np.float32([[1, 0, x_shift], [0, 1, y_shift]]), (self.win_size[1], self.win_size[0]), cv2.INTER_NEAREST, cv2.BORDER_ISOLATED)

                    if SHOW_ROI:
                        img = np.zeros(self.win_size, dtype = np.uint8)
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                        cv2.fillConvexPoly(img, self.roi, (64, 64, 64))
                        cv2.imshow(self.win_name, warped_img2 + ((1 - warped_img2/255) * img).astype(np.uint8))
                    else:
                        cv2.imshow(self.win_name, warped_img2)
                    key = cv2.waitKey()

                    if key == 27: # esc 
                        print("Exit calibration")
                        return 

SHOW_ROI = True

print("---------Key instruction---------\n"\
      "f: use full screen\n"\
      "enter: select roi with selected region\n"\
      "esc: exit calibration\n"\
      "------------------------------------")

cbg = CheckerboardGenerator(8, 8)
cbg.get_visible_area()
cbg.show_checkerboard(z_range = np.array([0, 0.2, 0.4, 0.8, 1.5, 2.0]), repeat_num = 5)

cv2.waitKey(0)