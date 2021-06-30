import cv2
import numpy as np
import matplotlib.pyplot as plt
import serial
import time

# 4 & 5-> 14x14 data 불안
plt.style.use('dark_background')

img_ori = cv2.imread('8.png')

height, width, channel = img_ori.shape

gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)

# Maximized Contrast

# 3x3 직사각형
structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

# tophat = ori - opening && black hat = ori - closing
# opening : erode -> dilate (center)
imgTopHat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, structuringElement)
imgBlackHat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, structuringElement)

imgGrayscalePlusTopHat = cv2.add(gray, imgTopHat)
gray = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)

# plt.figure(figsize=(12, 10))
# plt.imshow(gray, cmap='gray')
# plt.show()

# Adaptive Thresholding
img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)

img_thresh = cv2.adaptiveThreshold(
    img_blurred,
    maxValue=255.0,
    adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    thresholdType=cv2.THRESH_BINARY_INV,
    blockSize=19,
    C=9
)

# plt.figure(figsize=(12, 10))
# plt.imshow(img_thresh, cmap='gray')
# plt.show()

# find contour - contours line 그릴 수 있는 point만 저장
contours, _ = cv2.findContours(
    img_thresh,
    mode=cv2.RETR_LIST,
    method=cv2.CHAIN_APPROX_SIMPLE
)

temp_result = np.zeros((height, width, channel), dtype=np.uint8)

cv2.drawContours(temp_result, contours=contours, contourIdx=-1, color=(255, 255, 255))

# plt.figure(figsize=(12, 10))
# plt.imshow(temp_result)
# plt.show()

# prepare data - 여기까진 사진 내부의 모든 contour에 대해 사각형 만듦
temp_result = np.zeros((height, width, channel), dtype=np.uint8)

contours_dict = []

# boundingRect -> 외접하는 똑바로 세워진 사각형의 좌표 return
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)
    cv2.rectangle(temp_result, pt1=(x, y), pt2=(x + w, y + h), color=(255, 255, 255), thickness=2)

    # insert to dict
    contours_dict.append({
        'contour': contour,
        'x': x,
        'y': y,
        'w': w,
        'h': h,
        'cx': x + (w / 2),
        'cy': y + (h / 2)
    })

# plt.figure(figsize=(12, 10))
# plt.imshow(temp_result, cmap='gray')
# plt.show()

# 후보군 추출
MIN_AREA = 200
MIN_WIDTH, MIN_HEIGHT = 2, 8
MIN_RATIO, MAX_RATIO = 0.25, 1.0

possible_contours = []

cnt = 0
for d in contours_dict:
    area = d['w'] * d['h']
    ratio = d['w'] / d['h']

    if area > MIN_AREA \
            and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT \
            and MIN_RATIO < ratio < MAX_RATIO:
        d['idx'] = cnt
        cnt += 1
        possible_contours.append(d)

# visualize possible contours
temp_result = np.zeros((height, width, channel), dtype=np.uint8)

for d in possible_contours:
    cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x'] + d['w'], d['y'] + d['h']), color=(255, 255, 255),
                  thickness=2)







#--------------------이찬현 추가--------------------------#

from sklearn import model_selection, svm, metrics
import struct
from micromlgen import port

flag = False

def load_csv(fname):
    labels = []
    images = []

    with open(fname, "r", encoding="utf-8") as f:
        for line in f:
            cols = line.split(",")
            if len(cols) < 2: continue
            labels.append(int(cols.pop(0)))

            global flag
            if flag==True:
                vals = list(map(lambda n: int(n), cols))
                vals = np.array(vals)
                vals = np.reshape(vals, (14, 14))
                plt.imshow(vals, cmap='gray')
                plt.show()
                flag = False

            vals = list(map(lambda n: int(n) / 256, cols))
            images.append(vals)
        return {"labels": labels, "images": images}

data = load_csv("./train.csv")
test = load_csv("./t10k.csv")

clf = svm.SVC(kernel = 'linear', gamma = 0.001)
clf.fit(data["images"], data["labels"])
#--------------------------------------------------------#












# 글자 크기로 추출
MAX_DIAG_MULTIPLYER = 5  # 5
MAX_ANGLE_DIFF = 15.0  # 12.0
MAX_AREA_DIFF = 0.5  # 0.5
MAX_WIDTH_DIFF = 0.5
MAX_HEIGHT_DIFF = 0.5
MIN_N_MATCHED = 3  # 3

def find_chars(contour_list):
    matched_result_idx = []

    for d1 in contour_list:
        matched_contours_idx = []
        for d2 in contour_list:
            if d1['idx'] == d2['idx']:
                continue

            dx = abs(d1['cx'] - d2['cx'])
            dy = abs(d1['cy'] - d2['cy'])

            diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)

            distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))
            if dx == 0:
                angle_diff = 90
            else:
                angle_diff = np.degrees(np.arctan(dy / dx))
            area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
            width_diff = abs(d1['w'] - d2['w']) / d1['w']
            height_diff = abs(d1['h'] - d2['h']) / d1['h']

            if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \
                    and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
                    and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                matched_contours_idx.append(d2['idx'])

        # append this contour
        matched_contours_idx.append(d1['idx'])

        if len(matched_contours_idx) < MIN_N_MATCHED:
            continue

        matched_result_idx.append(matched_contours_idx)

        unmatched_contour_idx = []
        for d4 in contour_list:
            if d4['idx'] not in matched_contours_idx:
                unmatched_contour_idx.append(d4['idx'])

        unmatched_contour = np.take(possible_contours, unmatched_contour_idx)

        # recursive
        recursive_contour_list = find_chars(unmatched_contour)

        for idx in recursive_contour_list:
            matched_result_idx.append(idx)

        break

    return matched_result_idx


result_idx = find_chars(possible_contours)

matched_result = []
for idx_list in result_idx:
    matched_result.append(np.take(possible_contours, idx_list))

# visualize possible contours
temp_result = np.zeros((height, width, channel), dtype=np.uint8)

for r in matched_result:
    for d in r:
        #         cv2.drawContours(temp_result, d['contour'], -1, (255, 255, 255))
        cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x'] + d['w'], d['y'] + d['h']), color=(255, 255, 255),
                      thickness=2)

# plt.figure(figsize=(12, 10))
# plt.imshow(temp_result, cmap='gray')
# plt.show()

# 번호판 영역 추출
PLATE_WIDTH_PADDING = 1.4
PLATE_HEIGHT_PADDING = 1.5
MIN_PLATE_RATIO = 3
MAX_PLATE_RATIO = 10

plate_imgs = []
plate_infos = []

# sorted: cx 오름차순으로 정렬, plate: 큰사각형 중심x,y
for i, matched_chars in enumerate(matched_result):
    sorted_chars = sorted(matched_chars, key=lambda x: -x['cx'])
    # print(len(sorted_chars))
    for j in range(4):
        plate_cx = sorted_chars[j]['cx']
        plate_cy = sorted_chars[j]['cy']

        plate_width = sorted_chars[j]['w'] * PLATE_WIDTH_PADDING

        plate_height = sorted_chars[j]['h'] * PLATE_HEIGHT_PADDING

        img_cropped = cv2.getRectSubPix(
            img_thresh,
            patchSize=(int(plate_width), int(plate_height)),
            center=(int(plate_cx), int(plate_cy))
        )

        plate_imgs.append(img_cropped)
        plate_infos.append({
            'x': int(plate_cx - plate_width / 2),
            'y': int(plate_cy - plate_height / 2),
            'w': int(plate_width),
            'h': int(plate_height)
        })

        # plt.imshow(img_cropped, cmap='gray')
        # plt.show()
    break

    longest_idx, longest_text = -1, 0



    # morph 연산
plate_chars = []
predict = []

for i, plate_img in enumerate(plate_imgs):
    plate_img = cv2.resize(plate_img, dsize=(0, 0), fx=1.6, fy=1.6)
    _, plate_img = cv2.threshold(plate_img, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # find contours again (same as above)
    contours, _ = cv2.findContours(plate_img, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)

    plate_min_x, plate_min_y = plate_img.shape[1], plate_img.shape[0]
    plate_max_x, plate_max_y = 0, 0

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        area = w * h
        ratio = w / h

        if area > MIN_AREA \
                and w > MIN_WIDTH and h > MIN_HEIGHT \
                and MIN_RATIO < ratio < MAX_RATIO:
            if x < plate_min_x:
                plate_min_x = x
            if y < plate_min_y:
                plate_min_y = y
            if x + w > plate_max_x:
                plate_max_x = x + w
            if y + h > plate_max_y:
                plate_max_y = y + h

    img_result = plate_img[plate_min_y:plate_max_y, plate_min_x:plate_max_x]

    #img_result = cv2.GaussianBlur(img_result, ksize=(3, ), sigmaX=0)
    _, img_result = cv2.threshold(img_result, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img_result = cv2.copyMakeBorder(img_result, top=10, bottom=10, left=10, right=10, borderType=cv2.BORDER_CONSTANT,
                                    value=(0, 0, 0))
    





    #-------------------------------이찬현 추가------------------------------#
    # morph 연산
    kernel = np.ones((1, 1), np.uint8)
    img_result = cv2.morphologyEx(img_result, cv2.MORPH_CLOSE, kernel)

    # pad 연산
    img_result = np.pad(img_result, (3, 3), mode = 'constant', constant_values=0)


    img_result = cv2.resize(img_result, dsize=(28, 28), interpolation=cv2.INTER_AREA)

    img_result2 = img_result.reshape(784, )

    img_result2 = list(map(lambda n: int(n) / 256, img_result2))

    img_result3 = []

    img_result3.append(img_result2)


    predict.append(clf.predict(img_result3)[0])

    img_result3 = []


    plt.imshow(img_result, cmap='gray')
    plt.show()



print(predict)



val = ['0'+str(predict[-2])+str(predict[-1]), '008', '022', '027', '090']
print(val)

ser = serial.Serial("COM2", 9600, timeout = 1)
print("write 1234", end='\n')
for i in range(0,5):
    op = str(val[i])
    ser.write(op.encode())
    print(ser.readline())