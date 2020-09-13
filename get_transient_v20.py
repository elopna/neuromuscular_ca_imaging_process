import pandas as pd
import matplotlib.pyplot as plt
from imutils import contours
from skimage import measure
import numpy as np
import argparse
import imutils
import cv2 as cv2
from itertools import chain
from scipy.interpolate import interp1d
from scipy import ndimage
from skimage.morphology import watershed
from skimage.feature import peak_local_max
from get_keypoints import alignImages
import os

def get_mask(img):
#     dst = cv2.fastNlMeansDenoising(img,None,7,21,21)[:,:,0] #recommended parameters 
    dst = cv2.fastNlMeansDenoising(img,None,11,8,21)[:,:,0]
#     dst = img[:,:,0].copy()
    thresh = cv2.threshold(dst, 0, 255, cv2.THRESH_TOZERO+cv2.THRESH_OTSU)[1]
    kernel = np.ones((1,1),np.uint8)
    thresh = cv2.erode(thresh,kernel,iterations = 1)
    kernel = np.ones((3,3),np.uint8)
    thresh = cv2.dilate(thresh,kernel,iterations = 1)
#     plt.imshow(img[:,:,0])
#     plt.show()
#     plt.imshow(thresh)
#     plt.show()
    return thresh

def get_multi_mask_n_background(files, path):
    list_of_masks = []
    background = []
    varss = []
    means = []
    img = cv2.imread(f'{path}{files[0]}')
    res_mask = np.zeros(img[:,:,0].shape,np.uint8)
    for i in np.arange(0,int(0.15*len(files)),10):
        img = cv2.imread(f'{path}{files[i]}')
        thresh = get_mask(img)
        list_of_masks.append(thresh)
        res_mask = cv2.bitwise_or(res_mask,thresh)
        background.append(np.mean(np.sort(list(chain.from_iterable(img[:,:,0])))[:500]))
        means.append(np.mean(list(chain.from_iterable(img[:,:,0]))))
        varss.append(np.var(img))
    for i in np.arange(int(0.15*len(files)),len(files),100):
        img = cv2.imread(f'{path}{files[i]}')
        thresh = get_mask(img)
        list_of_masks.append(thresh)
        res_mask = cv2.bitwise_or(res_mask,thresh)
        background.append(np.mean(np.sort(list(chain.from_iterable(img[:,:,0])))[:500]))
        means.append(np.mean(list(chain.from_iterable(img[:,:,0]))))
        varss.append(np.var(img))
    plt.imshow(img[:,:,0])
    plt.show()
    
#     need to count elements (ROI) in res_mask and then calculate each element separately
    # compute the exact Euclidean distance from every binary
    # pixel to the nearest zero pixel, then find peaks in this
    # distance map
    D = ndimage.distance_transform_edt(res_mask)
    localMax = peak_local_max(thresh, indices=False, min_distance=0, labels=thresh)
    # perform a connected component analysis on the local peaks,
    # using 8-connectivity, then appy the Watershed algorithm
    markers = ndimage.label(localMax, structure=np.ones((3, 3)))[0]
    labels = watershed(-D, markers, mask=res_mask)
    print("{} unique ROI found:".format(len(np.unique(labels)) - 1))
    
    plt.imshow(res_mask)
    plt.show()
    background = np.round(np.median(background),2)
    variance = np.round(np.mean(varss),2)
    mean = np.round(np.mean(means),2)
    return res_mask, background, mean, variance, labels

def plot_transient(files, path, thresh, background, mltpl, count):
    means = []
    for file in files:
        img = cv2.imread(f'{path}{file}')
        img = img - background
        img_thr = cv2.bitwise_and(img,img,mask = thresh)
        mean_val = img_thr.mean()
        means.append(mean_val)
    f = np.mean(means[:1])
    dff = [(df/f-1)*100 for df in means]
    dff_raw = dff.copy()
    dff = list(pd.Series(dff).rolling(15, center=True).median().fillna(method='bfill').fillna(method='ffill').values)
    start_ampl = np.mean(dff[:10])
    dff_raw = dff_raw - start_ampl
    dff = dff - start_ampl
    x_labels = range(0, int(1000*mltpl), 100)
    plt.plot(range(0, int(1000*mltpl), int(2*mltpl)), dff_raw)
    plt.xticks(x_labels)
    plt.xlabel('Time, ms')
    plt.plot(range(0, int(1000*mltpl), int(2*mltpl)), dff, c='r')
    plt.xticks(x_labels)
    plt.xlabel('Time, ms')
    plt.show()
    max_val = np.max(dff)
    max_val_t = np.argmax(dff)
    ampl = np.mean(np.sort(dff)[-10:])
    t1_rise = np.searchsorted(dff[:max_val_t], ampl*0.2, side="left") - 1
    t2_rise = np.searchsorted(dff[:max_val_t], ampl*0.8, side="left") + 1

    # print(max_val_t, ampl)
    # print(t1_rise,t2_rise)

    try:
        f_rise = interp1d(dff[t1_rise:t2_rise],np.arange(t1_rise,t2_rise),'linear')
        t2_decay = np.searchsorted(-np.array(dff[max_val_t+10:]), -ampl/np.e, side="left") -1
        f_decay = interp1d(dff[max_val_t-1:len(dff)],np.arange(max_val_t-1,len(dff)),'linear')
        rise_time = f_rise(ampl*0.8) - f_rise(ampl*0.2)
        rise_time = rise_time * 2  * mltpl
        decay = f_decay(ampl/np.e) - f_decay(ampl)
        decay = decay * 2 * mltpl
        print('Amplitude =', ampl)
        print('Rise time =', rise_time)
        print('Decay =', decay)
    except:
        rise_time, decay = np.nan, np.nan
    return dff, ampl, rise_time, decay

def transient_analysis(path, fps=1000, mask=0, control_img=''):
    for_excel = pd.DataFrame(columns=['ROI number', 'Amplitude', 'Rise time', 'Decay'])
    files = os.listdir(path)
    count_files = len(files)
    mltpl = count_files/fps
    thresh, background, mean, variance, labels = get_multi_mask_n_background(files, path)
    if mask:
        h, width, height = alignImages(files[0], control_img)
        thresh = cv2.warpPerspective(mask, h, (width, height))
    # print(background, mean, variance)
    for lb in np.unique(labels)[1:]:
        plt.imshow(labels==lb)
        plt.title(f'ROI {lb}')
        plt.show()
        trsh_temp = cv2.bitwise_and(thresh,thresh,mask=np.int8(labels==lb))
        res, ampl, rise_time, decay = plot_transient(files, path, trsh_temp, background, mltpl, count_files)
        df_temp = pd.DataFrame([[lb, np.round(ampl,2), np.round(rise_time,2), np.round(decay,2)]],columns=['ROI number', 'Amplitude', 'Rise time', 'Decay'])
        for_excel = for_excel.append(df_temp)
    #     plt.imshow(labels==lb)
    #     plt.title(lb)
    #     plt.show()
    # res = plot_transient(files, path, thresh, background)
    # print(res[1:])
    name = path.split('/')[0]
    for_excel.to_excel(f'result_{name}.xls',index=False)
    print()
    print('Result table:')
    print(for_excel)
    return thresh
