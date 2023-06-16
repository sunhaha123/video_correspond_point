
import librosa

import cv2

import os
#import mp3play
import numpy as np
import time
import random
from moviepy.editor import ImageSequenceClip, concatenate_videoclips, AudioFileClip
# g_FilterType = [cv2.COLORMAP_AUTUMN, cv2.COLORMAP_BONE, cv2.COLORMAP_JET, cv2.COLORMAP_WINTER, cv2.COLORMAP_RAINBOW, \
#                 cv2.COLORMAP_OCEAN, cv2.COLORMAP_SUMMER, cv2.COLORMAP_SPRING, cv2.COLORMAP_COOL, cv2.COLORMAP_HSV, cv2.COLORMAP_PINK , \
#                 cv2.COLORMAP_HOT ]
# 特效对应概率 
# -1 为原图 -2 为素描 -3 灰度图 
g_probabilitySpecial = [cv2.COLORMAP_AUTUMN, cv2.COLORMAP_BONE, cv2.COLORMAP_BONE, cv2.COLORMAP_WINTER,  cv2.COLORMAP_WINTER,\
                cv2.COLORMAP_OCEAN, cv2.COLORMAP_SUMMER, cv2.COLORMAP_SUMMER,cv2.COLORMAP_SUMMER, cv2.COLORMAP_SPRING, cv2.COLORMAP_SPRING,\
                cv2.COLORMAP_PINK , cv2.COLORMAP_PINK , cv2.COLORMAP_PINK ,\
                cv2.COLORMAP_HOT, -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-2,-2,-2,-2, -3,-3 ]
g_FilterType = [cv2.COLORMAP_AUTUMN, cv2.COLORMAP_BONE, cv2.COLORMAP_WINTER,  \
                cv2.COLORMAP_OCEAN, cv2.COLORMAP_SUMMER, cv2.COLORMAP_SPRING, cv2.COLORMAP_PINK , \
                cv2.COLORMAP_HOT ]
# 素描特效 开始
def dodgeNaive(image, mask):
    # determine the shape of the input image
    width, height = image.shape[:2]
 
    # prepare output argument with same size as image
    blend = np.zeros((width, height), np.uint8)
 
    for col in range(width):
        for row in range(height):
            # do for every pixel
            if mask[col, row] == 255:
                # avoid division by zero
                blend[col, row] = 255
            else:
                # shift image pixel value by 8 bits
                # divide by the inverse of the mask
                tmp = (image[col, row] << 8) / (255 - mask)
                # print('tmp={}'.format(tmp.shape))
                # make sure resulting value stays within bounds
                if tmp.any() > 255:
                    tmp = 255
                    blend[col, row] = tmp
 
    return blend
 
 
def dodgeV2(image, mask):
    return cv2.divide(image, 255 - mask, scale=256)
 
 
def burnV2(image, mask):
    return 255 - cv2.divide(255 - image, 255 - mask, scale=256)
 
# RGB 装换为 素描 
def rgb_to_sketch(srcImage):

    img_gray = cv2.cvtColor(srcImage, cv2.COLOR_BGR2GRAY)
    # 读取图片时直接转换操作
    # img_gray = cv2.imread('example.jpg', cv2.IMREAD_GRAYSCALE)
 
    img_gray_inv = 255 - img_gray
    img_blur = cv2.GaussianBlur(img_gray_inv, ksize=(21, 21),
                                sigmaX=0, sigmaY=0)
    return dodgeV2(img_gray, img_blur)

# 素描效果结束


# 生成 滤镜
# 传入 imread 后的 mat
# 随机生成滤镜
def creatFilterPic(im_gray, filterType):
    # picType = random.randint(0,11)
    #return cv2.applyColorMap(im_gray, g_FilterType[filterType])
    return cv2.applyColorMap(im_gray, filterType)


#判断变量类型的函数
def typeof(variate):
    type1 = ""
    if type(variate) == type(1):
        type1 = "int"
    elif type(variate) == type("str"):
        type1 = "str"
    elif type(variate) == type(12.3):
        type1 = "float"
    elif type(variate) == type([1]):
        type1 = "list"
    elif type(variate) == type(()):
        type1 = "tuple"
    elif type(variate) == type({"key1":"123"}):
        type1 = "dict"
    elif type(variate) == type({"key1"}):
        type1 = "set"
    return type1


# librosa 数字语音处理 获取节拍点
# 传入参数 音乐 传输 节点时间点(list) 
def getBeats(musicUrl):
    y, sr = librosa.load(musicUrl, sr=None)
    #tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    onset_env = librosa.onset.onset_strength(y=y, sr=sr, aggregate=np.median)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    beatsRes = np.array(librosa.frames_to_time(beats[:-1], sr=sr)) # 转为列表
    return beatsRes


# 特效 模糊边框的实现
def blurBorder(image):
    # 均值模糊 : 去随机噪声有很好的去噪效果
    # （1, 15）是垂直方向模糊，（15， 1）是水平方向模糊
    srcImage = image.copy() # 保存原始图像
    #print(image.shape)
    image = cv2.resize(image, (800, 800))
    #print(image.shape)
    dst = cv2.blur(image, (50, 50))  # 均值模糊
    #print(dst.shape)
    # 计算起点 src[200:300, 200:400] = backface
    # 800 533 
    height, width = srcImage.shape[0:2]
    if abs(height - 800) < 3:
        #roi = cv2.rectangle((0, (int)((800-width)/2)), (width, height))
        #left = (int)((800-width)/2)
        dst[0:height, (int)((800-width)/2):(int)((800-width)/2)+width] = srcImage
        #dst[(int)((800-width)/2):width, 0:height] = srcImage
    elif abs(width - 800) < 3:
        #up = (int)((800-height)/2)
        dst[(int)((800-height)/2):(int)((800-height)/2)+height, 0:width] = srcImage
        #dst[0, width, (int)((800-height)/2):height] = srcImage
        #roi = cv2.rectangle(((int)((800-height)/2), 0), (width, height))
    else:
        dst[(int)((800-height)/2):(int)((800-height)/2)+height, (int)((800-width)/2):(int)((800-width)/2)+width] = srcImage
        #roi = cv2.rectangle(((int)((800-height)/2), (int)((800-width)/2)), (width, height))
    return dst

# 尺寸调节
def resizePic(imSrc):
    # 打印出图片尺寸
    #print(imSrc.shape)
    
    # 将图片高和宽分别赋值给height(x)，width
    height, width = imSrc.shape[0:2]

    # 缩放的倍数
    zoomSmall = height / 800
    if (zoomSmall < width / 800):
        zoomSmall = width / 800
    print(zoomSmall)
    # 缩放到原来的 zoomSmall分之一，输出尺寸格式为（宽，高）
    return blurBorder(cv2.resize(imSrc, (int(width / zoomSmall), int(height / zoomSmall))))
    
    # # 最近邻插值法缩放
    # # 缩放到原来的四分之一
    # img_test2 = cv.resize(img, (0, 0), fx=0.25, fy=0.25, interpolation=cv.INTER_NEAREST)
if __name__ == '__main__':
    # # Load a mp3 file
    # y, sr = librosa.load('./111.mp3', sr=None)
    # #tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
    # onset_env = librosa.onset.onset_strength(y, sr=sr, aggregate=np.median)
    # tempo, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr)
    # res = np.array(librosa.frames_to_time(beats[:-1], sr=sr)) # 转为列表
    musicUrl = 'F:/1code/video_corr_point/example.mp3'
    # 预处理
    strPrePicUrl = 'F:/1code/video_corr_point/images/'
    picTotal = []
    
    for i in range(1,8):
        strPicUrl = strPrePicUrl + '('+str(i) + ')' +'.jpg'
        im = cv2.imread(strPicUrl)#, cv2.IMREAD_GRAYSCALE)
       
        im = resizePic(im)
        print(im.shape)
        picTotal.append(im)
    res = getBeats(musicUrl)
    res = np.insert(res, 0, 0)
    print(res) # 打印节拍点

    # 节拍点数量
    print("size: ", end = ' ')
    print(res.size)
    

    # file_name = "./example.mp3"
    # import mp3play
    # mp3 = mp3play.load(file_name)
    # mp3.play()  
    os.system('F:/1code/video_corr_point/example.mp3')  
    now = time.time()
    i = 0 
    
    #print(typeof(res[i]))
    picTypeTemp = -1
    ########################
    image_files=[]
    durations=[]
    # play pictures     
    while (i<res.size-1):
        
        picNum = random.randint(0,6)
        #strPicUrl = strPrePicUrl + str(picNum) + '.jpg'

        #im.resize()
        # 11 种 11 12 13 14 15 为原图
        picTypeTemp = random.randint(1,8)
        #picTypehaha = (picTypehaha + 1) % 28
        print("picType = ", end = '')
        print(picTypeTemp)
        picType = g_probabilitySpecial[picTypeTemp] # 概率获取图像
        #picType = (picType + 1) % 11
        #print(picType)
        print("picNum = ", end = '')
        print(picNum)
        print("picType = ", end = '')
        print(picType)

        im_color = picTotal[picNum]

        # if picType == -1:
        #     im_color = picTotal[picNum]
        # elif picType == -2:
        #     im_color = rgb_to_sketch(picTotal[picNum])
        # elif picType == -3:
        #     im_color = cv2.cvtColor(picTotal[picNum], cv2.COLOR_BGR2GRAY) 
        # else:
        #     im_color = creatFilterPic(picTotal[picNum], picType)
            
        cv2.imshow('output', im_color)
        cv2.imwrite(f'F:/1code/video_corr_point/output/{i}.jpg',im_color)
        delay = (int)(res[i+1]*1000) - (int)(res[i]*1000)
        print("delay", end = ' ')
        print(delay)
        cv2.waitKey((int)(delay)) 
        ######################
        image_files.append(f'F:/1code/video_corr_point/output/{i}.jpg')
        durations.append(delay/1000)
        
        # cv2.waitKey(1)
        print("peng!peng!peng!!!!!",end=' ')
        print(i)
        i = i+1
    print(image_files, durations)
    image_clips = [ImageSequenceClip([image_file], durations=[duration]) for image_file, duration in zip(image_files, durations)]
    final_clip = concatenate_videoclips(image_clips, method="compose")
    bg_music_path =  "F:/1code/video_corr_point/example.mp3"
    bg_music = AudioFileClip(bg_music_path)
    final_clip = final_clip.set_audio(bg_music)
    # 设置帧率
    fps = 28
    final_clip = final_clip.set_fps(fps)
    # 保存最终视频
    output_file = "F:/1code/video_corr_point/output_video.mp4"
    final_clip.write_videofile(output_file, codec="h264_nvenc")
