import numpy as np 
import cv2 
import argparse
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image as im
import os
from scipy.ndimage.filters import convolve

# Region Of Interest Class
class ROI:
    # Attributes initializer
    def __init__(self, image, image_scaling, sigma=2, kernel_size=5, weak_pixel = 200, strong_pixel = 255, lowthreshold = 0.09, highthreshold = 0.17):
        # attributes
        self.width = None
        self.height = None
        self.image = cv2.imread(image)
        self.scalling_fct = image_scaling
        self.size = kernel_size
        self.smoothed_image = None
        self.sigma = sigma
        self.gradientMat = None
        self.thetaMat = None
        self.nonMaxImg = None
        self.thresholding = None
        self.highthreshold = highthreshold
        self.lowthreshold = lowthreshold
        self.weak_pixel = weak_pixel
        self.strong_pixel = strong_pixel
        self.final_imgs = []
        self.mean = []
        self.run()

    # Resize input image
    def resize_img(self, img, scale_per):
        self.width =int(scale_per * img.shape[1] / 100)
        self.height =int(scale_per * img.shape[0] / 100)
        # resized = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)
        resized = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return resized

    # Image cropper
    def crop_img(self, img, x_cord = 0, y_cord = 0, width=200, height=200):
        cropped_img = []
        for key, imgs in enumerate(img):
            cropped_img.append(imgs[x_cord:width, y_cord:height])
        return cropped_img

    # Generate Gaussian Kernel for Canny Image
    def gaussian_kernel(self, size=5, sigma=1):
        size = int(size) // 2
        x, y = np.mgrid[-size:size+1, -size:size+1]
        normal = 1/(2.0 * np.pi * sigma**2)
        g = np.exp(-((x**2 + y**2)/(2.0*sigma**2))) * normal
        return g

    # Sobel fiter for Image Gradient and Theta
    def sobel_filter(self, img):
        Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
        Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

        Ix = ndimage.convolve(img, Kx)
        Iy = ndimage.convolve(img, Ky)

        G = np.hypot(Ix, Iy)
        G = G / G.max() * 255
        theta = np.arctan2(Iy, Ix)
        return (G, theta)
    
    def NMS(self, img, D):
        M, N = img.shape
        Z = np.zeros((M,N), dtype=np.int32)
        angle = D * 180. / np.pi
        angle[angle < 0] += 180

        for i in range(1,M-1):
            for j in range(1,N-1):
                try:
                    q = 255
                    r = 255

                   #angle 0
                    if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                        q = img[i, j+1]
                        r = img[i, j-1]
                    #angle 45
                    elif (22.5 <= angle[i,j] < 67.5):
                        q = img[i+1, j-1]
                        r = img[i-1, j+1]
                    #angle 90
                    elif (67.5 <= angle[i,j] < 112.5):
                        q = img[i+1, j]
                        r = img[i-1, j]
                    #angle 135
                    elif (112.5 <= angle[i,j] < 157.5):
                        q = img[i-1, j-1]
                        r = img[i+1, j+1]

                    if (img[i,j] >= q) and (img[i,j] >= r):
                        Z[i,j] = img[i,j]
                    else:
                        Z[i,j] = 0


                except IndexError as e:
                    pass
        return Z
    
    # Generate Threshold
    def threshold(self, img):
        highthreshold = img.max() * self.highthreshold
        lowthreshold = self.highthreshold * self.lowthreshold

        M, N = img.shape
        res = np.zeros((M,N), dtype=np.int32)

        weak = np.int32(self.weak_pixel)
        strong = np.int32(self.strong_pixel)

        strong_i, strong_j = np.where(img >= highthreshold)
        zeros_i, zeros_j = np.where(img < lowthreshold)

        weak_i, weak_j = np.where((img <= highthreshold) & (img >= lowthreshold))

        res[strong_i, strong_j] = strong
        res[weak_i, weak_j] = weak

        return (res)
    
    # Create histeresis of image
    def hysteresis(self, img):
        M, N = img.shape
        weak = self.weak_pixel
        strong = self.strong_pixel

        for i in range(1, M-1):
            for j in range(1, N-1):
                if (img[i,j] == weak):
                    try:
                        if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                            or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                            or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                            img[i, j] = strong
                        else:
                            img[i, j] = 0
                    except IndexError as e:
                        pass
        return img
    
    def mean_array(self, arr):
        return sum(arr) / len(arr)

    def detect_edges(self, input_img):
        for i, img in enumerate(input_img):
            self.smoothed_image = convolve(img, self.gaussian_kernel(self.size, self.sigma))
            self.gradientMat, self.thetaMat = self.sobel_filter(self.smoothed_image)
            self.nonMaxImg = self.NMS(self.gradientMat, self.thetaMat)
            self.thresholding = self.threshold(self.nonMaxImg)
            imgs = self.hysteresis(self.thresholding)
            self.mean.append(self.mean_array(imgs))
            self.final_imgs.append(imgs)

        return self.final_imgs
    
    def load_data(self, dir_name = 'images'):    
        '''
        Load images from the "faces_imgs" directory
        Images are in JPG and we convert it to gray scale images
        '''
        imgs = []
        for filename in os.listdir(dir_name):
            img = mpimg.imread(dir_name + '/' + filename)
            #img = skimage.color.rgb2gray(img)
            imgs.append(img)
        return imgs

    def visualize(self, imgs, format=None, gray=False):
        for i, img in enumerate(imgs):
            if img.shape[0] == 3:
                img = img.transpose(1,2,0)
            plt_idx = i+1
            plt.subplot(2, 2, plt_idx)
            # print(img)
            plt.imshow(img, format)
        plt.show()

    # BGR to RGB
    def RGB_img(self, img):
        rgb_img = []
        for key, imgs in enumerate(img):
            rgb_img.append(cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB))
        return rgb_img
    
    # RGB to gray image transformation
    def gray_img(self, img):
        gray_imgs = []
        for key, imgs in enumerate(img):
            gray_imgs.append(cv2.cvtColor(imgs, cv2.COLOR_RGB2GRAY))
        return gray_imgs
    
    # blur image
    def blur_img(self, img, width=0, height=0):
        img = cv2.GaussianBlur(img, (height, width), cv2.BORDER_DEFAULT)
        return img

    # get edges of image OR Canny Image transformaion 
    def edges_img(self, img, max_intensity = 100, min_intensity = 100):
        img = cv2.Canny(img, max_intensity, min_intensity)
        return img
    
    # Dilation of Image
    def dilation(self, img, kernel = np.ones((5, 5), np.uint8), iter=1):
        img = cv2.dilate(img, kernel, iter)
        return img

    # erosion of image
    def erosion(self, img, kernel = np.ones((5, 5), np.uint8), iter=1):
        img = cv2.erode(img, kernel, iter)
        return img
    
    # find Contours from Image
    def find_contours(self, img):
        contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        return contours
    
    # draw contours on Image
    def draw_contours(self, img, contours):
        img = cv2.drawContours(img, contours, -1, (0, 255, 0), 3)
        return img
        
    # diplay image
    def disp(self, img):
        cv2.imshow('Window', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def disp_img(self, img):
        plt.imshow(img[:, : ,::-1])
        plt.show()

    # Image processing
    def run(self):
        # External Attributes
        # kernel_1 = np.ones((3, 3), np.uint16)        
        # kernel_2 = np.ones((1, 1), np.uint16)
        # resize_img = self.resize_img(self.image, self.scalling_fct)
        # img = self.RGB_img(resize_img)
        # img = self.gray_img(img)
        # img = self.blur_img(img)
        # img = self.edges_img(img, 10, 45)
        # img = self.erosion(img)
        # img = self.dilation(img)
        # self.disp(img)
        imgs = self.load_data()
        
        # self.visualize(imgs)
        imgs = self.crop_img(imgs, 358, 1503, 363, 1508)
        # imgs = self.crop_img(imgs, 0, 0, 240, 240)
        self.visualize(imgs)
        imgs = self.RGB_img(imgs)
        self.visualize(imgs)
        imgs = self.gray_img(imgs)
        self.visualize(imgs)
        # g = self.gaussian_kernel(5, self.sigma)
        # print(g.shape)
        # print(self.image.shape)
        # print(imgs.shape)
        final_img = self.detect_edges(imgs)
        # print(imgs)
        self.visualize(final_img)

        plt.plot(self.mean)
        plt.show()
        
        # self.disp(imgs)
        # self.disp_img(final_img)


    # Arguments parser
    def argument_parser():
        ap = argparse.ArgumentParser()
        ap.add_argument("--image", type=str, default='./images/4.png', help="path to the input image")
        ap.add_argument("--image_scaling", type=int, default=25, help="add sclaing factor of image")
        args = ap.parse_args()
        return args

    # Main Func
    def main(args):
        ROI(**vars(args))
        
# Main Method
if '__main__' == __name__:
    input_args = ROI.argument_parser()
    ROI.main(input_args)