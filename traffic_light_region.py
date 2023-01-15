import argparse
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import shutil
import pickle
import pandas as pd
import numpy as np
import colorsys

# Region of Interest
class ROI:
    def __init__(self, input, region, night_mode):
        self.region = region
        self.mean = 0
        self.fps = 30
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.output_dir_path = './data'
        self.state_list = ['Green', 'Red', 'Blue', 'Yellow']
        self.pickle_model = open('./pickle_data/model.pkl', 'rb')
        self.clf = pickle.load(self.pickle_model)
        self.night_mode = night_mode

        # Checking input
        if os.path.isfile(input):
            # Further functionality needs to be added for Folder Inference :))
            if input[-4:] in ['.png', '.jpg']:
               self.input = cv2.imread(input)
               self.target_mode = 'Image'
            elif input[-4:] in ['.mp4', '.mkv', '.avi']:
                self.input = cv2.VideoCapture(input)
                self.target_mode = 'Video'
            else:
                print("Invalid input file. The file should be an image or a video !!")
                exit(-1)
        else:
            print("Input file doesn't exist. Check the input path")
            exit(-1)

        # Assigning ROI based on night or day video mode
        if self.night_mode:
            self.red_region = region[2]
            self.green_region = region[3]
        else:
            self.red_region = region[0]
            self.green_region = region[1]

        # Creating Output Directory
        if not os.path.exists(self.output_dir_path):
                os.makedirs(self.output_dir_path)
        else:
            shutil.rmtree(self.output_dir_path)       # Removes all the subdirectories!
            os.makedirs(self.output_dir_path)

        self.csv_data_raw = open(self.output_dir_path + '/state_data_raw.csv', 'a')
        self.csv_data_final = open(self.output_dir_path + '/state_data_final.csv', 'a')

        # Run Main method
        self.run()
    
    def region_of_interest(self, img, dims):
        '''
        Crop input images
        Returns region of interest image
        '''
        cropped_img =  img[dims[0]:dims[2], dims[1]:dims[3]]
        return cropped_img

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

    def rgb2hsv(self, rgb):
        '''
        takes RGB values as args and converts it into HSV form. HSV form is easy to define a color, intensity and grayness of color.
        returns the list of Three Values. e.g [123, 98, 74]
        Note: Hue Range: 0° to 360°, Saturation: from 0% to 100%, Values: from 0% to 100%
        '''
        params = []
        R, G, B = (rgb[0]/255, rgb[1]/255, rgb[2]/255)
        hsv = colorsys.rgb_to_hsv(R, G, B)
        hsv_list = list(i for i in hsv)
        params.extend(hsv_list)
        new_params = self.range_conversion(hsv_list)
        return new_params
    
    def RGB_list(self, img):
        ''' Takes Image input and creates list of RGB values'''
        dir = []
        for i in img:
            for j in i:
                dir.append(j.tolist())
        return dir
        
    def HSV_list(self, rgb):
        '''Takes RGB list as input and creates HSV List using func. rgb2hsv'''
        hsv_list = []
        for i in rgb:
            hsv_list.append(self.rgb2hsv(i))
        return hsv_list

    def pixel_change(self, img, hsv_list):
        '''Takes image and HSV list as input. It transforms RGB image into HSV image by replacing pixel values'''
        hsv_list_counter = 0
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                # print(hsv_list_counter)
                img[i, j] = (int(hsv_list[hsv_list_counter][0]), int(hsv_list[hsv_list_counter][1]), int(hsv_list[hsv_list_counter][2]))
                hsv_list_counter += 1
        return img

    def range_conversion(self, color_list):
        """
        Takes HSV color list as input
        Return: HSV list with different color range!
        eg. H: 0° - 360° --> H: 0° - 180°
            S: 0° - 100° --> S: 0 - 255
            V: 0° - 100° --> V: 0 - 255
        """
        new_hsv = []
        H = color_list[0] * 180
        S = color_list[1] * 255
        V = color_list[2] * 255

        new_hsv.extend([H, S, V])

        return new_hsv

    def frame_visualizer(self, imgs, format=None, gray=False):
        '''
        Display Images from the argument list.
        It displays max 4 images on a window.
        '''
        for i, img in enumerate(imgs):
            if img.shape[0] == 3:
                img = img.transpose(1,2,0)
            plt_idx = i+1
            plt.subplot(2, 2, plt_idx)
            # print(img)
            plt.imshow(img, format)
        plt.show()

    def post_processing(self, df):
        df_final = df.copy()
        df_final['State'].replace('', np.nan, inplace=True)
        df_final.dropna(subset=['State'], inplace=True)
        df_final.to_csv(self.csv_data_final, index=False)
        print("Finished post processing")

    def light_state(self, frame):
        rgb_list_red = self.RGB_list(frame[0])
        rgb_list_green = self.RGB_list(frame[1])
        
        hsv_list_red = self.HSV_list(rgb_list_red)
        hsv_list_green = self.HSV_list(rgb_list_green)

        # Mean of Hue values 
        hue_sum_red = 0     # Sum of all Hue values which satisfies the condition for RED Region
        hue_sum_green = 0   # Sum of all Hue values which satisfies the condition for GREEN Region
        length_green = 0    # Length of Green value
        length_red = 0      # Length of Red color
        
        # Intensity of Saturation and Value
        sat_on_pixels_red = 0     # number of Saturation Pixels that are Bright in Red Region
        sat_on_pixels_green = 0   # number of Saturation Pixels that are Bright in Green Region

        val_on_pixels_red = 0     # number of Value Pixels that are Bright in Red Region
        val_on_pixels_green = 0   # number of Value Pixels that are Bright in Green Region

        for i in hsv_list_green:
            if i[0] >= 25 and i[0] <= 40:
                hue_sum_green += i[0]
                length_green += 1
                if i[1] >= 170 and i[1] <= 255:
                    sat_on_pixels_green += 1
                if i[2] >= 170 and i[2] <= 255:
                    val_on_pixels_green += 1

        for j in hsv_list_red:
            if j[0] >= 90 and j[0] <= 120:
                hue_sum_red += j[0]
                length_red += 1
                if j[1] >= 123 and j[1] <= 255:
                    sat_on_pixels_red += 1
                if j[2] >= 193 and j[2] <= 255:
                    val_on_pixels_red += 1

        #hue_intensity_green = (hue_on_pixels_green/len(hsv_list_green)) * 100
        sat_intensity_green = (sat_on_pixels_green/len(hsv_list_green)) * 100
        val_intensity_green = (val_on_pixels_green/len(hsv_list_green)) * 100

        #hue_intensity_red = (hue_on_pixels_red/len(hsv_list_red)) * 100
        sat_intensity_red = (sat_on_pixels_red/len(hsv_list_red)) * 100
        val_intensity_red = (val_on_pixels_red/len(hsv_list_red)) * 100

        if length_green == 0:
            length_green = 1
        if length_red == 0:
            length_red = 1

        # Mean of Hue
        hsv_mean_green = hue_sum_green / length_green
        hsv_mean_red = hue_sum_red / length_red

        # Detection based on color Threshold 
        if self.night_mode:
            if (sat_intensity_green > sat_intensity_red) and (val_intensity_green > val_intensity_red):
                if hsv_mean_green >= 25 and hsv_mean_green <= 40:
                    return 'Green'
            elif (sat_intensity_red > sat_intensity_green) and (val_intensity_red > val_intensity_green):
                if (hsv_mean_red >= 90 and hsv_mean_red <= 120):
                    return 'Red'
            else:
                return 'Not Detected'

        else:
            if (sat_intensity_green > sat_intensity_red) and (val_intensity_green > val_intensity_red):
                if hsv_mean_green >= 25 and hsv_mean_green <= 40:
                    return 'Green'
            elif (sat_intensity_red > sat_intensity_green) and (val_intensity_red > val_intensity_green):
                if (hsv_mean_red >= 90 and hsv_mean_red <= 120):
                    return 'Red'
            else:
                return 'Not Detected'

        return 'Not Detected'

    # Run main
    def run(self):
        if self.target_mode == 'Video':
            # data store
            output_data = []
            # Display window parameters
            coordinates = (1495, 280)
            # undetected frame counter
            counter = 0

            if self.input.isOpened() == False:
                print('Error openning video file')
            while(self.input.isOpened()):
                ret, frame = self.input.read()
                if ret == True:

                    # Data Storage
                    state_data = {}
                    
                    # Region of Interest
                    img_red = self.region_of_interest(frame, self.red_region)
                    img_green = self.region_of_interest(frame, self.green_region)

                    # Light State
                    state = self.light_state([img_red, img_green])
                    state_data['State'] = state

                    if state == 'Not Detected':
                        cv2.imwrite('./not detected/img' + str(counter) + '.jpg', frame)
                        counter += 1

                    output_data.append(state_data)
                else:
                    break
                
            self.input.release()

            # State raw data
            df = pd.DataFrame(output_data)
            df.to_csv(self.csv_data_raw, index=False)

            # State post processing data
            self.post_processing(df)

        elif self.target_mode == 'Image':

            # Display window parameters
            coordinates = (1495, 280)

            # Region of Interest
            frame = self.input
            img_red = self.region_of_interest(frame, self.red_region)
            img_green = self.region_of_interest(frame, self.green_region)

            # Image Visualizer
            # self.frame_visualizer([img_green])

            rgb_list_red = self.RGB_list(img_red)
            rgb_list_green = self.RGB_list(img_green)
            
            hsv_list_red = self.HSV_list(rgb_list_red)
            hsv_list_green = self.HSV_list(rgb_list_green)

            for i in hsv_list_green:
                print(i[0])

            print()
            for i in hsv_list_green:
                print(i[1])

            print()
            for i in hsv_list_green:
                print(i[2])

            # Light State
            state = self.light_state([img_red, img_green])
            print(state)
            if state == self.state_list[1]:
                frame = cv2.putText(frame, state, coordinates, self.font, 1, (0, 0, 255), 2, cv2.LINE_AA)
            elif state == self.state_list[0]:
                frame = cv2.putText(frame, state, coordinates, self.font, 1, (0, 255, 0), 2, cv2.LINE_AA)
            elif state == 'Not Detected':
                frame = cv2.putText(frame, state, coordinates, self.font, 1, (200, 0, 0), 2, cv2.LINE_AA)
            cv2.imshow('Video', frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            
    # Argument Parser
    def argument_parser():
        ap = argparse.ArgumentParser()
        # ap.add_argument("--input", type=str, default='./videos/new_video.mp4', help=("path to the input file", "e.g. .mkv .mp4 .jpg .png"))
        ap.add_argument("--input", type=str, default='./not detected/img990.jpg', help=("path to the input file", "e.g. .mkv .mp4 .jpg .png"))
        ap.add_argument("--region", type=list, default = [[350, 1504, 354, 1508], [358, 1504, 362, 1508], [217, 1464, 221, 1468], [224, 1463, 228, 1467]], help="list of region dimensions eg. [x, y, width, height]")
        ap.add_argument("--night_mode", type=bool, default=False, help='detect in night video or day video')
        args = ap.parse_args()
        return args
    
    # Main Func
    def main(args):
        ROI(**vars(args))

# Main Method
if '__main__' == __name__:
    input_args = ROI.argument_parser()
    ROI.main(input_args)

















###########################################################################################################

# Data Collection
'''if state is self.state_list[0]:
    rgb_list_green = self.RGB_list(img_green)
    green_data = self.HSV_list(rgb_list_green)
    for i in green_data:
        raw_data = {'H': "{:.2f}".format(i[0]), 'S': "{:.2f}".format(i[1]), 'V': "{:.2f}".format(i[2]), 'State': state}
        storing_data.append(raw_data)
elif state is self.state_list[1]:
    rgb_list_red = self.RGB_list(img_red)
    red_data = self.HSV_list(rgb_list_red)
    for i in red_data:
        raw_data = {'H': "{:.2f}".format(i[0]), 'S': "{:.2f}".format(i[1]), 'V': "{:.2f}".format(i[2]), 'State':state}
        storing_data.append(raw_data)
else:
    pass'''


# if state == self.state_list[1]:
#     frame = cv2.putText(frame, state, coordinates, self.font, 1, (0, 0, 255), 2, cv2.LINE_AA)
# elif state == self.state_list[0]:
#     frame = cv2.putText(frame, state, coordinates, self.font, 1, (0, 255, 0), 2, cv2.LINE_AA)
# elif state == 'Not Detected':
#     frame = cv2.putText(frame, state, coordinates, self.font, 1, (200, 0, 0), 2, cv2.LINE_AA)
#     cv2.imwrite('./not detected/img' + str(counter) + '.jpg', frame)
#     counter += 1
# cv2.imshow('Video', frame)
# # Press Q on keyboard to  exit
# if cv2.waitKey(30) & 0xFF == ord('q'):
#     break


# coordinates for day video
# [[350, 1504, 354, 1508], [358, 1504, 362, 1508]]


# coordinates for night video
# [[217, 1464, 221, 1468], [224, 1463, 228, 1467]]


## Left off: Missing some frames from detection. need to tight up the threshold