import argparse
import random



rgb = [random.randrange(0, 255), random.randrange(0, 255), random.randrange(0, 255)]


class RGB_HSV:
    def __init__(self, rgb):
        self.rgb = rgb
        self.run()

    def rgb_hsv(self, rgb = [0, 255, 0]):
        params = []
        R, G, B = (rgb[0]/255, rgb[1]/255, rgb[2]/255)
        c_max = max(R, G, B)
        c_min = min(R, G, B)
        v = c_max

        if c_min == c_max:
            return 0.0, 0.0, v
        s = (c_max-c_min) / c_max
        rc = (c_max-R) / (c_max-c_min)
        gc = (c_max-G) / (c_max-c_min)
        bc = (c_max-B) / (c_max-c_min)
        if R == c_max:
            h = 0.0+bc-gc
        elif G == c_max:
            h = 2.0+rc-bc
        else:
            h = 4.0+gc-rc
        h = (h/6.0) % 1.0
        params.extend([h*360, s*100, v*100])
        return params

    def run(self):
        hsv = self.rgb_hsv(self.rgb)
        print(hsv)

    # Arguments parser
    def argument_parser():
        ap = argparse.ArgumentParser()
        ap.add_argument("--rgb", type=list, default=[0, 255, 0], help="list of RGB values")
        args = ap.parse_args()
        return args

    # Main Func
    def main(args):
        RGB_HSV(**vars(args))

if '__main__' == __name__:
    input_args = RGB_HSV.argument_parser()
    RGB_HSV.main(input_args)