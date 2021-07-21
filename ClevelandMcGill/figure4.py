import numpy as np
import os
import skimage.draw
import sys
import math

sys.path.append('../')


class Figure4:

    SIZE = (100, 100)

 
    @staticmethod
    def data_to_type1(data):
        '''
        '''
        barchart = np.zeros((100, 100), dtype=np.bool)

        # now we need 8 more pairs
        all_values = [0] * 10
        all_values[0] = np.random.randint(5, 86)
        all_values[1] = data[0] - 1  # fixed pos 1
        all_values[2] = data[1] - 1 # fixed pos 2
        all_values[3] = np.random.randint(5, 86)
        all_values[4] = np.random.randint(5, 86)
        all_values[5] = np.random.randint(5, 86)
        all_values[6] = np.random.randint(5, 86)
        all_values[7] = np.random.randint(5, 86)
        all_values[8] = np.random.randint(5, 86)
        all_values[9] = np.random.randint(5, 86)

        start = 0
        for i, d in enumerate(all_values):

            if i == 0:
                start += 2
            elif i == 5:
                start += 8
            else:
                start += 0

            gap = 2
            b_width = 7

            left_bar = start+i*gap+i*b_width
            right_bar = start+i*gap+i*b_width+b_width

            # print left_bar, right_bar

            rr, cc = skimage.draw.line(99, left_bar, 99-int(d), left_bar)
            barchart[rr, cc] = 1
            rr, cc = skimage.draw.line(99, right_bar, 99-int(d), right_bar)
            barchart[rr, cc] = 1
            rr, cc = skimage.draw.line(
                99-int(d), left_bar, 99-int(d), right_bar)
            barchart[rr, cc] = 1

            if i == 1 or i == 2:
                # mark the max
                # place dot here
                barchart[96:97, left_bar+b_width//2:left_bar+b_width//2+1] = 1

        return barchart


    @staticmethod
    def data_to_type2(data):
        '''
        '''
        barchart = np.zeros((100, 100), dtype=np.bool)

        # we build the barchart to the top
        all_values = [0] * 8
        all_values[0] = data[0] - 1  # fixed pos but max. 56
        current_max = 98-all_values[0]
        # print current_max/4.+1
        all_values[1] = np.random.randint(3, current_max//3.+1)

        all_values[2] = np.random.randint(3, current_max//3.+1)

        all_values[3] = np.random.randint(3, current_max//3.+1)

        current_max = np.sum(all_values[0:4])

        # draw left, right of the left stacked barchart
        rr, cc = skimage.draw.line(99, 10, 99-int(current_max), 10)
        barchart[rr, cc] = 1

        rr, cc = skimage.draw.line(99, 40, 99-int(current_max), 40)
        barchart[rr, cc] = 1

        current = 0
        for i, d in enumerate(all_values):

            rr, cc = skimage.draw.line(
                99-(int(d)+current), 10, 99-(int(d)+current), 40)
            barchart[rr, cc] = 1
            current += int(d)

            if i == 0:
                # mark the max
                barchart[99-int(d)//2:99-int(d)//2+1, 25:26] = 1

        all_values[4] = data[1] - 1  # fixed pos but max. 56
        current_max = 98-all_values[4]
        # print current_max/4.+1
        all_values[5] = np.random.randint(3, current_max//3.+1)

        all_values[6] = np.random.randint(3, current_max//3.+1)

        all_values[7] = np.random.randint(3, current_max//3.+1)

        current_max = np.sum(all_values[4:])

        # draw left, right of the left stacked barchart
        rr, cc = skimage.draw.line(99, 60, 99-int(current_max), 60)
        barchart[rr, cc] = 1

        rr, cc = skimage.draw.line(99, 90, 99-int(current_max), 90)
        barchart[rr, cc] = 1

        current = 0
        for i, d in enumerate(all_values[4:]):

            rr, cc = skimage.draw.line(
                99-(int(d)+current), 60, 99-(int(d)+current), 90)
            barchart[rr, cc] = 1
            current += int(d)

            if i == 0:
                # mark the max
                barchart[99-int(d)//2:99-int(d)//2+1, 75:76] = 1

        return barchart



    @staticmethod
    def data_to_type3(data):
        '''
        '''
        barchart = np.zeros((100, 100), dtype=np.bool)

        # now we need 8 more pairs
        all_values = [0] * 10
        all_values[0] = np.random.randint(5, 86)
        all_values[1] = data[0] - 1  # fixed pos 1
        all_values[2] = np.random.randint(5, 86)
        all_values[3] = np.random.randint(5, 86)
        all_values[4] = np.random.randint(5, 86)
        all_values[5] = np.random.randint(5, 86)
        all_values[6] = data[1]  # fixed pos 2
        all_values[7] = np.random.randint(5, 86)
        all_values[8] = np.random.randint(5, 86)
        all_values[9] = np.random.randint(5, 86)

        start = 0
        for i, d in enumerate(all_values):

            if i == 0:
                start += 2
            elif i == 5:
                start += 8
            else:
                start += 0

            gap = 2
            b_width = 7

            left_bar = start+i*gap+i*b_width
            right_bar = start+i*gap+i*b_width+b_width

            # print left_bar, right_bar

            rr, cc = skimage.draw.line(99, left_bar, 99-int(d), left_bar)
            barchart[rr, cc] = 1
            rr, cc = skimage.draw.line(99, right_bar, 99-int(d), right_bar)
            barchart[rr, cc] = 1
            rr, cc = skimage.draw.line(
                99-int(d), left_bar, 99-int(d), right_bar)
            barchart[rr, cc] = 1

            if i == 1 or i == 6:
                # mark the max
                barchart[96:97, left_bar+b_width//2:left_bar+b_width//2+1] = 1

        return barchart

        
    @staticmethod
    def data_to_type4(data):
        '''
        '''
        barchart = np.zeros((100, 100), dtype=np.bool)

        # we build the barchart to the top
        all_values = [0] * 8

        current_max = 98 - (data[0] - 1)

        all_values[0] = np.random.randint(3, current_max//3.+1)

        all_values[1] = np.random.randint(3, current_max//3.+1)

        all_values[2] = np.random.randint(3, current_max//3.+1)

        below_last_sum = np.sum(all_values[0:3])

        all_values[3] = data[0] - 1

        current_max = np.sum(all_values[0:4])

        above_last_sum = current_max

        # draw left, right of the left stacked barchart
        rr, cc = skimage.draw.line(99, 10, 99-int(current_max), 10)
        barchart[rr, cc] = 1

        rr, cc = skimage.draw.line(99, 40, 99-int(current_max), 40)
        barchart[rr, cc] = 1

        current = 0
        for i, d in enumerate(all_values):

            rr, cc = skimage.draw.line(
                99-(int(d)+current), 10, 99-(int(d)+current), 40)
            barchart[rr, cc] = 1
            current += int(d)

            if i == 3:
                # mark the max
                # lower left: (below_last_sum, 10)
                # upper right: (above_last_sum, 40)
                # print current, d
                barchart[99-current+(int(d)//2):99-current +
                         (int(d)//2)+1, 25:26] = 1

        below_last_sum2 = below_last_sum
        above_last_sum2 = above_last_sum

        # print below_last_sum, above_last_sum

        ctr = 0

        while below_last_sum2 == below_last_sum or above_last_sum2 == above_last_sum:

            if ctr == 20:
                # this happens around 10 times in 100.000 samples.. so not really important
                raise Exception()

            current_max = 98 - (data[1] - 1)

            all_values[4] = np.random.randint(3, current_max//3.+1)

            all_values[5] = np.random.randint(3, current_max//3.+1)

            all_values[6] = np.random.randint(3, current_max//3.+1)

            below_last_sum2 = np.sum(all_values[4:8])

            all_values[7] = data[1] - 1

            current_max = np.sum(all_values[4:])
            above_last_sum2 = current_max

            ctr += 1  # exception counter

        # draw left, right of the left stacked barchart
        rr, cc = skimage.draw.line(99, 60, 99-int(current_max), 60)
        barchart[rr, cc] = 1

        rr, cc = skimage.draw.line(99, 90, 99-int(current_max), 90)
        barchart[rr, cc] = 1

        current = 0
        for i, d in enumerate(all_values[4:]):

            rr, cc = skimage.draw.line(
                99-(int(d)+current), 60, 99-(int(d)+current), 90)
            barchart[rr, cc] = 1
            current += int(d)

            if i == 3:
                # mark the max
                # print current, d
                
                barchart[99-current+(int(d)//2):99-current +
                         (int(d)//2)+1, 75:76] = 1

        return barchart



    @staticmethod
    def data_to_type5(data):
        '''
        '''

        barchart = np.zeros((100, 100), dtype=np.bool)

        # we build the barchart to the top
        all_values = [0] * 8

        current_max = 98 - (data[0] - 1) - (data[1] - 1)

        if current_max <= 6:
            # this won't work
            raise Exception('Out of bounds')

        all_values[0] = np.random.randint(3, current_max//2.+1)

        all_values[1] = np.random.randint(3, current_max//2.+1)

        all_values[2] = data[0] - 1

        all_values[3] = data[1] - 1

        current_max = np.sum(all_values[0:4])

        # draw left, right of the left stacked barchart
        rr, cc = skimage.draw.line(99, 10, 99-int(current_max), 10)
        barchart[rr, cc] = 1

        rr, cc = skimage.draw.line(99, 40, 99-int(current_max), 40)
        barchart[rr, cc] = 1

        current = 0
        for i, d in enumerate(all_values):

            rr, cc = skimage.draw.line(
                99-(int(d)+current), 10, 99-(int(d)+current), 40)
            barchart[rr, cc] = 1
            current += int(d)

            if i == 2 or i == 3:
                # mark the max
                # print current, d
                barchart[99-current+(int(d)//2):99-current +
                         (int(d)//2)+1, 25:26] = 1

        current_max = 98

        all_values[4] = np.random.randint(3, current_max//3.+1)

        all_values[5] = np.random.randint(3, current_max//3.+1)

        all_values[6] = np.random.randint(3, current_max//3.+1)

        all_values[7] = np.random.randint(3, current_max//3.+1)

        current_max = np.sum(all_values[4:])

        while current_max > 97:
            current_max = 98

            all_values[4] = np.random.randint(3, current_max//3.+1)

            all_values[5] = np.random.randint(3, current_max//3.+1)

            all_values[6] = np.random.randint(3, current_max//3.+1)

            all_values[7] = np.random.randint(3, current_max//3.+1)

            current_max = np.sum(all_values[4:])

        # draw left, right of the left stacked barchart
        rr, cc = skimage.draw.line(99, 60, 99-int(current_max), 60)
        barchart[rr, cc] = 1

        rr, cc = skimage.draw.line(99, 90, 99-int(current_max), 90)
        barchart[rr, cc] = 1

        current = 0
        for i, d in enumerate(all_values[4:]):

            rr, cc = skimage.draw.line(
                99-(int(d)+current), 60, 99-(int(d)+current), 90)
            barchart[rr, cc] = 1
            current += int(d)

        return barchart

