# Your Agent for solving Raven's Progressive Matrices. You MUST modify this file.
#
# You may also create and submit new files in addition to modifying this file.
#
# Make sure your file retains methods with the signatures:
# def __init__(self)
# def Solve(self,problem)
#
# These methods will be necessary for the project's main method to run.

# Install Pillow and uncomment this line to access image processing.
#from PIL import Image
import time

import numpy
import cv2
import timeit


class Agent:
    # The default constructor for your Agent. Make sure to execute any
    # processing necessary before your Agent starts solving problems here.
    #
    # Do not add any variables to this signature; they will not be used by
    # main().
    def __init__(self):
        pass

    def binarize(self, img):
        ''''This function returns a binarized image'''
        thresh, binImage = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
        return binImage

    def pixelCount(self, img):
        '''This function finds the number of black pixels in the image
        The function returns the number of black pixels in the image'''
        whitepx = cv2.countNonZero(img)
        blackpx = img.size - whitepx
        return blackpx

    def calcDPR(self,  img):
        darkpixels = self.pixelCount(img)
        totalpixels = img.size
        score_dpr = darkpixels/totalpixels
        return score_dpr

    def ratioDPR(self, imgA, imgB):
        img1DPR = self.calcDPR(imgA)
        img2DPR = self.calcDPR(imgB)
        if img1DPR > 0:
            ratio = img2DPR / img1DPR
        else:
            ratio = -1000
        return ratio

    def percDiff(self, newNum, origNum):
        if origNum == 0:
            return 0
        numerator = newNum - origNum
        percDiff = (numerator/origNum) * 100
        return percDiff

    def matchTemplate(self, imgX, imgY):
        '''This function checks the entire template of two images
        if exact images  then answer is 1.0, otherwise 0.0. The fn returns the threshold'''
        threshold = cv2.matchTemplate(imgX, imgY, cv2.TM_CCOEFF_NORMED)[0][0]
        return threshold

    def matchShapeUsingContours(self, imgX, imgY):
        '''This function checks similarity using contours
        if two images have similar contours then it is 0.0, otherwise it is a larger number'''
        contours1, _ = cv2.findContours(imgX, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours2, _ = cv2.findContours(imgY, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        if len(contours1) > 1 and len(contours2) > 1:
            cont_diff = cv2.matchShapes(contours1[1], contours2[1], cv2.CONTOURS_MATCH_I1, 0)
        else:
            cont_diff = cv2.matchShapes(contours1[0], contours2[0], cv2.CONTOURS_MATCH_I1, 0)
        return cont_diff

    def numOfContours(self, imgX):
        '''The function returns num of contours'''
        contours, _ = cv2.findContours(imgX, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        numOfContours = len(contours)
        if numOfContours > 1:
            numOfContours -= 1
        return numOfContours

    def findOrientation(self, imgX):
        '''Finds the orientation of ONE shape in an image.
        Provided if there are SAME shapes within the image'''
        screenX = numpy.zeros(imgX.shape, dtype='uint8')

        thresh = cv2.threshold(imgX, 110, 255, cv2.THRESH_BINARY_INV)[1]
        # get contours and keep largest
        cntrs = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        #find outer contour
        cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]

        #cntrs = cv2.findContours(screenX, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #cntrs = cntrs[0] if len(cntrs) == 2 else cntrs[1]
        # get rotated rectangle from outer contour
        rotrect = cv2.minAreaRect(cntrs[0])
        box = cv2.boxPoints(rotrect)
        box = numpy.int0(box)

        # draw rotated rectangle on copy of img as result
        result = imgX.copy()
        cv2.drawContours(result, [box], 0, (0, 0, 255), 2)

        # get angle from rotated rectangle
        angle = rotrect[-1]
        # the `cv2.minAreaRect` function returns values in the
        # range [-90, 0); as the rectangle rotates clockwise the
        # returned angle trends to 0 -- in this special case we
        # need to add 90 degrees to the angle
        if angle < -45:
            angle = -(90 + angle)
        # otherwise, just take the inverse of the angle to make
        # it positive
        else:
            angle = angle
        return angle


    def identifyShapes(self, imgX):
        '''This function identifies shapes and returns a dictionary with shapes and numbers'''
        dictShapes = {
            "numOfShapes": 0,
            "numOfSides": 0,
            "sameShapes": False,
            "triangle": 0,
            "square": 0,
            "star": 0,
            "circle": 0,
            "unknownShapes": 0,
            "orientIfSameShape": -1
        }

        # apply canny edge detection
        edges = cv2.Canny(imgX, 90, 130)

        # apply morphology close
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morph = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # get contours and keep largest
        contours = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = contours[0] if len(contours) == 2 else contours[1]
        dictShapes['numOfShapes'] = len(contours)
        big_contour = 0
        if len(contours) > 0:
            big_contour = max(contours, key=cv2.contourArea)

            # get number of vertices (sides)
            peri = cv2.arcLength(big_contour, True)
            approx = cv2.approxPolyDP(big_contour, 0.03 * peri, True)
            dictShapes['numOfSides'] = len(approx)




        for cont in contours:
            p = cv2.arcLength(cont, True)
            aprx = cv2.approxPolyDP(cont, 0.03 * p, True)
            x = aprx.ravel()[0]
            y = aprx.ravel()[1]
            if len(aprx) == 3:
                dictShapes['triangle'] += 1
            elif len(aprx) == 4:
                dictShapes['square'] += 1
            elif len(aprx) == 8:
                dictShapes['circle'] += 1
            elif len(aprx) == 10:
                dictShapes['star'] += 1
            else:
                dictShapes['unknownShapes'] += 1

        sameShapes = False
        if dictShapes['triangle'] > 0 and (dictShapes['circle'] == dictShapes['star'] == dictShapes['square'] == 0):
            dictShapes['sameShapes'] = True
            dictShapes['orientation'] = self.findOrientation(imgX)
        elif dictShapes['circle'] > 0 and (dictShapes['triangle'] == dictShapes['star'] == dictShapes['square'] == 0):
            dictShapes['sameShapes'] = True
            dictShapes['orientation'] = self.findOrientation(imgX)
        elif dictShapes['star'] > 0 and (dictShapes['triangle'] == dictShapes['triangle'] == dictShapes['square'] == 0):
            dictShapes['sameShapes'] = True
            dictShapes['orientation'] = self.findOrientation(imgX)
        elif dictShapes['square'] > 0 and (dictShapes['triangle'] == dictShapes['triangle'] == dictShapes['triangle'] == 0):
            dictShapes['sameShapes'] = True
            dictShapes['orientation'] = self.findOrientation(imgX)
        else:
            dictShapes['sameShapes'] = False



        #cv2.waitKey(0)
        return dictShapes

    def shapesRepeating(self, imgA, imgB, imgC, imgD, imgE, imgF, imgG, imgH):
        '''Returns if the shapes are being repeated and what shape is then missing from 3rd row'''
        row1 = []
        row2 = []
        row3 = []
        shapesRepeating = False
        # First populate each row with what shape we have
        for keys in imgA:
            if keys == 'triangle' or keys == 'square' or keys == 'circle' or keys == 'star':
                if imgA[keys] > 0 and imgA['numOfShapes'] == imgA[keys]:
                    row1.append(keys)
        for keys in imgB:
            if keys == 'triangle' or keys == 'square' or keys == 'circle' or keys == 'star':
                if imgB[keys] > 0 and imgB['numOfShapes'] == imgB[keys]:
                    row1.append(keys)
        for keys in imgC:
            if keys == 'triangle' or keys == 'square' or keys == 'circle' or keys == 'star':
                if imgC[keys] > 0 and imgC['numOfShapes'] == imgC[keys]:
                    row1.append(keys)

        for keys in imgD:
            if keys == 'triangle' or keys == 'square' or keys == 'circle' or keys == 'star':
                if imgD[keys] > 0 and imgD['numOfShapes'] == imgD[keys]:
                    row2.append(keys)
        for keys in imgE:
            if keys == 'triangle' or keys == 'square' or keys == 'circle' or keys == 'star':
                if imgE[keys] > 0 and imgE['numOfShapes'] == imgE[keys]:
                    row2.append(keys)
        for keys in imgF:
            if keys == 'triangle' or keys == 'square' or keys == 'circle' or keys == 'star':
                if imgF[keys] > 0 and imgF['numOfShapes'] == imgF[keys]:
                    row2.append(keys)

        for keys in imgG:
            if keys == 'triangle' or keys == 'square' or keys == 'circle' or keys == 'star':
                if imgG[keys] > 0 and imgG['numOfShapes'] == imgG[keys]:
                    row3.append(keys)
        for keys in imgH:
            if keys == 'triangle' or keys == 'square' or keys == 'circle' or keys == 'star':
                if imgH[keys] > 0 and imgH['numOfShapes'] == imgH[keys]:
                    row3.append(keys)

        # Identify if there is a pattern
        row1 = list(dict.fromkeys(row1))
        row2 = list(dict.fromkeys(row2))
        missingShape = None
        if len(row1) == len(row2) == 3:
            print(f"row1: {row1}")
            print(f"row2: {row2}")
            shapesRepeating = True
            for i in range(0, len(row1)):
                if row1[i] not in row3:
                    missingShape = row1[i]
        return shapesRepeating, missingShape

    def isExact(self, imgX, imgY, threshold = None):
        '''This function comapres two images X and Y and returns if they are exact'''
        isExact = False
        if threshold == None:
            threshold = 0.85
        thresh = self.matchTemplate(imgX, imgY)
        #print(f"The threshhold from matchTemplate is: {thresh}")
        if thresh > threshold:
            isExact = True
        return isExact

    def isMirrored(self, imgX, imgY):
        '''This function compares two images X and Y and returns if they are mirrored'''
        isMirrored = False
        thresh_vertical = self.matchTemplate(imgX, cv2.flip(imgY, 0)) #flip x-axis/vertical
        thresh_horizontal = self.matchTemplate(imgX, cv2.flip(imgY, 1)) #flip y-axis/horizontal
        thresh_both = self.matchTemplate(imgX, cv2.flip(imgY, -1)) #flip xy-axis/both
        if thresh_vertical > 0.85:
            isMirrored = 'vertical'
        elif thresh_horizontal > 0.85:
            isMirrored = 'horizontal'
        elif thresh_both > 0.85:
            isMirrored= 'both'
        else:
            isMirrored = False

        return isMirrored

    def pxDensityLoc(self, img):
        '''This function returns the (x, y) coordinate of the avg black pixel
        density. Identifies where the center of mass of the black pixels is
        in the image'''
        x = 0
        y = 0
        x1 = 0
        y1 = 0
        counter = 0
        row_size = len(img)
        col_size = len(img)

        for i in range(0, row_size):
            for j in range(0, col_size):
                if img[i][j] < 150:
                    x += j
                    y += i
                    counter += 1
        x1 = int(x / counter)
        y1 = int(y / counter)
        # black = (0, 0, 0)
        # white = (255, 255, 255)
        # image = cv2.circle(IMG, (x1, y1), 2, black, 2)
        # cv2.imshow('c', image)
        # cv2.waitKey(0)

        return x1, y1


    def checkThreshold(self, a, b, t):
        '''Check to see if b is within a certain threshold of a. Returns true or false'''
        withinThreshold = False
        thresh = t/100
        lb = a - abs(thresh*a)
        ub = a + abs(thresh*a)
        if lb <= b <= ub:
            withinThreshold = True
        return withinThreshold

###########################################################
###########################################################
##################### 3 X 3 MATRIX ########################
#################### FRAME GENERATION #####################
    def rowCalculation(self, imgA, imgB, imgC, imgD, imgE, imgF, imgG, imgH):
        '''This function checks the relationships in each row and returns the frames for
        all three rows'''
        row1 = {
            'AB_pixelChange': 0,
            'BC_pixelChange': 0,
            'avg_pixelChange': 0,
            'AB_DPR': 0,
            'BC_DPR': 0,
            'AC_isExact': False,
            'AC_isMirrored': False,
        }
        row2 = {
            'DE_pixelChange': 0,
            'EF_pixelChange': 0,
            'avg_pixelChange': 0,
            'DE_DPR': 0,
            'EF_DPR': 0,
            'DF_isExact': False,
            'DF_isMirrored': False
        }
        row3 = {
            'GH_pixelChange': 0,
            'GH_DPR': 0
        }

        row1['AB_pixelChange'] = self.pixelCount(imgB) - self.pixelCount(imgA)
        row1['BC_pixelChange'] = self.pixelCount(imgC) - self.pixelCount(imgB)
        row1['avg_pixelChange'] = int((row1['AB_pixelChange'] + row1['BC_pixelChange']) / 2)
        if self.calcDPR(imgA) > 0:
            row1['AB_DPR'] = self.calcDPR(imgB) / self.calcDPR(imgA)
        else:
            row1['AB_DPR'] = -1
        if self.calcDPR(imgB) > 0:
            row1['BC_DPR'] = self.calcDPR(imgC) / self.calcDPR(imgB)
        else:
            row1['BC_DPR'] = -1

        row2['DE_pixelChange'] = self.pixelCount(imgE) - self.pixelCount(imgD)
        row2['EF_pixelChange'] = self.pixelCount(imgF) - self.pixelCount(imgE)
        row2['avg_pixelChange'] = int((row2['DE_pixelChange'] + row2['EF_pixelChange']) / 2)
        if self.calcDPR(imgD) > 0:
            row2['DE_DPR'] = self.calcDPR(imgE) / self.calcDPR(imgD)
        else:
            row2['DE_DPR'] = -1
        if self.calcDPR(imgE) > 0:
            row2['EF_DPR'] = self.calcDPR(imgF) / self.calcDPR(imgE)
        else:
            row2['EF_DPR'] = -1

        row3['GH_pixelChange'] = self.pixelCount(imgH) - self.pixelCount(imgG)
        if self.calcDPR(imgG) > 0:
            row3['GH_DPR'] = self.calcDPR(imgH) / self.calcDPR(imgG)
        else:
            row3['GH_DPR'] = -1

        row1['AC_isExact'] = self.isExact(imgA, imgC, 0.95)
        if row1['AC_isExact'] == False or row1['AC_isExact'] == None:
            row1['AC_isMirrored'] = self.isMirrored(imgA, imgC)
        row2['DF_isExact'] = self.isExact(imgD, imgF, 0.95)
        if row2['DF_isExact'] == False or row2['DF_isExact'] == None:
            row2['DF_isMirrored'] = self.isMirrored(imgD, imgF)

        return row1, row2, row3

    def colCalculation(self,  imgA, imgB, imgC, imgD, imgE, imgF, imgG, imgH):
        '''This function checks the relationships in each column and returns the frames for
        all three columns'''
        col1 = {
            'AD_pixelChange': 0,
            'DG_pixelChange': 0,
            'avg_pixelChange': 0,
            'AD_DPR': 0,
            'BC_DPR': 0
        }
        col2 = {
            'BE_pixelChange': 0,
            'EH_pixelChange': 0,
            'avg_pixelChange': 0,
            'BE_DPR': 0,
            'EH_DPR': 0
        }
        col3 = {
            'CF_pixelChange': 0,
            'CF_DPR': 0,
        }

        return col1, col2, col3

    def diagonalCalculation(self, imgA, imgB, imgC, imgD, imgE, imgF, imgG, imgH):
        diagAE = {
            'pixelChange': 0,
            'AE_DPR': 0,
            'isExact': False
        }
        diagBF = {
            'pixelChange': 0,
            'BF_DPR': 0,
            'isExact': False
        }
        diagDH = {
            'pixelChange': 0,
            'DH_DPR': 0,
            'isExact': False
        }

        diagFG ={
            'pixelChange': 0,
            'FG_DPR': 0,
            'isExact': False
        }

        diagCG ={
            'avg_pixelChange': 0,
            'CE_DPR': 0,
            'EG_DPR': 0,
            'isExact': False
        }

        #Diagonal AE:
        diagAE['pixelChange'] = self.pixelCount(imgE) - self.pixelCount(imgA)
        diagAE['isExact'] = self.isExact(imgA, imgE, 0.95)
        if self.calcDPR(imgA) > 0:
            diagAE['AE_DPR'] = self.ratioDPR(imgA, imgE)

        #Diagonal BF:
        diagBF['pixelChange'] = self.pixelCount(imgF) - self.pixelCount(imgB)
        diagBF['isExact'] = self.isExact(imgB, imgF, 0.95)
        if self.calcDPR(imgF) > 0:
            diagBF['BF_DPR'] = self.ratioDPR(imgB, imgF)

        #Diagonal DH:
        diagDH['pixelChange'] = self.pixelCount(imgH) - self.pixelCount(imgD)
        diagDH['isExact'] = self.isExact(imgD, imgH, 0.95)
        if self.calcDPR(imgH) > 0:
            diagDH['DH_DPR'] = self.ratioDPR(imgD, imgH)

        #Diagonal FG:
        diagFG['pixelChange'] = self.pixelCount(imgG) - self.pixelCount(imgF)
        diagFG['isExact'] = self.isExact(imgF, imgG, 0.95)
        if self.calcDPR(imgH) > 0:
            diagFG['FG_DPR'] = self.ratioDPR(imgF, imgG)

        #Diagonal CG:


        return diagAE, diagBF, diagDH, diagFG, diagCG

################ CHECK FOR ANSWERS 3 x 3 ###################
    def transformationCheck(self, transformation, imgG, answer):
        '''This function checks to find the equivalent transformation that occurred
        between A -> C and D -> F with G -> potential answer'''
        print("-> -> -> -> TRANSFORMATION CHECK")
        result = -1
        print(f"The transformation is: {transformation}")

        if transformation == 'isExact':
            for i in range(0, len(answer)):
                if self.isExact(imgG, answer[i]):
                    result = i + 1
        else:
            if transformation == 'vertical':
                for i in range(0, len(answer)):
                    if self.matchTemplate(answer[i], cv2.flip(imgG, 0)) > 0.85:
                        result = i +1
            elif transformation == 'horizontal':
                for i in range(0, len(answer)):
                    if self.matchTemplate(answer[i], cv2.flip(imgG, 1)) > 0.85:
                        result = i +1
            elif transformation == 'both':
                for i in range(0, len(answer)):
                    if self.matchTemplate(answer[i], cv2.flip(imgG, -1)) > 0.85:
                        result = i +1
            else:
                result = -1


        return result

    def rowRelationshipCheck(self, row1, row2, row3, imgG, imgH, answer):
        '''The function checks the row frames and finds an equivalent for imgH using
        DPR and pixel ratios and addition/subtraction of pixels'''
        print("-> -> -> -> ROW RELATIONSHIP CHECK")
        result_arr = []
        result = -1

        #1. If average pixel change is 0 for three rows, find exact image.
        if row1['avg_pixelChange'] == 0 and row2['avg_pixelChange'] == 0 and row3['GH_pixelChange'] == 0:
            print("Rule 1")
            imgH_pixel = self.pixelCount(imgH)
            for i in range(0, len(answer)):
                imgX_pixel = self.pixelCount(answer[i])
                HX_pixelChange = imgX_pixel - imgH_pixel
                if HX_pixelChange == 0:
                    print(f"Appending image {i + 1} to result arr")
                    result_arr.append(i + 1)
            # remove duplicates from result array
            result_arr = list(dict.fromkeys(result_arr))

        #2. If average pixel change is within a 5% similarity of each other, use that same avg pixel change to find answer
        if len(result_arr) == 0 and self.checkThreshold(row1['avg_pixelChange'], row2['avg_pixelChange'], 5):
            print("Rule 2")
            imgH_pixel = self.pixelCount(imgH)
            for i in range(0, len(answer)):
                X_pixel = self.pixelCount(answer[i])
                HX_pixelChange = X_pixel - imgH_pixel
                row3_avg_pixelChange = (row3['GH_pixelChange']+ HX_pixelChange) / 2

                #Check if AND is a possibility?
                G_AND_H = cv2.bitwise_and(imgG, imgH)
                G_AND_H_px = self.pixelCount(G_AND_H)

                if self.checkThreshold(X_pixel, G_AND_H_px, 0.5):
                    print(f"We should exit from here. G AND H can produce a bitwise logic")
                    result = -1
                    break

                if self.checkThreshold(row2['avg_pixelChange'], row3_avg_pixelChange, 5):
                    print(f"Appending image {i + 1} to result arr")
                    result_arr.append(i + 1)
            # remove duplicates from result array
            result_arr = list(dict.fromkeys(result_arr))

        #3. If pixel change is consistent row wise and within a 8% similarity of each other, use that same consistency to find answer
        if len(result_arr) == 0 and self.checkThreshold(row1['AB_pixelChange'], row1['BC_pixelChange'], 8) and self.checkThreshold(row2['DE_pixelChange'], row2['EF_pixelChange'], 8):
            print('Rule 3')
            imgH_pixel = self.pixelCount(imgH)
            for i in range(0, len(answer)):
                HX_pixelChange = self.pixelCount(answer[i]) - imgH_pixel
                if self.checkThreshold(row3['GH_pixelChange'], HX_pixelChange, 8):
                    print(f"Appending image {i + 1} to result arr")
                    result_arr.append(i + 1)

            #remove duplicates from result array
            result_arr = list(dict.fromkeys(result_arr))

        # 4. If pixel Change is within 5% similarity and is negative then positive, then image is moving inwards. Use same similarity to find answer
        if len(result_arr) == 0 and self.checkThreshold(-row1['AB_pixelChange'], row1['BC_pixelChange'], 8) and self.checkThreshold(-row2['DE_pixelChange'], row2['EF_pixelChange'], 8):
            print('Rule 4')
            imgH_pixel = self.pixelCount(imgH)
            for i in range(0, len(answer)):
                HX_pixelChange = self.pixelCount(answer[i]) - imgH_pixel
                if self.checkThreshold(-row3['GH_pixelChange'], HX_pixelChange, 8):
                    print(f"Appending image {i + 1} to result arr")
                    result_arr.append(i + 1)
            #If there are more than one results then we need to find if first image is an exact match. Because that would mean one of the two image is incorrect.
            if len(result_arr) > 1 and not all(x == result_arr[0] for x in result_arr):
                print(f"Inside Rule 4 - attempting to resolve multiple answer by matching shapes")
                for i in range(0, len(result_arr)):
                    num = result_arr[i] - 1 #select answer image number
                    if not self.isExact(imgG, answer[num]):
                        result = num + 1
            # remove duplicates from result array
            result_arr = list(dict.fromkeys(result_arr))

        #5. If DPR is within a 5% similarity of each other row wise, use that same DPR to find answer
        #   Check first if AB, DE and GH have same DPRs
        if len(result_arr) == 0 and self.checkThreshold(row1['AB_DPR'], row2['DE_DPR'], 5) and self.checkThreshold(row2['DE_DPR'], row3['GH_DPR'], 5):
            print('Rule 5')
            #   Check second if BC and EF have same DPRs, then find answer with similar DPR
            if self.checkThreshold(row1['BC_DPR'], row2['EF_DPR'], 5):
                for i in range(0, len(answer)):
                    HX_DPR = self.ratioDPR(imgH, answer[i])
                    if HX_DPR != -1000 and self.checkThreshold(row2['EF_DPR'], HX_DPR, 5):
                        print(f"Appending image {i + 1} to result arr")
                        result_arr.append(i + 1)
            # remove duplicates from result array
            result_arr = list(dict.fromkeys(result_arr))




        print(result_arr)
        if result == -1:
            #if result arr has only one reported answer, return result
            if len(result_arr) == 1:
                result = result_arr[0]
            #check if result array reports multiple same answers
            if len(result_arr) > 1 and all(x == result_arr[0] for x in result_arr):
                result = result_arr[0]
            # Attempt to vote for answer
            if len(result_arr) > 1:
                print(f"\n**************Attempting to vote for answer***************\n")

                #Vote 1: Find pixel density location of G and H, find an answer with
                #        5% similarity in pixel density location
                print(f"Vote 1: px density location")
                x_imgG, y_imgG = self.pxDensityLoc(imgG)
                x_imgH, y_imgH = self.pxDensityLoc(imgH)
                if self.checkThreshold(x_imgG, x_imgH, 10) and not self.checkThreshold(y_imgG, y_imgH, 10):
                    print(f"Vote 1(a): X coordinate in imgG and imgH are same and Y coordinate is not. Check for answer with similar x coordinates.")
                    for i in range(0, len(result_arr)):
                        index = result_arr[i]
                        print(index)
                        ans = answer[index-1]
                        x_ans, y_ans = self.pxDensityLoc(ans)
                        print(f"For the possible image {index}, the x coord - y coord are: {x_ans, y_ans}")
                        if self.checkThreshold(x_imgG, x_ans, 10):
                            result = index
                elif not self.checkThreshold(x_imgG, x_imgH, 10) and self.checkThreshold(y_imgG, y_imgH, 10):
                    print(f"Vote 1(b): Y coordinate in imgG and imgH are same and X coordinate is not. Check for answer with similar y coordinates.")
                    for i in range(0, len(result_arr)):
                        index = result_arr[i]
                        ans = answer[index-1]
                        x_ans, y_ans = self.pxDensityLoc(ans)
                        print(f"For the possible image {index}, the x coord - y coord are: {x_ans, y_ans}")
                        if self.checkThreshold(y_imgG, y_ans, 10):
                            result = index

                # Vote 2: Check type of shapes and orientation
                G = self.identifyShapes(imgG)
                H = self.identifyShapes(imgH)

                if result == -1 and G['numOfSides'] == H['numOfSides'] and G['orientation'] == H['orientation']:
                    print(f"Vote 2: Same number of sides and shape orientatoin. Checking for orientation of one shape in the image")
                    for i in range(0, len(result_arr)):
                        index = result_arr[i]
                        ans = answer[index - 1]
                        X = self.identifyShapes(ans)
                        if H['numOfSides'] == X['numOfSides'] and H['orientation'] == X['orientation']:
                            result = index



            #Otherwise skip the question
            if result == -1:
                print("\n****ROW RELATIONSHIP CHECK FAILED****\n")
        return result

    def colRelationshipCheck(self):
        '''The function checks the column frames and finds an equivalent for imgF using
        DPR and pixel ratios and addition/subtraction of pixels'''
        print("-> -> -> -> COLUMN RELATIONSHIP CHECK")
        result = -1
        return result

    def diagRelationshipCheck(self, diagAE, diagBF, diagDH, diagFG, imgA, imgB, imgC, imgD, imgE, imgF, imgG, answer):
        '''The function checks the diagonal frames and finds an equivalent for imgE using
        DPR, pixel ratios and addition/subtraction of pixels'''
        print("-> -> -> -> DIAGONAL RELATIONSHIP CHECK")
        result_arr = []
        result = -1
        imgE_pixels = self.pixelCount(imgE)
        print(f"AE: {diagAE}")
        print(f"BF: {diagBF}")
        print(f"DH: {diagDH}")
        print(f"FG: {diagFG}")


        #1. If all three diagonals have 0 pixel change, then find exact match with img E
        if diagAE['pixelChange'] == diagBF['pixelChange'] == diagDH['pixelChange'] == 0:
            print('Rule 1')
            for i in range(0, len(answer)):
                imgX_pixels = self.pixelCount(answer[i])
                pixelChange = imgX_pixels - imgE_pixels
                if pixelChange == 0 or self.checkThreshold(imgX_pixels, imgE_pixels, 5):
                    print(f"Appending image {i + 1} to result arr")
                    result_arr.append(i + 1)
            # remove duplicates from result array
            result_arr = list(dict.fromkeys(result_arr))

        #2. If diagonal AE/BF pixelChange match and DH/FG pixelChange match with similar pixel change,
        #   then we have diagonals matching for two rows each.
        #   DH and EX would match too with similar pixel change. Find
        if self.checkThreshold(diagAE['pixelChange'], diagBF['pixelChange'], 5):
            if self.checkThreshold(diagDH['pixelChange'], diagFG['pixelChange'], 5):
                print('Rule 2')
                for i in range(0, len(answer)):
                    imgX_pixels = self.pixelCount(answer[i])
                    pixelChange = imgX_pixels - imgE_pixels
                    if self.checkThreshold(pixelChange, diagDH['pixelChange'], 5):
                        print(f"Appending image {i + 1} to result arr")
                        result_arr.append(i + 1)
                # remove duplicates from result array
                result_arr = list(dict.fromkeys(result_arr))

        #3. If all diagonal relationship DPRs have 5% similarity, find same for EX:
        if self.checkThreshold(diagAE['AE_DPR'], diagBF['BF_DPR'], 5) and self.checkThreshold(diagDH['DH_DPR'], diagFG['FG_DPR'], 5):
            print('Rule 3 - A-E-X')
            for i in range(0, len(answer)):
                EX_DPR = self.ratioDPR(imgE, answer[i])
                if self.checkThreshold(diagAE['AE_DPR'], EX_DPR, 5):
                    print(f"Appending image {i + 1} to result arr")
                    result_arr.append(i + 1)
            # remove duplicates from result array
            result_arr = list(dict.fromkeys(result_arr))

        # X - Specific to D-10: C to D is same as D to H
        '''
        if self.matchShapeUsingContours(imgA, imgE) < 0.15:
                print(f"ATTEMPT 1: D-10")
                for i in range(0, len(answer)):
                    thresh = self.matchShapeUsingContours(imgA, answer[i])
                    if thresh < 0.15 and not thresh == 0:
                        result_arr.append(i + 1)
                # remove duplicates from result array
                result_arr = list(dict.fromkeys(result_arr))
        '''

        print(result_arr)
        if result == -1:
            #if result arr has only one reported answer, return result
            if len(result_arr) == 1:
                result = result_arr[0]
            #check if result array reports multiple same answers
            if len(result_arr) > 1 and all(x == result_arr[0] for x in result_arr):
                result = result_arr[0]

            # Attempt to vote for answer
            if len(result_arr) > 1:
                print(f"\n**************Attempting to vote for answer***************\n")
                #Vote 1 - D10 problem. Find if other shapes are same
                for j in range(0, len(result_arr)):
                    answer_index = result_arr[j] - 1
                    threshWith_A = self.matchShapeUsingContours(imgA, answer[answer_index])
                    threshWith_E = self.matchShapeUsingContours(imgE, answer[answer_index])
                    if not threshWith_A < 0.01 and not threshWith_E < 0.01:
                        result = result_arr[j]


            if result == -1:
                print("****DIAGONAL RELATIONSHIP CHECK FAILED****\n")
        return result

    def bitwiseOpCheck(self, imgA, imgB, imgC, imgD, imgE, imgF, imgG, imgH, answer):
        '''The function checks the bitwise operations (AND/OR/XOR) and pixel AND OR XOR using
        DPR, pixel ratios and addition/subtraction of pixels'''
        print("-> -> -> -> BITWISE OPERATION CHECK")
        result_arr = []
        result = -1
        #Check row operations
        # 1(i). If pixels in A minus pixels in B result to pixels in C, check second row, then perform same for G pixels - H pixels to get I
        A_subtract_B_pixels = self.pixelCount(imgA) - self.pixelCount(imgB)
        D_subtract_E_pixels = self.pixelCount(imgD) - self.pixelCount(imgE)
        if self.checkThreshold(self.pixelCount(imgC), A_subtract_B_pixels, 5) and self.checkThreshold(self.pixelCount(imgF), D_subtract_E_pixels, 5):
            print('\n1(i) - Row Pixels (A-B)')
            G_subtract_H_pixels = self.pixelCount(imgG) - self.pixelCount(imgH)
            for i in range(0, len(answer)):
                if self.checkThreshold(self.pixelCount(answer[i]), G_subtract_H_pixels, 5):
                    print(f"Appending image {i + 1} to result arr")
                    result_arr.append(i + 1)

            # remove duplicates from result array
            result_arr = list(dict.fromkeys(result_arr))

        # 1(ii). If pixels in A plus pixels in B result to pixels in C, check second row, then perform same for G pixels + H pixels to get I
        A_plus_B_pixels = self.pixelCount(imgA) + self.pixelCount(imgB)
        D_plus_E_pixels = self.pixelCount(imgD) + self.pixelCount(imgE)
        if self.checkThreshold(self.pixelCount(imgC), A_plus_B_pixels, 5) and self.checkThreshold(self.pixelCount(imgF), D_plus_E_pixels, 5):
            print('\n1(ii) - Row Pixels (A+B)')
            G_plus_H_pixels = self.pixelCount(imgG) + self.pixelCount(imgH)
            for i in range(0, len(answer)):
                if self.checkThreshold(self.pixelCount(answer[i]), G_plus_H_pixels, 5):
                    print(f"Appending image {i + 1} to result arr")
                    result_arr.append(i + 1)
            # remove duplicates from result array
            result_arr = list(dict.fromkeys(result_arr))

        # 1(iii). If pixels B + C equals A pixels, check second row, then perform same for H + X pixels to equal G pixels
        B_plus_C_pixels = self.pixelCount(imgB) + self.pixelCount(imgC)
        E_plus_F_pixels = self.pixelCount(imgE) + self.pixelCount(imgF)
        if self.checkThreshold(self.pixelCount(imgA), B_plus_C_pixels, 5) and self.checkThreshold(self.pixelCount(imgD), E_plus_F_pixels, 5):
            print('\n1(iii) - Row Pixels (B+C)')
            imgG_pixels = self.pixelCount(imgG)
            for i in range(0, len(answer)):
                H_plus_X_pixels = self.pixelCount(imgH) + self.pixelCount(answer[i])
                if self.checkThreshold(H_plus_X_pixels, imgG_pixels, 5):
                    print(f"Appending image {i + 1} to result arr")
                    result_arr.append(i + 1)
            # remove duplicates from result array
            result_arr = list(dict.fromkeys(result_arr))


        # 2. bitwise AND
        # 2(i) A + B = C and D + E = F
        if self.isExact(imgC, cv2.bitwise_and(imgA, imgB)) and self.isExact(imgF, cv2.bitwise_and(imgD, imgE)):
            print('\n2i - AND')
            imgX = cv2.bitwise_and(imgG, imgH)
            for i in range(0, len(answer)):
                if self.isExact(imgX, answer[i], 0.88):
                    print(f"Appending image {i + 1} to result arr")
                    result_arr.append(i + 1)

            # remove duplicates from result array
            result_arr = list(dict.fromkeys(result_arr))



        # 2(ii) A AND C = B (B =/= C) and D AND F = E (E =/= F),
        if self.isExact(imgB, cv2.bitwise_and(imgA, imgC)) and self.isExact(imgE, cv2.bitwise_and(imgD, imgF)):
            print('\n2ii - AND')
            for i in range(0, len(answer)):
                imgX = cv2.bitwise_and(imgG, answer[i])
                if not self.isExact(imgH, answer[i]) and self.isExact(imgH, imgX):
                    print(f"Appending image {i + 1} to result arr")
                    result_arr.append(i + 1)

            # remove duplicates from result array
            result_arr = list(dict.fromkeys(result_arr))

        # 3. bitwise OR
        # 3(i) A OR B = C and D OR E = F
        A_OR_B = cv2.bitwise_or(imgA, imgB)
        D_OR_E = cv2.bitwise_or(imgD, imgE)
        if self.isExact(imgC, A_OR_B) and self.isExact(imgF, D_OR_E):
            print('\n3i - OR')
            G_OR_H = cv2.bitwise_or(imgG, imgH)
            for i in range(0, len(answer)):
                if self.isExact(answer[i], G_OR_H):
                    print(f"Appending image {i + 1} to result arr")
                    result_arr.append(i + 1)

            # remove duplicates from result array
            result_arr = list(dict.fromkeys(result_arr))


        # 4. bitwise XOR
        # 4(i) A XOR B = C and D XOR E = F
        A_XOR_B = cv2.bitwise_not(cv2.bitwise_xor(imgA, imgB))
        D_XOR_E = cv2.bitwise_not(cv2.bitwise_xor(imgD, imgE))
        if self.isExact(imgC, A_XOR_B) and self.isExact(imgF, D_XOR_E):
            print('\n4i - XOR')
            G_XOR_H = cv2.bitwise_not(cv2.bitwise_xor(imgG, imgH))
            for i in range(0, len(answer)):
                if self.isExact(answer[i], G_XOR_H):
                    print(f"Appending image {i + 1} to result arr")
                    result_arr.append(i + 1)

            # remove duplicates from result array
            result_arr = list(dict.fromkeys(result_arr))


        temp_result = [] #Only used for this function
        if result == -1:
            #if result arr has only one reported answer, return result
            if len(result_arr) == 1:
                result = result_arr[0]
            #check if result array reports multiple same answers
            if len(result_arr) > 1 and all(x == result_arr[0] for x in result_arr):
                result = result_arr[0]
            # Attempt to vote for answer
            if len(result_arr) > 1:
                print(f"\n**************Attempting to vote for answer***************\n")
                # Vote 1: Solve using shape matching
                for j in range(0, len(result_arr)):
                    answer_index = result_arr[j] - 1
                    cont = self.matchShapeUsingContours(imgG, answer[answer_index])
                    if result == -1 and cont < 0.20:
                        print(f"Vote 1: Attempt to solve multiple answers using shape matching")
                        print(f"The contours is: {cont}")
                        temp_result.append(result_arr[j])

                if result == -1 and len(temp_result) != 1:
                    temp_result = []
                # Vote 2: Solve by identifying shape orientation
                    x_imgD, y_imgD = self.pxDensityLoc(imgD)
                    x_imgE, y_imgE = self.pxDensityLoc(imgE)
                    x_imgF, y_imgF = self.pxDensityLoc(imgF)
                    print(f"D: {x_imgD}, {y_imgD}")
                    print(f"E: {x_imgE}, {y_imgE}")
                    print(f"F: {x_imgF}, {y_imgF}")

                    x_imgG, y_imgG = self.pxDensityLoc(imgG)
                    x_imgH, y_imgH = self.pxDensityLoc(imgH)
                    print(f"G: {x_imgG}, {y_imgG}")
                    print(f"H: {x_imgH}, {y_imgH}")


        if result == -1 and len(temp_result) == 1:
            result = temp_result[0]

        if result == -1:
                print("\n****BITWISE OPERATION CHECK FAILED****\n")
        print(f"\n\nThe final result array is: {result_arr}\n")
        return result

    def shapesCheck(self, imgA, imgB, imgC, imgD, imgE, imgF, imgG, imgH, answer):
        '''The function checks shapes and number of shapes to return an answer'''
        print("-> -> -> -> SHAPE DETECTION OPERATION CHECK")
        result_arr = []
        result = -1

        A = self.identifyShapes(imgA)
        B = self.identifyShapes(imgB)
        C = self.identifyShapes(imgC)
        D = self.identifyShapes(imgD)
        E = self.identifyShapes(imgE)
        F = self.identifyShapes(imgF)
        G = self.identifyShapes(imgG)
        H = self.identifyShapes(imgH)
        print(f"A: {A}")
        print(f"B: {B}")
        print(f"C: {C}")
        print(f"D: {D}")
        print(f"E: {E}")
        print(f"F: {F}")
        print(f"G: {G}")
        print(f"H: {H}")

        #CHECK ROW RELATIONSHIP
        print(f"\nChecking Row relationships with shapes")
        #Check number of shapes in each row if they are the same
        if A['numOfShapes'] == B['numOfShapes'] == C['numOfShapes'] and D['numOfShapes'] == E['numOfShapes'] == F['numOfShapes']:
            pass

        #CHECK DIAGONAL RELATIONSHIP
        print(f"Checking diagonal relationships with shapes")
        # Check number of shapes in center diagonally if they are the same
        if A['numOfShapes'] == E['numOfShapes']:
            print('Rule 1')
            for i in range(0, len(answer)):
                condition = False
                X = self.identifyShapes(answer[i])
                if self.isExact(imgA, answer[i]):
                    condition = True
                elif self.isExact(imgB, answer[i]):
                    condition = True
                elif self.isExact(imgC, answer[i]):
                    condition = True
                elif self.isExact(imgD, answer[i]):
                    condition = True
                elif self.isExact(imgE, answer[i]):
                    condition = True
                elif self.isExact(imgF, answer[i]):
                    condition = True
                elif self.isExact(imgG, answer[i]):
                    condition = True
                elif self.isExact(imgH, answer[i]):
                    condition = True


                if X['numOfShapes'] == A['numOfShapes'] and condition == False:
                    print(f"Appending image {i + 1} to result arr")
                    result_arr.append(i + 1)
            # remove duplicates from result array
            result_arr = list(dict.fromkeys(result_arr))


        #2 . If pixel density diagonally is checked and has 5% similarity, find same for EX -- specific to D09
        #  Should move to diagonal relationships check:
        A_px = self.pixelCount(imgA)
        E_px = self.pixelCount(imgE)
        B_px = self.pixelCount(imgB)
        F_px = self.pixelCount(imgF)
        D_px = self.pixelCount(imgD)
        H_px = self.pixelCount(imgH)
        if self.checkThreshold(A_px, E_px, 5) and self.checkThreshold(B_px, F_px, 5) and self.checkThreshold(D_px, H_px, 5):
            print('Rule 2')
            for i in range(0, len(answer)):
                if not self.isExact(answer[i], imgA) and not self.isExact(answer[i], imgE):
                    X_px = self.pixelCount(answer[i])
                    if self.checkThreshold(E_px, X_px, 5):
                        print(f"Appending image {i + 1} to result arr")
                        result_arr.append(i + 1)
            # remove duplicates from result array
            result_arr = list(dict.fromkeys(result_arr))


        #3. If shapes are repeating and have a number pattern diagonally
        repeatingRowShapes, missingShape = self.shapesRepeating(A, B, C, D, E, F, G, H)
        if repeatingRowShapes and missingShape != None and A['numOfShapes'] == E['numOfShapes']:
            result_arr = []
            if B['numOfShapes'] == F['numOfShapes'] and D['numOfShapes'] == H['numOfShapes']:
                print('Rule 3 - A-E-X')
                for i in range(0, len(answer)):
                    X = self.identifyShapes(answer[i])
                    if A['numOfShapes'] == X['numOfShapes'] == X[missingShape]:
                        print(f"Appending image {i + 1} to result arr")
                        result_arr.append(i + 1)
                # remove duplicates from result array
                result_arr = list(dict.fromkeys(result_arr))

        #4. Specific D-08. Diagonal shape missing. Check for shapes diagonally
        if result == -1:
            #Enter code for D-08
            pass


        #print(f"\n\nThe final result array is: {result_arr}\n")
        if result == -1:
            #if result arr has only one reported answer, return result
            if len(result_arr) == 1:
                result = result_arr[0]
            #check if result array reports multiple same answers
            if len(result_arr) > 1 and all(x == result_arr[0] for x in result_arr):
                result = result_arr[0]

            if result == -1:
                print("****SHAPE DETECTION CHECK FAILED****\n")
        return result

##################### MAIN 3 by 3 #########################
    def main3by3(self, question, unbin_answer):
        ''''This function takes in the question list of images, and answer list of images.
        It then sends the question images and answer images for comparisons and return
        the correct answer'''
        result = -1
        imgA = self.binarize(question[0])
        imgB = self.binarize(question[1])
        imgC = self.binarize(question[2])
        imgD = self.binarize(question[3])
        imgE = self.binarize(question[4])
        imgF = self.binarize(question[5])
        imgG = self.binarize(question[6])
        imgH = self.binarize(question[7])

        answer = []  # For binarized answers
        for ans in unbin_answer:
            bin = self.binarize(ans)
            answer.append(bin)





        #Find row relationships
        row1, row2, row3 = self.rowCalculation(imgA, imgB, imgC, imgD, imgE, imgF, imgG, imgH)
        #Find column relationships
        col1, col2, col3 = self.colCalculation(imgA, imgB, imgC, imgD, imgE, imgF, imgG, imgH)
        # Find diagonal relationships
        diagAE, diagBF, diagDH, diagFG, diagCG = self.diagonalCalculation(imgA, imgB, imgC, imgD, imgE, imgF, imgG, imgH)
        #print(f"Initial result = {result}")
        print(f"Row1 Frame: {row1}")
        print(f"Row2 Frame: {row2}")
        print(f"Row3 Frame: {row3}\n")
        #print(f"diagAE Frame: {diagAE}")
        #print(f"diagBF Frame: {diagBF}")
        #print(f"diagDH Frame: {diagDH}")
        #print(f"diagFG Frame: {diagFG}")
        #print(f"diagCG Frame: {diagCG}\n")

        #Check for transformation isEXACT first, if not true, result = -1
        if result == -1 and (row1['AC_isExact'] != False and row2['DF_isExact'] != False):
            print("AC and DF is Exact check")
            if row1['AC_isExact'] == row2['DF_isExact']:
                transformation = 'isExact'
                result = self.transformationCheck(transformation, imgG, answer)

        #Then check for transformation isMIRRORED, if not true result = -1
        elif result == -1 and (row1['AC_isMirrored'] != False and row2['DF_isMirrored'] != False):
            print('here')
            if row1['AC_isMirrored'] == row2['DF_isMirrored']:
                transformation = row1['AC_isMirrored']
                result = self.transformationCheck(transformation, imgG, answer)

        #Then check for row relationships and compare that
        if result == -1:
            result = self.rowRelationshipCheck(row1, row2, row3, imgG, imgH, answer)

        #Then check for column relationships and compare that


        #Then check for diagonal relationships and compare that
        if result == -1:
            result = self.diagRelationshipCheck(diagAE, diagBF, diagDH, diagFG, imgA, imgB, imgC, imgD, imgE, imgF, imgG, answer)

        #Then check for bitwise operations
        if result == -1:
            result = self.bitwiseOpCheck(imgA, imgB, imgC, imgD, imgE, imgF, imgG, imgH, answer)

        #Then check for shapes operation
        if result == -1:
            result = self.shapesCheck(imgA, imgB, imgC, imgD, imgE, imgF, imgG, imgH, answer)
        #Otherwise skip

        return result

###########################################################
###########################################################
##################### 2 X 2 MATRIX ########################
#################### FRAME GENERATION #####################
    def rotate(self, img, angle, rotPoint=None):
        (height, width) = img.shape[:2]
        if rotPoint is None:
            rotPoint = (width // 2, height // 2)
        rotMat = cv2.getRotationMatrix2D(rotPoint, angle, 1.0)
        dimensions = (width, height)
        return cv2.warpAffine(img, rotMat, dimensions)

    def isRotated(self, imgX, imgY):
        '''This function rotates image Y to check against image X.
        Returns a list of thetas for which image X and rotated image Y
        is exact. If not, returns False'''
        theta = []

        for i in range(0, 360, 45):
            if self.isExact(imgX, self.rotate(imgY, i)):
                theta.append(i)

        if len(theta) > 0:
            return theta
        else:
            return False

    def frameGenerator(self, imgA, imgB, imgC):
        '''This function checks the relationships in each row and returns the frames for
        all three rows'''
        row1 = {
            'A': {
                'pixelCount': 0,
                'identity': {
                    "numOfShapes": 0,
                    "numOfSides": 0,
                    "sameShapes": False,
                    "triangle": 0,
                    "square": 0,
                    "star": 0,
                    "circle": 0,
                    "unknownShapes": 0
                },
            },
            'B': {
                'pixelCount': 0,
                'identity': {
                    "numOfShapes": 0,
                    "numOfSides": 0,
                    "sameShapes": False,
                    "triangle": 0,
                    "square": 0,
                    "star": 0,
                    "circle": 0,
                    "unknownShapes": 0
                },
            },
            'AB_pixelChange': 0,
            'AB_DPR': 0,
            'AB_isExact': False,
            'AB_isMirrored': False,
            'AB_isRotated': False,
        }
        col1 = {
            'A': {
                'pixelCount': 0,
                'identity': {
                    "numOfShapes": 0,
                    "numOfSides": 0,
                    "sameShapes": False,
                    "triangle": 0,
                    "square": 0,
                    "star": 0,
                    "circle": 0,
                    "unknownShapes": 0
                },
            },
            'C': {
                'pixelCount': 0,
                'identity': {
                    "numOfShapes": 0,
                    "numOfSides": 0,
                    "sameShapes": False,
                    "triangle": 0,
                    "square": 0,
                    "star": 0,
                    "circle": 0,
                    "unknownShapes": 0
                },
            },
            'AC_pixelChange': 0,
            'AC_DPR': 0,
            'AC_isExact': False,
            'AC_isMirrored': False,
            'AC_isRotated': False,
        }

        row1['A']['identity'] = self.identifyShapes(imgA)
        row1['B']['identity'] = self.identifyShapes(imgB)
        col1['C']['identity'] = self.identifyShapes(imgC)
        col1['A']['identity'] = row1['A']['identity']

        #Row1:
        row1['A']['pixelCount'] = self.pixelCount(imgA)
        row1['B']['pixelCount'] = self.pixelCount(imgB)
        row1['AB_pixelChange'] = row1['B']['pixelCount'] - row1['A']['pixelCount']
        row1['AB_DPR'] = self.ratioDPR(imgA, imgB)
        row1['AB_isExact'] = self.isExact(imgA, imgB)
        if row1['AB_isExact'] == False:
            row1['AB_isMirrored'] = self.isMirrored(imgA, imgB)
        if self.checkThreshold(row1['AB_DPR'], 1, 5):
            row1['AB_isRotated'] = self.isRotated(imgA, imgB)

        #Col1
        col1['A']['pixelCount'] = row1['A']['pixelCount']
        col1['C']['pixelCount'] = self.pixelCount(imgC)
        col1['AC_pixelChange'] = col1['C']['pixelCount'] - col1['A']['pixelCount']
        col1['AC_DPR'] = self.ratioDPR(imgA, imgC)
        col1['AC_isExact'] = self.isExact(imgA, imgC)
        if col1['AC_isExact'] == False:
            col1['AC_isMirrored'] = self.isMirrored(imgA, imgC)
        if self.checkThreshold(col1['AC_DPR'], 1, 5):
            col1['AC_isRotated'] = self.isRotated(imgA, imgC)

        print(f"Row1: {row1}")
        print(f"Col1: {col1}")
        return row1, col1

################ CHECK FOR ANSWERS 2 x 2 ###################
    def transformation(self, transformation, imgB, imgC, answer, checkFor = None):
        '''This function checks to find the equivalent transformation that occurred
        between A -> B and A -> C with B/C -> potential answer'''
        print("-> -> -> -> TRANSFORMATION CHECK")
        result = -1
        print(f"The transformation is: {transformation}")

        #Check for isEXACT first
        if transformation == 'AB_isExact':
            for i in range(0, len(answer)):
                if self.isExact(imgC, answer[i]):
                    result = i + 1

        elif transformation == 'AC_isExact':
            for i in range(0, len(answer)):
                if self.isExact(imgB, answer[i]):
                    result = i + 1
        else:
            #Check for isMIRRORED second
            if checkFor == 'row' and transformation == 'vertical':
                for i in range(0, len(answer)):
                    if self.isMirrored(imgC, answer[i]) == transformation:
                        result = i + 1
            elif checkFor == 'row' and transformation == 'horizontal':
                for i in range(0, len(answer)):
                    if self.isMirrored(imgC, answer[i]) == transformation:
                        result = i + 1
            elif checkFor == 'row' and transformation == 'both':
                for i in range(0, len(answer)):
                    if self.isMirrored(imgC, answer[i]) == transformation:
                        result = i + 1
            elif checkFor == 'col' and transformation == 'vertical':
                for i in range(0, len(answer)):
                    if self.isMirrored(imgB, answer[i]) == transformation:
                        result = i + 1
            elif checkFor == 'col' and transformation == 'horizontal':
                for i in range(0, len(answer)):
                    if self.isMirrored(imgB, answer[i]) == transformation:
                        result = i + 1
            elif checkFor == 'col' and transformation == 'both':
                for i in range(0, len(answer)):
                    if self.isMirrored(imgB, answer[i]) == transformation:
                        result = i + 1
            else:
                result = -1

        #CHECK FOR ROTATION IF RESULT = -1
        if result == -1 and type(transformation) == list:
            if checkFor == 'row':
                for j in range(0, len(transformation)):
                    theta = transformation[j]
                    for i in range(0, len(answer)):
                        if self.isExact(imgC, self.rotate(answer[i], theta)):
                            result = i + 1
            elif checkFor == 'col':
                for j in range(0, len(transformation)):
                    theta = transformation[j]
                    for i in range(0, len(answer)):
                        if self.isExact(imgB, self.rotate(answer[i], theta)):
                            result = i + 1

        print(f"\nResults sending back: {result}")
        return result

    def pixelCheck(self, row1, col1, imgB, imgC, answer):
        '''This function performs row and column checks using pixels and DPR'''
        print("-> -> -> -> PIXEL CHECK")
        result_arr = []
        result = -1

        #ROW
        #Check if num of shapes is same in A to B
        A_identity = row1['A']['identity']
        B_identity = row1['B']['identity']
        C_identity = col1['C']['identity']
        #Same number of sides but different pixel density. Use DPR.
        if A_identity['numOfSides'] == B_identity['numOfSides']:
            print('Rule 1: Same number of sides in A and B (same outer shape).')
            for i in range(0, len(answer)):
                print(f"\nAnswer {i+1}")
                ans_identity = self.identifyShapes(answer[i])
                X_pixelCount = self.pixelCount(answer[i])
                if C_identity['numOfSides'] == ans_identity['numOfSides']:
                    print('Rule 1(a): Find equivalent DPR')
                    CD_DPR = self.ratioDPR(imgC, answer[i])
                    if self.checkThreshold(CD_DPR, row1['AB_DPR'], 5):
                        print(f"Resulting in : {i + 1}")
                        result_arr.append(i + 1)

                elif C_identity['numOfSides'] == ans_identity['numOfSides'] and self.checkThreshold(X_pixelCount - col1['C']['pixelCount'], row1['AB_pixelChange'], 5) :
                    print('Rule 1(b): Check pixel changes row wise')
                    print(f"Resulting in : {i + 1}")
                    result_arr.append(i + 1)
            # remove duplicates from result array
            result_arr = list(dict.fromkeys(result_arr))

        #COLUMN


        #Total pixel calculation row and column. Perform same for D.
        if len(result_arr) == 0:
            print('\nLast Rule: Find total change in pixels row and column.')
            total_pxChange = row1['AB_pixelChange'] + col1['AC_pixelChange']
            print(f"Total px Change: {total_pxChange}")
            for i in range(0, len(answer)):
                X_pixelCount = self.pixelCount(answer[i])
                BX_pixelChange = X_pixelCount - row1['B']['pixelCount']
                CX_pixelChange = X_pixelCount - col1['C']['pixelCount']
                total = BX_pixelChange + CX_pixelChange
                print(f"result total: {total}")
                if self.checkThreshold(total, total_pxChange, 5):
                    print(f"Resulting in : {i + 1}")
                    result_arr.append(i + 1)
            # remove duplicates from result array
            result_arr = list(dict.fromkeys(result_arr))

        print(result_arr)
        if result == -1:
            # if result arr has only one reported answer, return result
            if len(result_arr) == 1:
                result = result_arr[0]
            # check if result array reports multiple same answers
            if len(result_arr) > 1 and all(x == result_arr[0] for x in result_arr):
                result = result_arr[0]

            if result == -1:
                print("****PIXEL CHECK FAILED****\n")
        print(f"\nResults sending back: {result}")
        return result

##################### MAIN 2 by 2 #########################
    def main2by2(self, question, unbin_answer):
        ''''This function takes in the question list of images, and answer list of images.
        It then sends the question images and answer images for comparisons and return
        the correct answer'''
        result = -1
        imgA = self.binarize(question[0])
        imgB = self.binarize(question[1])
        imgC = self.binarize(question[2])

        answer = []  # For binarized answers
        for ans in unbin_answer:
            bin = self.binarize(ans)
            answer.append(bin)

        #Get frame from frame generator
        row1, col1 = self.frameGenerator(imgA, imgB, imgC)


        #TRANSFORMATION ----------------------------------------------------
        # 1. CHECK FOR isEXACT
        #Check for transformation AB is exact first, if not true result is -1
        if result == -1 and row1['AB_isExact'] == True:
            print('\nAB is Exact Check')
            transformation = 'AB_isExact'
            result = self.transformation(transformation, imgB, imgC, answer)
        #Check for transformation AC is exact second, if not true result is -1
        elif result == -1 and col1['AC_isExact'] == True:
            print('\nAC is Exact Check')
            transformation = 'AC_isExact'
            result = self.transformation(transformation, imgB, imgC, answer)

        # 2. CHECK FOR isMIRRORED
        #Next check for transformation AB is mirrored, if not true result is -1
        if result == -1 and row1['AB_isMirrored'] != False:
            print('\nAB is Mirrored Check')
            transformation = row1['AB_isMirrored']
            checkFor = 'row'
            result = self.transformation(transformation, imgB, imgC, answer, checkFor)
        #Next check for transformation AC is mirrored, if not true result is -1
        if result == -1 and col1['AC_isMirrored'] != False:
            print('\nAC is Mirrored Check')
            transformation = col1['AC_isMirrored']
            checkFor = 'col'
            result = self.transformation(transformation, imgB, imgC, answer, checkFor)

        # 3. CHECK FOR ROTATION
        if result == -1 and row1['AB_isRotated'] != False:
            print('\nAB is Rotated Check')
            transformation = row1['AB_isRotated']
            checkFor = 'row'
            result = self.transformation(transformation, imgB, imgC, answer, checkFor)
        elif result == -1 and col1['AC_isRotated'] != False:
            print('\nAC is Rotated Check')
            transformation = col1['AC_isRotated']
            checkFor = 'col'
            result = self.transformation(transformation, imgB, imgC, answer, checkFor)


        #PIXEL CHANGES-- ----------------------------------------------------
        if result == -1:
            print('\nPixel Check')
            result = self.pixelCheck(row1, col1, imgB, imgC, answer)

        return result

###########################################################
###########################################################
##################### MAIN PROGRAM ########################
###########################################################
###########################################################
    def initiate3by3(self, problem):
        '''This functions intitiates the 3x3 problems'''
        print("======================================================")
        print("======================================================")
        print(f"Problem: {problem.figures['7'].visualFilename}")
        alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        question = []
        unbin_answer = []
        # Load questions
        i = 0
        for char in alphabet:
            file_name = problem.figures[char].visualFilename
            question.append(cv2.imread(file_name, 0))
        # Load answers
        for i in range(1, 9):
            file_name = problem.figures[str(i)].visualFilename
            unbin_answer.append(cv2.imread(file_name, 0))
        result = self.main3by3(question, unbin_answer)
        return result

    def initiate2by2(self, problem):
        '''This functions intitiates the 2x2 problems'''
        print("======================================================")
        print("======================================================")
        print(f"Problem: {problem.figures['A'].visualFilename}")
        alphabet = ['A', 'B', 'C']
        question = []
        unbin_answer = []
        # Load questions
        i = 0
        for char in alphabet:
            file_name = problem.figures[char].visualFilename
            question.append(cv2.imread(file_name, 0))
        # Load answers
        for i in range(1, 7):
            file_name = problem.figures[str(i)].visualFilename
            unbin_answer.append(cv2.imread(file_name, 0))

        result = self.main2by2(question, unbin_answer)
        return result

    def Solve(self,problem):
        start = timeit.default_timer()
        #For 3x3 problems only:
        result = -1

        if len(problem.figures) <= 10:
            #For 2x2 problems only
            result = self.initiate2by2(problem)
        elif len(problem.figures) > 10:
            #For 3x3 problems only
            result = self.initiate3by3(problem)

        if result == -1:
            result = 3

        print(f'\nTHE FINAL RESULT IS: {result}\n')
        stop = timeit.default_timer()
        execution_time = stop-start
        print(f"The execution time is: {execution_time} seconds. \n\n")
        return result
