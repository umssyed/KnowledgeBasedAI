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
import numpy
import cv2


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
        ratio = img2DPR / img1DPR
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

    def compareXY(self, imgX, imgY):
        '''This function compares two images X and Y'''
        transformation = {
            'exactImage': False,
            'isMirrored': False,
        }

        thresh_vertical = self.matchTemplate(imgX, cv2.flip(imgY, 0)) #flip x-axis/vertical
        thresh_horizontal = self.matchTemplate(imgX, cv2.flip(imgY, 1)) #flip y-axis/horizontal
        thresh_both = self.matchTemplate(imgX, cv2.flip(imgY, -1)) #flip xy-axis/both
        if thresh_vertical > 0.85:
            transformation['isMirrored'] = 'vertical'
        elif thresh_horizontal > 0.85:
            transformation['isMirrored'] = 'horizontal'
        elif thresh_both > 0.85:
            transformation['isMirrored'] = 'both'
        else:
            transformation['isMirrored'] = False

        return transformation

    def answer3by3(self, imgF, imgH, answer, row1, row2, row3, AE_DPR, imgE):
        ''''''
        px_threshold = 50
        DPR_threshold = 0.05
        result = -1

        row1_DPRchange = self.percDiff(row1['BC_DPR'], row1['AB_DPR'])
        row2_DPRchange = self.percDiff(row2['EF_DPR'], row2['DE_DPR'])
        print(row1_DPRchange)
        print(row2_DPRchange)
        lb = row1_DPRchange - abs(0.1*row1_DPRchange)
        ub = row1_DPRchange + abs(0.1*row1_DPRchange)


        if lb <= row2_DPRchange <= ub:
            for i in range(0, len(answer)):
                HX_DPR = self.calcDPR(answer[i]) / self.calcDPR(imgH)
                row3_DPRchange = self.percDiff(HX_DPR, row3['GH_DPR'])
                if lb <= row3_DPRchange <= ub:
                    print('Rule DPR:')
                    print(f"image is: {i + 1}")
                    result = i+1

        #If

        #If Percentage of pixel change across each image is very small (less than 10 percent)
        if result == -1 and abs(row1['perc_AB_BC_pixelChange']) <= 0.1 and abs(row2['perc_DE_EF_pixelChange']) <= 0.1:
            #Rule 1 - if EX_DPR pixel change is close to 0. It means shapes are same (C-01)
            if row1['avg_pixelChange'] <= px_threshold and row2['avg_pixelChange'] <= px_threshold:
                print('inside rule 1')
                for i in range(0, len(answer)):
                    # Find image X such that HX percentage difference is also close to 0
                    HX_pixelChange = self.pixelCount(answer[i]) - self.pixelCount(imgH)
                    if HX_pixelChange == row3['GH_pixelChange']:
                        print('Rule 1:')
                        print(f"image is: {i+1}")
                        result = i+1

            #Rule 2 - if avg pixel change is larger and consistent in both rows (C-02)
            elif row1['avg_pixelChange'] - px_threshold <= row2['avg_pixelChange'] <= row1['avg_pixelChange'] + px_threshold:
                print('inside rule 2')
                px_avg = (row1['avg_pixelChange'] + row2['avg_pixelChange']) / 2
                for i in range(0, len(answer)):
                    # Find image X such that HX avg pixelChange is similar to row1 and row2
                    HX_pixelChange = self.pixelCount(answer[i]) - self.pixelCount(imgH)
                    row3_avg_pxChange = int(HX_pixelChange + row3['GH_pixelChange'])/2
                    if px_avg - px_threshold <= row3_avg_pxChange <= px_avg + px_threshold:
                        print('Rule 2:')
                        print(f"image is: {i+1}")
                        result = i+1

        #Check the three rows with first two columns
        if result == -1 and row1['AB_DPR'] == row2['DE_DPR'] == row3['GH_DPR']:
            print('inside DPR checking')
            if row1['BC_DPR'] == row2['EF_DPR']:
                #Find image X such that HX DPR is same as BC_DPR or EF_DPR
                for i in range(0, len(answer)):
                    HX_DPR = self.pixelCount(answer[i]) / self.pixelCount(imgH)
                    if HX_DPR == row1['BC_DPR'] == row2['EF_DPR']:
                        print(f"image is: {i+1}")
                        result = i+1



        #If no success diagonal EF and EH relationships with DPR C-12
        if result == -1:
            #Rule Diagonal
            print('inside diagonal rule')
            EF_DPR = self.calcDPR(imgF) / self.calcDPR(imgE)
            EH_DPR = self.calcDPR(imgH) / self.calcDPR(imgE)
            if round(EF_DPR, 1) == round(EH_DPR, 1):
                print('inside Diagonal')
                for i in range(0, len(answer)):
                    EX_DPR = self.calcDPR(answer[i]) / self.calcDPR(imgE)
                    print(EX_DPR)
                    if round(EX_DPR, 1) == round(EF_DPR, 1):
                        print(f"image is: {i+1}")
                        result = i+1
                    else:
                        result = -1


        return result



    def main3by3(self, question, unbin_answer):
        ''''This function takes in the question list of images, and answer list of images.
        It then sends the question images and answer images for comparisons and return
        the correct answer'''
        imgA = self.binarize(question[0])
        imgB = self.binarize(question[1])
        imgC = self.binarize(question[2])
        imgD = self.binarize(question[3])
        imgE = self.binarize(question[4])
        imgF = self.binarize(question[5])
        imgG = self.binarize(question[6])
        imgH = self.binarize(question[7])

        result = -1

        answer = [] #For binarized answers
        for ans in unbin_answer:
            bin = self.binarize(ans)
            answer.append(bin)
        row1 = {
            'AB_pixelChange': 0,
            'BC_pixelChange':0,
            'perc_AB_BC_pixelChange': 0,
            'avg_pixelChange': 0,
            'AB_DPR': 0,
            'BC_DPR': 0,
        }
        row2 = {
            'DE_pixelChange': 0,
            'EF_pixelChange': 0,
            'perc_DE_EF_pixelChange': 0,
            'avg_pixelChange': 0,
            'DE_DPR': 0,
            'EF_DPR': 0
        }
        row3 = {
            'GH_pixelChange': 0,
            'GH_DPR': 0
        }



        row1['AB_pixelChange'] = self.pixelCount(imgB) - self.pixelCount(imgA)
        row1['BC_pixelChange'] = self.pixelCount(imgC) - self.pixelCount(imgB)
        if row1['AB_pixelChange'] > 0:
            row1['perc_AB_BC_pixelChange'] = (row1['BC_pixelChange'] - row1['AB_pixelChange']) / row1['AB_pixelChange']
        row1['avg_pixelChange'] = int((row1['AB_pixelChange'] + row1['BC_pixelChange']) / 2)

        if self.calcDPR(imgA) > 0:
            row1['AB_DPR'] = self.calcDPR(imgB) / self.calcDPR(imgA)
        row1['BC_DPR'] = self.calcDPR(imgC) / self.calcDPR(imgB)

        row2['DE_pixelChange'] = self.pixelCount(imgE) - self.pixelCount(imgD)
        row2['EF_pixelChange'] = self.pixelCount(imgF) - self.pixelCount(imgE)
        if row2['DE_pixelChange'] > 0:
            row2['perc_DE_EF_pixelChange'] = (row2['EF_pixelChange'] - row2['DE_pixelChange']) / row2['DE_pixelChange']
        row2['avg_pixelChange'] = int((row2['DE_pixelChange'] + row2['EF_pixelChange']) / 2)
        row2['DE_DPR'] = self.calcDPR(imgE) / self.calcDPR(imgD)
        row2['EF_DPR'] = self.calcDPR(imgF) / self.calcDPR(imgE)

        row3['GH_pixelChange'] = self.pixelCount(imgH) - self.pixelCount(imgG)
        row3['GH_DPR'] = self.calcDPR(imgH) / self.calcDPR(imgG)


        print(f"row1: {row1}")
        print(f"row2: {row2}")
        print(f"row3: {row3}")

        AE_DPR = 0
        if self.calcDPR(imgA)> 0:
            AE_DPR = self.calcDPR(imgE) / self.calcDPR(imgA)

        #Before going to answer3by3, see if a transformation will help us:
        transAC = self.compareXY(imgA, imgC)
        transDF = self.compareXY(imgD, imgF)
        print(transAC)
        print(transDF)
        if transAC['isMirrored'] != False and transAC['isMirrored'] == transDF['isMirrored']:
            transformation = transAC['isMirrored']
            for i in range(0, len(answer)):
                transGX = self.compareXY(imgG, answer[i])
                if transGX['isMirrored'] == transformation:
                    print(f"image is: {i + 1}")
                    result = i + 1


        if result == -1:
            result = self.answer3by3(imgF, imgH, answer, row1, row2, row3, AE_DPR, imgE)


        return result

    def Solve(self,problem):

        #For 3x3 problems only:
        print(f"Problem: {problem.figures['7'].visualFilename}")
        alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        question = []
        unbin_answer = []
        #Load questions
        i = 0
        for char in alphabet:
            file_name = problem.figures[char].visualFilename
            question.append(cv2.imread(file_name, 0))

        for i in range(1, 9):
            file_name = problem.figures[str(i)].visualFilename
            unbin_answer.append(cv2.imread(file_name, 0))


        result = self.main3by3(question, unbin_answer)
        print(f'THE FINAL RESULT IS: {result}\n')
        return result





    # The primary method for solving incoming Raven's Progressive Matrices.
    # For each problem, your Agent's Solve() method will be called. At the
    # conclusion of Solve(), your Agent should return an int representing its
    # answer to the question: 1, 2, 3, 4, 5, or 6. Strings of these ints
    # are also the Names of the individual RavensFigures, obtained through
    # RavensFigure.getName(). Return a negative number to skip a problem.
    #
    # Make sure to return your answer *as an integer* at the end of Solve().
    # Returning your answer as a string may cause your program to crash.