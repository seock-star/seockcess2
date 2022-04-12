import cv2
import argparse
import numpy as np
import os
import sys
import glob


if __name__ == "__main__" :
 ap = argparse.ArgumentParser()
 ap.add_argument('-d', '--dataset', required = True, \
 help = 'Path to the input image')
 args = vars(ap.parse_args())
 dataset= args["dataset"]
 print('처리중...')
 verbose=True
 imagePath=os.path.join('D:\Emotion3\sleepy_face\train')
 for imagePath in glob.glob(dataset +"/*.jpg"): 
  if cv2.waitKey(1) & 0xFF ==ord("q"):
           break        
  print(imagePath, '처리중...')
  src= cv2.imread(imagePath)
  dst = cv2.resize(src, dsize=(48,48), interpolation=cv2.INTER_AREA)
  dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
  
  loc1=imagePath.rfind("\\")
  loc2=imagePath.rfind(".")
  fname='results/'+imagePath[loc1+1:loc2]+'.png'
  # if verbose:
   # cv2.imshow("show",dst)
   # cv2.waitKey(0)             
  
  cv2.imwrite(fname,dst)
  cv2.destroyAllWindows()