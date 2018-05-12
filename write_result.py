# -*- coding:utf-8 -*-
import os
import random
import cv2
import torch
from PIL import Image
from model_class import get_class
def test_video(srcfile,model_ft):
    loc_file=os.path.join('.','person_val',srcfile)
    video_file=os.path.join('videos_original',srcfile.replace('.txt','.mp4'))
    cap=cv2.VideoCapture(video_file)
    f=open(loc_file,'r')
    lines=f.readlines()
    resultname='result_'+srcfile
    fw=open('./result/'+resultname,'w')
    for line in lines:
        line_split=line.strip().split()
        frameid=line_split[0]
        x1=int(float(line_split[1]))
        y1=int(float(line_split[2]))
        x2=int(float(line_split[3]))
        y2=int(float(line_split[4]))
        cap.set(1,int(frameid))
        _,frame=cap.read()
        frame=frame[y1:y2,x1:x2]
        frame = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))  
        preds=get_class(frame,model_ft)
        if(preds==0):
            fw.write(line)



if __name__=='__main__':
    model_ft=torch.load('best_model_res.pkl')
    filenames=os.listdir('person_val')
    for file in filenames:
        print(file)
        test_video(file,model_ft)
