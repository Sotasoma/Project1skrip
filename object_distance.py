import pyrealsense2 as rs
import numpy as np
import cv2
import matplotlib.pyplot as plt



def ssd_detect(img):

    config_file='yolov4-tiny.cfg'
    frozen_model='yolov4-tiny.weights'

    model=cv2.dnn.readNet(frozen_model,config_file)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
    classLabels=[]
    file_name='coco.names'
    with open(file_name,'r')as f:
        classLabels= [line.strip() for line in f.readlines()] # we put the names in to an array
    # model.setInputSize(320,320)
    # model.setInputScale(1.0/127.5)##255/2=127.5
    # model.setInputMean((127.5,127.5,127.5))## mobilenet=>[-1,1]
    # model.setInputSwapRB(True)

    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))

    ClassIndex,confidence,bbox=model.detect(img,confThreshold=0.5) 
    for ClassInd,conf,boxes in zip(ClassIndex.flatten(),confidence.flatten(),bbox):
        if ClassInd in range(9,81):
            x=np.where(ClassIndex == ClassInd)
            ClassIndex= np.delete(ClassIndex, x)
            bbox=np.delete(bbox,x)
            
            

    font_scale=0.5
    font=cv2.FONT_HERSHEY_SIMPLEX
    for ClassInd,conf,boxes in zip(ClassIndex.flatten(),confidence.flatten(),bbox):
        cv2.rectangle(img,boxes,(255,0,0),2)
        cv2.putText(img,classLabels[ClassInd-1],(boxes[0]+0,boxes[1]-10),font,fontScale=font_scale,color=(0,255,0),thickness=2)


    plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    #cv2.imwrite("object.png",cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
    #frame=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return bbox,ClassIndex,confidence,img





print("reset start")
ctx = rs.context()
devices = ctx.query_devices()
for dev in devices:
    dev.hardware_reset()
print("reset done")
pipeline = rs.pipeline()

dist=cv2.VideoWriter('videos/output/distance_1mtr.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15,(640,480))
disparity=cv2.VideoWriter('videos/output/disparity_1mtr.avi',cv2.VideoWriter_fourcc(*'DIVX'), 15,(640,480))


config = rs.config()
config.enable_stream(rs.stream.infrared, 2, 640, 480, rs.format.y8, 30)
config.enable_stream(rs.stream.infrared, 1, 640, 480, rs.format.y8, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)


font_scale=3
font=cv2.FONT_HERSHEY_PLAIN

profile =pipeline.start(config)

try:
    while True:
        # Get frameset of color and depth
        frames = pipeline.wait_for_frames()
        ir_left = frames.get_infrared_frame(1) # Left IR Camera, it allows 1, 2 or no input
        img1 = np.asanyarray(ir_left.get_data())
        
        ir_right = frames.get_infrared_frame(2) # Left IR Camera, it allows 1, 2 or no input
        img2 = np.asanyarray(ir_right.get_data())
        cv2.namedWindow('IR Left', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('IR left', img1)
        
        color_frame = frames.get_color_frame()
        color_image=np.asanyarray(color_frame.get_data())
        cv2.imshow('RGB',color_image)
        
        
        try:
            box,Cindex,confidence,image=ssd_detect(color_image)
        except:
        
            continue
        

        distance_x=[]
        distance_y=[]
        try:
            for i in range(len(box)):
                x=(box[i][0]+box[i][2])/2
                y=(box[i][1]+box[i][3])/2
                distance_x.append(x)
                distance_y.append(y)
#             print('x-cord',distance_x)
#             print('y-cord',distance_y)
        except:continue


        # ------------------------------------------------------------
        # CALCULATE DISPARITY (DEPTH MAP)


        # StereoSGBM Parameter explanations:


        # Matched block size. It must be an odd number >=1 . Normally, it should be somewhere in the 3..11 range.
        block_size =31
        num_disp = 128


        stereo= cv2.StereoBM_create(
           
            numDisparities=num_disp,
            blockSize=block_size
            
            )
        disparity_SGBM = stereo.compute(img1,img2)

        plt.imshow(disparity_SGBM, cmap='plasma')
        plt.colorbar()
        plt.show()

        # Normalize the values to a range from 0..255 for a grayscale image
        disparity_SGBM1 = cv2.normalize(disparity_SGBM, disparity_SGBM, alpha=255,
                                      beta=0, norm_type=cv2.NORM_MINMAX)
        disparity_SGBM1 = np.uint8(disparity_SGBM)
        cv2.imshow("Disparity", disparity_SGBM)
       # disparity.write(disparity_SGBM)
        cv2.imwrite("disparity_SGBM_norm.png", disparity_SGBM1)
        disparity=[]
        for i in range(len(box)):
            disp=disparity_SGBM[int(distance_x[i])][int(distance_y[i])]
            disparity.append(disp/16)
        print(disparity)
        
        focal_length=6
        baseline=54.8
        alpha=87
        width_right=640
        f=1.933
        #f_pixel=(width_right*0.5)/np.tan(alpha*0.5*np.pi/180)
        f_pixel=942.8
        units=100
        distance=[]

        for i in range(len(disparity)):

                depth=(baseline*f_pixel)/(units*disparity[i])
                if disparity[i]!=0:
                    distance.append(depth/100)
                    continue
                else:
                    distance.append('0')
                    continue


        print('distance:',distance)

        
        font_scale=0.4
        font=cv2.FONT_HERSHEY_SIMPLEX
        for dist,boxes in zip(distance,box):
            if dist=='0':
                continue
            else:
                cv2.putText(image,str(float(dist))[0:4]+'m',(boxes[0]+70,boxes[1]-10),font,fontScale=font_scale,color=(0,255,0),thickness=2)
        cv2.imshow("Distance",image)
        
        cv2.imwrite("object3.png",cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        #dist.write(image)
        
        
        cv2.namedWindow('IR Right', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('IR Right', img2)
        key = cv2.waitKey(1)
        # Press esc or 'q' to close the image window
        if key & 0xFF == ord('q') or key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()