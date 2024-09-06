from tools import *
import os 
import os.path
import numpy as np
import json


# Open the JSON file containing objects bounding boxs labels for each npy file 
#
with open('D:/labels_as_json/labels.json') as f:
    labels = json.load(f)
f.close()
npy_files=os.listdir('D:/vod_npy/vod_lidar_training_npy')
i=''
for ii in npy_files :
    i=os.path.splitext(ii)[0]
    #check if thee labels exists
    if i in labels.keys():
        npy_file=os.path.join('D:/vod_npy/vod_lidar_training_npy',ii)
        points=np.load(npy_file)
        rows,cols=np.shape(points)
        # get the extrinsic matrix parameter from lidar to camera
        matrix=t_camera_lidar()
        transformation_matrix=t_lidar_camera(matrix)
        # transform the pintcloud coordinates to camera coordinate system
        points_in_camera=transform_pcl(points,transformation_matrix)
        label_list=labels[i]
        annotation=np.zeros((rows,1))
        for j in range(rows):
            pt_coord=points_in_camera[j,:3]
            lowest_distance=50000 # initialize the distance to ∞
            label_class=""
            #iterate throw all labels : a pointclould can be in the intersection of two or more bounding boxes
            for object in label_list:
                bb_center=[object["x"],object["y"],object["z"]]
                h=object["h"]
                w=object["w"]
                l=object["l"]
                box_coord=centre_to_bbox_coords(bb_center,h,w,l)
                if is_coord_in_bbox(pt_coord,box_coord)==True:
                    # calculate the distance from the pointclooud to the center of the bounding box
                    distance=np.linalg.norm(pt_coord-bb_center)
                    # the bounding box label is assigned to the pointcloud if the distance between its center and this points is the smallest
                    if distance<lowest_distance:
                        lowest_distance=distance
                        label_class=object["label_class"]
            get_annotation(label_class,j,annotation)
        labeled_data=np.hstack((points,annotation))
        destination_directory=os.path.join("D:/labeled_vod",i+'.npy')
        np.save(destination_directory,labeled_data)
        print(i,'done')

            