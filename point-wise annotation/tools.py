import numpy as np
import os 
import os.path
import logging
import tensorflow as tf 



def get_sensor_transforms(sensor: str):  # -> Optional[(np.ndarray, np.ndarray)]:
        """
This method returns the corresponding intrinsic and extrinsic transformation from the dataset.
        :param sensor: Sensor name in string for which the transforms to be read from the dataset.
        :return: A numpy array tuple of the intrinsic, extrinsic transform matrix.
        """
        if (sensor == 'radar'):
            try:
                calib_dir="D:/P2M/view_of_delft_PUBLIC/radar/training/calib"
                file_list = os.listdir(calib_dir)
                calibration_file = os.path.join(calib_dir,file_list[0])
            except FileNotFoundError:
                logging.error("txt does not exist at")
                return None, None
        elif (sensor=='lidar'):
            try:
                calib_dir="D:/P2M/view_of_delft_PUBLIC/lidar/training/calib"
                file_list = os.listdir(calib_dir)
                calibration_file = os.path.join(calib_dir,file_list[0])
            except FileNotFoundError:
                logging.error("txt does not exist at")
                return None, None    
        else:
            raise AttributeError('Not valid sensor')

        with open(calibration_file, "r") as f:
            lines = f.readlines()
            intrinsic = np.array(lines[2].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)  # Intrinsics
            extrinsic = np.array(lines[5].strip().split(' ')[1:], dtype=np.float32).reshape(3, 4)  # Extrinsic
            extrinsic = np.concatenate([extrinsic, [[0, 0, 0, 1]]], axis=0)

        return intrinsic, extrinsic
#intrensic
def camera_projection_matrix():
        """
Property which gets the camera projection matrix.
        :return: Numpy array of the camera projection matrix.
        """
        camera_projection_matrix,T_camera_lidar = get_sensor_transforms('lidar')
        return camera_projection_matrix

def t_camera_lidar():
        """
Property which returns the homogeneous transform matrix from the lidar frame, to the camera frame.
        :return: Numpy array of the homogeneous transform matrix from the lidar frame, to the camera frame.
        """
        camera_projection_matrix,T_camera_lidar = get_sensor_transforms('lidar')
        return T_camera_lidar

def t_camera_radar():
        """
Property which returns the homogeneous transform matrix from the radar frame, to the camera frame.
        :return: Numpy array of the homogeneous transform matrix from the radar frame, to the camera frame.
        """
        camera_projection,T_camera_radar =get_sensor_transforms('radar')
        return T_camera_radar

def t_lidar_camera(t_camera_lidar):
        """
Property which returns the homogeneous transform matrix from the camera frame, to the lidar frame.
        :return: Numpy array of the homogeneous transform matrix from the camera frame, to the lidar frame.
        """
        T_lidar_camera = np.linalg.inv(t_camera_lidar)
        return T_lidar_camera
        
def t_radar_camera(t_camera_radar):
        """
Property which returns the homogeneous transform matrix from the camera frame, to the radar frame.
        :return: Numpy array of the homogeneous transform matrix from the camera frame, to the radar frame.
        """
        T_radar_camera = np.linalg.inv(t_camera_radar)
        return T_radar_camera


def t_lidar_radar(t_lidar_camera,t_camera_radar):
        """
Property which returns the homogeneous transform matrix from the radar frame, to the lidar frame.
        :return: Numpy array of the homogeneous transform matrix from the radar frame, to the lidar frame.
        """
        T_lidar_radar = np.dot(t_lidar_camera,t_camera_radar)
        return T_lidar_radar

def t_radar_lidar(t_radar_camera,t_camera_lidar):
        """
Property which returns the homogeneous transform matrix from the lidar frame, to the radar frame.
        :return: Numpy array of the homogeneous transform matrix from the lidar frame, to the radar frame.
        """
        T_radar_lidar = np.dot(t_radar_camera,t_camera_lidar)
        return T_radar_lidar

def homogeneous_transformation(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    """
This function applies the homogenous transform using the dot product.
    :param points: Points to be transformed in a Nx4 numpy array.
    :param transform: 4x4 transformation matrix in a numpy array.
    :return: Transformed points of shape Nx4 in a numpy array.
    """
    if transform.shape != (4, 4):
        raise ValueError(f"{transform.shape} must be 4x4!")
    if points.shape[1] != 4:
        raise ValueError(f"{points.shape[1]} must be Nx4!")
    # To accelerate calculation , we used the GPU
        gpus = tf.config.experimental.list_physical_devices('GPU')

        if gpus:
        # Set TensorFlow to use the first GPU
                with tf.device(gpus[0]):
                        point_tensor=tf.Variable(points)
                        points_transposed = tf.transpose(points)
                        transformed_points = tf.matmul(transform, points_transposed)
                        transformed_points = tf.transpose(transformed_points)
                        transformed_points_numpy=transformed_points.numpy()
                return transformed_points_numpy

def transform_pcl(points: np.ndarray, transform_matrix: np.ndarray):
    """
This function transforms homogenous points using a transformation matrix.
    :param points: Points to be transformed.
    :param transform_matrix: Homogenous transformation matrix.
    :return: Transformed homogenous points.
    """
    point_homo = np.hstack((points[:, :3],
                            np.ones((points.shape[0], 1),
                                    dtype=np.float32)))

    points_new_frame = homogeneous_transformation(point_homo, transform=transform_matrix)

    return points_new_frame

def  centre_to_bbox_coords(centre, width, height,length):
    # this method returns the bounding box coordinates 
    return [[centre[0] - width/2, centre[1] - height/2,centre[2]-length/2],
            [centre[0] + width/2, centre[1] + height/2,centre[2]-length/2]] 

def is_coord_in_bbox(coord, bbox):
    # return True the point cloud is in the bounding box of the detected object in the frame
    #print(coord)
    #print(bbox)
    bbox_array = np.array(bbox)
    x_valid = coord[0] >= bbox_array[0,0]-0.5 and coord[0] < bbox_array[1,0]+0.5
    y_valid = coord[1] >= bbox_array[0,1]-0.5 and coord[1] < bbox_array[1,1]+0.5
    z_valid = coord[2] >= bbox_array[0,2]-0.5 and coord[2] < bbox_array[1,2]+0.5

    return x_valid and y_valid and z_valid

def get_annotation(label_class:str,j:int,annotation:np.ndarray):
        if label_class.lower()=="car":
                annotation[j,0]=1
        elif label_class.lower()=="pedestrian":
                annotation[j,0]=2
        elif label_class.lower()=="cyclist":
                annotation[j,0]=3
        elif label_class.lower()=="rider":
                annotation[j,0]=4
        elif label_class.lower()== "bicycle_rack":
                annotation[j,0]=6
        elif label_class.lower()=="human_depiction ":
                annotation[j,0]=7
        elif label_class.lower()=="moped_scooter":
                annotation[j,0]=8
        elif label_class.lower()=="motor":
                annotation[j,0]=9
        elif label_class.lower()=="truck":
                annotation[j,0]=10
        elif label_class.lower()=="ride_other":
                annotation[j,0]=11
        elif label_class.lower()=="vehicle_other":
                annotation[j,0]=12
        elif label_class.lower()=="ride_uncertain":
                annotation[j,0]=13
  
         
            

     
     

'''def filter_files(label_directory:str,npy_directory:str, frame_directory:str):
     l1=os.listdir(label_directory)
     l2=os.listdir(npy_directory)
     l3=os.listdir(frame_directory)
     l=[]
     for i in l1:
          a=os.path.splitext(i)[0]
          npy=a+'npy'
          jpg=a+'.jpg'
          if (npy in l2) and (jpg in l3):
               l.append(i)
     return l'''
