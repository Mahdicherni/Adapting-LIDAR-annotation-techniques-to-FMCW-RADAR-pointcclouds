import json
import os
import os.path

def get_labels(source_directory):
        """
This method obtains the label information from the location specified by the KittiLocations object. If the file
does not exist, it returns None.

        :return: List of strings with label data.
        """
        try:
            final_list={}
            file_list = os.listdir(source_directory)
            for i in file_list:
                label_file = os.path.join(source_directory,i)
                with open(label_file, 'r') as text:
                    labels = text.readlines()
                text.close()
                labels_list=[]
                labels_list=get_labels_dict(labels)
                final_list[os.path.splitext(i)[0]]=labels_list
                print(i,'done')
            file_path='D:/labels_as_json/labels.json'
            with open(file_path, 'w') as f:
                json.dump(final_list, f)
        except FileNotFoundError:
            return None
        # Save the dictionary of lists as JSON



def get_labels_dict(list_labels):
        """
This method returns a list of dictionaries containing the label data.
        :return: List of dictionaries containing label data.
        """

        labels = []  # List to be filled
    
        for line in list_labels:  # Go line by line to split the keys
            act_line = line.split()
            label, _, _, _, _, _, _, _, h, w, l, x, y, z, rot, score = act_line
            h, w, l, x, y, z, rot, score = map(float, [h, w, l, x, y, z, rot, score])

            labels.append({'label_class': label,
                           'h': h,
                           'w': w,
                           'l': l,
                           'x': x,
                           'y': y,
                           'z': z,
                           'rotation': rot,
                           'score': score}
                          )
        return labels
# u must change 
get_labels("D:/P2M/view_of_delft_PUBLIC/lidar/training/label_2")
