'''
===================================================
              Obtain Landmark Points 
===================================================
Author: Tanmoy Das
Email : tanmoydas1997@gmail.com


FrameExtract(path1,path2)
    This functions takes in a video and returns a 3D numpy array containing all the landmark points of all the frames of that video. It also saves this Landmark Points in a file for later use.

Parameters:	
path1 : string
    The file path to the video input file 
path2 : string
    The file path to the dlib model
Returns:	
p1 : Numpy Array
    Array of Landmark Points

apply_kmeans(points,path)
    This function takes in a numpy array of Landmark Points for a single video and applies K-means Clustering to it to get key frames. It saves the centroid coordinates in a file.

Parameters:	
points: Numpy Array 
    The Numpy Array of Landmark Points of all frames of a video. 
path : string
    The file path to the dlib model
Returns:	


'''

import cv2
import dlib
import os
import numpy as np
import matplotlib.pyplot as plt
#from imutils import face_utils
folder = "subject1/surprise" #
#path_of_video="dataset/s1_an_1.avi"
path_of_model="shape_predictor_68_face_landmarks.dat"

def FrameExtract(path1,path2):
    print(f"Working with video file {path1[path1.find('/')+1:]} ")
    video_obj = cv2.VideoCapture(path1) #Video object
    detector = dlib.get_frontal_face_detector() #Face detector
    predictor = dlib.shape_predictor(path2) #Landmark identifier. Set the filename to whatever you named the downloaded file
    count = 0
    ret = 1
    p=[]
    
    while ret:                        #Runs ONCE for each FRAME
        ret, frame = video_obj.read()
        if(ret == 0):
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) #Convert image to grayscale
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_image = clahe.apply(gray)
        detections = detector(clahe_image, 1)
         #Detect the faces in the image
        for k,d in enumerate(detections):
            #print("enter loop") #For each detected face
            shape = predictor(clahe_image, d)
            #shape1 = face_utils.shape_to_np(shape) This works by utilizing imutis library
            #print(shape1)
                #print(shape.part(i).y)#Get coordinates
            #print(shape)
        vec = np.empty([68, 2], dtype = int)
        for b in range(68):
            vec[b][0] = shape.part(b).x
            vec[b][1] = shape.part(b).y
        #print(vec.shape)
        v = vec.tolist()
        #print(type(v))
        #print(type(p))
        p.append(v)

        #
        #print(shape1-vec)
        #cv2.imshow("image", frame) #Display the frame
        #print("enter after loop")
        if not os.path.exists(path1[:path1.find(".")]):
               os.makedirs(path1[:path1.find(".")])
        #print(count)
        #cv2.imwrite(path1[:path1.find(".")]+"/frame%d.jpg" % count, imag) 
        
        count += 1
    p1=np.array(p)
    print(p1.shape)
    np.save(path1[:path1.find(".")]+'/landmark_points_array.out', p1)
    #np.savetxt(path1[:path1.find(".")]+"/reshaped.txt", p1.reshape((3,-1)), fmt="%s", header=str(p1.shape))
    return p1



def plot_landmark(frame,frame_no=-1):
    import matplotlib.pyplot as plt
    for i in range(len(frame)):
        x=frame[i][0]
        y=frame[i][1] #***Figure out why flipped in y-axis necessitating this -(minus) sign***
        #plt.text(x,y,i,ha="center", va="center",fontsize=8)
        plt.scatter(x,y,c='r', s=10)
    ax=plt.gca()
    ax.invert_yaxis()
    plt.show()


def plot_landmark_on_frame(frame, path_of_image):
    import matplotlib.pyplot as plt
    im = plt.imread(path_of_image)
    implot = plt.imshow(im)
    plt.show()

    for i in range(len(frame)):
        x = frame[i][0]
        y = frame[i][1]
        plt.text(x, y, i, ha="center", va="center", fontsize=8)
        plt.scatter(x, y, c='r', s=40)
    plt.show()




def apply_kmeans(points,path):
    print(f"Applying k-means to {path[path.find('/')+1:]}")
    print(points.shape)
    from sklearn.cluster import KMeans


    no_of_clusters=6
    #X = np.array([[1, 2], [2, 3]], [1, 4, 2], [1, 0, 2], [4, 2, 3], [4, 4, 8], [4, 0, 6]])

    kmeans_x = KMeans(n_clusters=no_of_clusters, random_state=0).fit(points[:,:,0])
    kmeans_y = KMeans(n_clusters=no_of_clusters, random_state=0).fit(points[:,:,1])

    centroid_x = kmeans_x.cluster_centers_
    centroid_y = kmeans_y.cluster_centers_
    
    np.savetxt(path[:path.find(".")]+'/centroid_x.out', centroid_x)
    np.savetxt(path[:path.find(".")]+'/centroid_y.out', centroid_y)
    
    #See helper_snippets Snippet #2
    
# Driver Code 
if __name__ == '__main__': 
    
    video_file_list=os.listdir(folder)

    video_list = list(filter(lambda x: x.endswith('.avi'), video_file_list))
    print(f"List of videos: {video_list}")
    for filename in video_list:
        path_of_video= folder + "/" + filename    
        point=FrameExtract(path_of_video,path_of_model)
        apply_kmeans(point,path_of_video)




