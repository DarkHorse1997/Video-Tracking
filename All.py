import cv2
import dlib
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.spatial import distance
from sklearn.cluster import KMeans

print (folder)
sns.set()
path_of_model="/home/tanmoydas1997/Documents/Video-Tracking/shape_predictor_68_face_landmarks.dat"
#from imutils import face_utils
folder = input()

print(folder)

index = folder.find("g/")
folder = folder[index+2:]
#path_of_video="dataset/s1_an_1.avi"


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
    panel = pd.Panel(
                    p1,
                    items = ['Frame {}' .format(i) for i in range(0,p1.shape[0])],
                    major_axis = ['Landmark {}' .format(i) for i in range(0,68) ],
                    minor_axis = ['x','y']
                    )
    pnl = panel.to_frame()
    print(p1)
    print(pnl)
    pnl.to_csv(path1[:path1.find(".")]+'/landmark_points_dataframe.csv')
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

def elbow_curve(points):
    Nc = range(1, 20)

    kmeas = [KMeans(n_clusters=i) for i in Nc]
    Y = points
    score = [kmeas[i].fit(Y).inertia_ for i in range(len(kmeas))]
    plt.plot(Nc,score)
    plt.xlabel('Number of Clusters')
    plt.ylabel('Score')
    plt.title('Elbow Curve')
    plt.show()

def apply_kmeans(points,path):
    print(f"Applying k-means to {path[path.find('/')+1:]}")
    print(points.shape)
    
    no_of_clusters=3
    #X = np.array([[1, 2], [2, 3]], [1, 4, 2], [1, 0, 2], [4, 2, 3], [4, 4, 8], [4, 0, 6]])
    new_points = np.reshape(points,(points.shape[0],68*2))
    
    kmeans = KMeans(n_clusters=no_of_clusters, random_state=0).fit(new_points)

    centroid = kmeans.cluster_centers_
    centroid = np.reshape(centroid,(no_of_clusters,68,2))
    
    np.savetxt(path[:path.find(".")]+'/centroid.out', centroid)
    
def find_minframe(points,centroid,no_of_clusters):    
    #z = scipy.spatial.distance.cdist(A,B,'chebyshev')
    minframe=[]
    #print(z)
    for i in range(no_of_clusters):

        dist=[]
        for j in range(50):
            y = distance.cdist(points[j],centroid[i],'euclidean')#cent
            #print(y)
            #print(np.sum(np.diag(y)))
            dist.append(np.trace(y))
            #print(str(i)+"  "+str(np.trace(y)))
        dd=np.array(dist)
        minframe.append(np.argmin(dd))
    return minframe
# NOTE:The following function does not work unless all frames are extracted and stored in frames/**
# NOTE:This can e fixed in FrameExtract
# NOTE:This function has a lot of issues
def list_of_key_frames():
    list_of_images=[]
    for i in minframe:
        path_of_image="../input/frames/frame"+str(i)+".jpg"
        im=plt.imread(path_of_image)
        list_of_images.append(im)
        implot = plt.imshow(im)
        plt.show()
    return list_of_images



def grid_display(list_of_images, list_of_titles=[], no_of_columns=2, figsize=(10,10)):

    fig = plt.figure(figsize=figsize)
    column = 0
    for i in range(len(list_of_images)):
        column += 1
        #  check for end of column and create a new figure
        if column == no_of_columns+1:
            fig = plt.figure(figsize=figsize)
            column = 1
        fig.add_subplot(1, no_of_columns, column)
        plt.imshow(list_of_images[i])
        plt.axis('off')
        if len(list_of_titles) >= len(list_of_images):
            plt.title(list_of_titles[i])

    #grid_display(list_of_images, list_of_titles=[], no_of_columns=3, figsize=(10,10))


    #See helper_snippets Snippet #2
    
# Driver Code 
if __name__ == '__main__': 
    
    print(folder)

    video_file_list=os.listdir(folder)

    video_list = list(filter(lambda x: x.endswith('.avi'), video_file_list))
    print(f"List of videos: {video_list}")
        
    for filename in video_list:
        path_of_video= folder + "/" + filename    
        point=FrameExtract(path_of_video,path_of_model)
        apply_kmeans(point,path_of_video)

    
    os.system('spd-say "your program has finished"')






#path="dataset/s1_an_1.avi"

#z = scipy.spatial.distance.cdist(A,B,'chebyshev')
def find_distance(cent,no_of_clusters,distance_type,path):
    print(f"Finding {distance_type} distance between key points for {path[path.find('/')+1:]} ")
    
    distance_vector = []  
    #print(cent[1].shape)
    for i in range(0,no_of_clusters-1,1):
        print(f"Calculating distance matrix between key-frame {i} and key-frame {i+1}")
        #print(cent[i])
        y = distance.cdist(cent[i],cent[i+1],distance_type)
        #print(y)
        #print(y.shape)
        distance_vector.append(np.diag(y))
        #print(np.diag(y))
        
    #print(np.array(distance_vector))
    distance_vector = np.array(distance_vector)
    
    np.savetxt(path + "/displacement_vector.out",distance_vector)
    print(f"Displacement Vector saved to file : {path + '/displacement_vector.out'}")
    return distance_vector

def convert_to_dataframe(array,path): 
    print(f'Shape of Input array {array.shape}')
    #print(array[0].shape)
    print("Converting Numpy array to Pandas Dataframe \n ")
    #print(array.shape[0])  #Number of key frames 5
    #print(array[0].shape[0])  #Number of landmark points 68
    
    t1=[]
    for i in range(array[0].shape[0]):
        t1.append(f'landmark_{i}')
    
    t2=[]

    for i in range(array.shape[0]):
        t2.append(f'frame {i} - {i+1}') 
           
    a = pd.DataFrame(data=array,index=t2,columns=t1)
    print(f'Shape of Output Dataframe : {a.shape}')
    a = a.reset_index()
    a.to_csv(path + "/displacement_vector.csv")
    print("Successfully Converted \n")
    print(f'Shape of Output Dataframe : {a.shape}')
    return a


def plot_displacement_all(displacement,filename): # This function helps to plot all landmark points for each video in 1 graph
    #print(displacement.shape)   
    
    #plt.figure(figsize = (20,15))
    
    #sns.lineplot(data=a)
    for i in range(68):

        #ax = sns.lineplot(data = displacement, x = 'index', y = f'landmark_{i}',legend = "full" ,label = f"landmark {i}") to get legend (which doesn't fit right now)
        ax = sns.lineplot(data = displacement, x = 'index', y = f'landmark_{i}',legend = "full" )

    sns.set_context("talk")
    ax.set_title(f'{filename}')
    ax.set_ylabel("Displacement Value")
    ax.set_xlabel("Frames")
    
    #plt.legend(loc='upper center')
    #print(f'{folder}/{filename}/{filename}.png')
    #plt.savefig(f'{folder}/{filename}.png', format='png', dpi=300) All images are saved together
    plt.savefig(f'{folder}/{filename}/{filename}.png', format='png', dpi=200) #Each image saved in different folder
    #plt.show()
    #
    #for i in range(68):

        #y=displacement[:, 0]
        #a.plot(y=f'landmark_{i}',ax=ax)
    #return ax    
def plot_displacement_single(): # This video helps to plot a SINGLE landmark point for all videos in 1 graph
    #WHAT TO DO 
    #print(displacement.shape)
    print("Opening plot_displacment")
    video_file_list=os.listdir(folder)
    #folder = "subject1/disgust"
    video_list = list(filter(lambda x: not(x.endswith('.avi') or x.endswith('.svg') or x.endswith('.png')), video_file_list))
    print(f"List of videos: {video_list}")
    
    for filename in video_list:

        path = folder + "/" + filename 
        displacement = np.loadtxt(path + "/displacement_vector.out")
        print(f'{path + "/displacement_vector.out"}')
        disp = convert_to_dataframe(displacement)

    #sns.lineplot(data=a)
        ax = sns.lineplot(data = disp, x = 'index', y = 'landmark_0', label = f"{filename}")
        #plt.show()
        #plt.legend()
    plt.show()
    #ax=plt.gca()
    #for i in range(68):

        #y=displacement[:, 0]
        #a.plot(y=f'landmark_{i}',ax=ax)
    
    return ax    


if __name__ == '__main__': 
    
    video_file_list=os.listdir(folder)
    #folder = "subject1/disgust"
    video_list = list(filter(lambda x: not(x.endswith('.avi') or x.endswith('.svg') or x.endswith('.png')), video_file_list))
    print(f"List of videos: {video_list}")
    for filename in video_list:
        path = folder + "/" + filename 
        #print(path)   
        centroid_x = np.loadtxt(path + '/centroid_x.out')
        centroid_y = np.loadtxt(path + '/centroid_y.out')
        cent = convert_xy_to_points(centroid_x,centroid_y)
        p1 =  np.load(path + '/landmark_points_array.out.npy')
        displacement=find_distance(cent,cent.shape[0],'euclidean',path)
        disp=convert_to_dataframe(displacement,path)
        #plot_displacement_single() # 
        plot_displacement_all(disp,filename)
     
    #plt.savefig('anger.svg', format='svg', dpi=1200)
    #plt.show()
   



    
