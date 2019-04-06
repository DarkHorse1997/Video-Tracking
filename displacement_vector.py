import numpy as np
from scipy.spatial import distance
import os
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
sns.set()
folder = "subject1/surprise"
#path="dataset/s1_an_1.avi"

def convert_xy_to_points(centroid_x,centroid_y):
    l=[]
    l11=[]
    centroid=[]
    for i in range(centroid_x.shape[0]):
        for j in range(centroid_x.shape[1]):
            l=[centroid_x[i,j],centroid_y[i,j]]
            #print(l)
            l11.append(l)
            #print(l11)
            #print("\n\n")
        centroid.append(l11)
        l11=[]

    cent=np.array(centroid)
    print(f'Dimensions of Landmark Points array : {cent.shape} \n')
    return cent
    #print(cent)


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

def convert_to_dataframe(array): 
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
    print("Successfully Converted \n")
    print(f'Shape of Output Dataframe : {a.shape}')
    return a


def plot_displacement_all(displacement,filename): # This function helps to plot all landmark points for each video in 1 graph
    #print(displacement.shape)
    
    
    plt.figure(figsize = (20,15))
    
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
    plt.savefig(f'{folder}/{filename}/{filename}.png', format='png', dpi=300) #Each image saved in different folder
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
        disp=convert_to_dataframe(displacement)
        #plot_displacement_single() # 
        plot_displacement_all(disp,filename)
     
    #plt.savefig('anger.svg', format='svg', dpi=1200)
    #plt.show()
   



    