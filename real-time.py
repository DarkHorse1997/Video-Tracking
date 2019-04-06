import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np
import pandas as pd
import os
sns.set()
from displacement_vector import convert_to_dataframe

folder = "subject1/surprise"

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

#https://stackoverflow.com/questions/48482256/what-is-the-pandas-panel-deprecation-warning-actually-recommending
def plot_landmark(p1):
    print(p1.shape) #(#Frames,#Landmarks,#Coordinates)
    
    panel = pd.Panel(
                    p1,
                    items = ['Frame {}' .format(i) for i in range(0,p1.shape[0])],
                    major_axis = ['Landmark {}' .format(i) for i in range(0,68) ],
                    minor_axis = ['x','y']
                    )
    pnl = panel.to_frame()
    print(pnl)
    dat = pnl.xs('Landmark 0')
    
    ax = sns.pointplot(data = dat, x = dat.xs('x'), y = dat.xs('y'))
    plt.show()
    ax = sns.lineplot(data = dat, x = dat.xs('x'), y = dat.xs('y'))
    plt.show()
    
    print(pnl.xs('Landmark 0'))
     # Print by Landmarks
    #print(pnl['Frame 0']) # Print by Frames
    
    

    print("\n \n")
    

    
    
def plot_landmark1(frame,frame_no=-1):
    
    for i in range(len(frame)):
        x=frame[i][0]
        y=frame[i][1] #***Figure out why flipped in y-axis necessitating this -(minus) sign***
        #plt.text(x,y,i,ha="center", va="center",fontsize=8)
        plt.scatter(x,y,c='r', s=10)
    ax=plt.gca()
    ax.invert_yaxis()    
    plt.show()

if __name__ == '__main__': 
    
    video_file_list=os.listdir(folder)
    #folder = "subject1/disgust"
    video_list = list(filter(lambda x: not(x.endswith('.avi') or x.endswith('.svg') or x.endswith('.png')), video_file_list))
    print(f"List of videos: {video_list}")
    for filename in video_list:
        path = folder + "/" + filename 
        #print(path)   
        #centroid_x = np.loadtxt(path + '/centroid_x.out')
        #centroid_y = np.loadtxt(path + '/centroid_y.out')
        #cent = convert_xy_to_points(centroid_x,centroid_y)
        p1 =  np.load(path + '/landmark_points_array.out.npy')
        plot_landmark(p1)
