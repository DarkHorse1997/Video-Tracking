from displacement_vector import convert_to_dataframe
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np
import os
sns.set()

folder = "subject1/surprise"


def plot_displacement_single(index): # This video helps to plot a SINGLE landmark point for all videos in 1 graph
    #WHAT TO DO 
    #print(displacement.shape)
    print("Opening plot_displacment")
    video_file_list=sorted(os.listdir(folder))
    #folder = "subject1/disgust"
    video_list = list(filter(lambda x: not(x.endswith('.avi') or x.endswith('.svg') or x.endswith('.png')), video_file_list))
    print(f"List of videos: {video_list}")
    
    for filename in video_list:

        path = folder + "/" + filename 
        displacement = np.loadtxt(path + "/displacement_vector.out")
        print(f'{path + "/displacement_vector.out"}')
        disp = convert_to_dataframe(displacement)

    #sns.lineplot(data=a)
        ax = sns.lineplot(data = disp, x = 'index', y = f'landmark_{index}', label = f"{filename}")
        #plt.show()
        #plt.legend()
    if not os.path.exists(f'landmark_graphs {folder}'):
               os.makedirs(f'landmark_graphs {folder}')
    plt.savefig(f'landmark_graphs {folder}/landmark_{index}.png', format='png', dpi=300)
    plt.close()
    #plt.show()
    
    
    return ax    


if __name__ == '__main__': 

    
    
    
    for i in range(68):
        plot_displacement_single(i)
    
        #print(path)   
       
        
        
       
         # 
        #plot_displacement_all(disp,filename)
     
    #plt.savefig('anger.svg', format='svg', dpi=1200)
    #plt.show()
   