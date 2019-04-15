''' This Code Snippet was used in plot_landmark1() of real_time.py . It was used to create a panel but there is some bug in the code which makes the landmark point values for all frames the same '''

#https://stackoverflow.com/questions/28368598/dataframe-of-dataframes-with-pandas
dic = dict()
    for i in range(p1.shape[0]):

        d = pd.DataFrame(data=p1[0],columns=['x', 'y'])
        key = f'Frame_{i}'
        #print(literal)
        dic[key] = d
        
    p11 = pd.concat(dic)

    print(p11.shape)
    panel2 = pd.Panel(dic)
    pnl2 = panel2.to_frame()
    pn = panel2.to_xarray()
    #print(p11['y']) # Indexing by columns,need by rows


'''This code snippet is used to traverse directories and return only those directories which contain video(avi) files'''

for dirName, subdirList, fileList in os.walk(folder):
        #print('Found directory: %s' % dirName)
        if not fileList:
            continue
        else:

            if('avi' in fileList[0]):
                print(dirName)


'''This code snippet is used to get list of video files in each of the folders '''

import os 
import glob
l = glob.glob('subject *')


for i in l:
   os.chdir(i)
   print(os.getcwd())
   s = ['anger','disgust','fear','happiness','sadness','surprise']
   for j in s:
       os.chdir(j)
       print(os.getcwd())
       video = os.listdir('.') #files = [f for f in os.listdir('.') if os.path.isfile(f)] OR files = glob.glob(".avi")
       print(video)
       os.chdir("..")
   os.chdir("..")





