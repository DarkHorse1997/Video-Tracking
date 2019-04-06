''' This Code Snippet was used in plot_landmark1() of real_time.py . It was used to create a panel but there is some bug in the code which makes the landmark point values for all frames the same '''


dic = dict()
    for i in range(p1.shape[0]):

        d = pd.DataFrame(data=p1[0],columns=['x', 'y'])
        key = f'Frame_{i}'
        #print(literal)
        dic[key] = d
        
    p11 = pd.concat(dic)

    print(p11.shape)