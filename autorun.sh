#!/bin/bash
#echo "Activating conda environment"
#source activate base

   
cd "subject1"

for s in {anger,disgust,fear,happiness,sadness,surprise}
do
    cd $s
    echo "Entering $s"
    echo "$PWD" | python ../../dlib_land.py
    
    echo | python ../../displacement_vector.py
    echo | python ../../landmark_single.py

    cd ..
    echo "Exited $s"
done
cd ..

#echo "$PWD" | python ../../../dlib_land.py



