import os
import random
# assign directory
img_directory = 'data/FinnForest/rgb/train'
directory = 'data/sam_segs/snoge_frames2/'

# iterate over files in
# that directory

limit = 150
count = 0

# with open('unlabeled.txt', 'w') as file:
#    with open('approved.txt') as topo_file:
#        for line in topo_file:
#            if count < limit:
#                break
#            file.write("./data/FinnForest/forestseg_katam/" + str(line.strip()) + ".png")
#            file.write('\n')
#           count = count + 1

with open('test.txt', 'w') as file:
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f): #and count < limit:
            idx = f.split("/")[3]
            file.write("./data/sam_segs/snoge_frames2/" + str(idx))
            file.write('\n')
            #count = count + 1

#output_file = 'test2.txt'

#files_list = [f for f in os.listdir(directory) if os.path.isfile(os.path.join(directory, f))]
#selected_files = random.sample(files_list, min(limit, len(files_list)))

#with open(output_file, 'w') as file:
#    for filename in selected_files:
#        file.write(f"./data/trees/train/{filename}\n")
