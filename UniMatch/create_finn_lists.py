import os
# assign directory
img_directory = 'data/FinnForest/rgb/train'
directory = 'data/FinnForest/rgb/train/'

# iterate over files in
# that directory

count = 0
limit = 63

with open('labeled.txt', 'w') as file:
    with open('unlabeled.txt', 'w') as unfile:
        for filename in os.listdir(directory):
            f = os.path.join(directory, filename)
            # checking if it is a file
            if os.path.isfile(f) and count < limit:
                idx = f.split("/")[4]
                idx = idx.split(".")[0]
                file.write("./data/FinnForest/rgb/train/" + str(idx) + ".jpg")
                file.write('\n')
                count = count + 1
            elif os.path.isfile(f) and count >= limit:
                idx = f.split("/")[4]
                unfile.write("./data/FinnForest/rgb/train/" + str(idx))
                unfile.write('\n')

#directory = 'data/FinnForest/annotations/val/semantic'

#with open('val.txt', 'w') as file:
#    with open('test_finn.txt', 'w') as testfile:
#        for filename in os.listdir(directory):
#            f = os.path.join(directory, filename)
#            # checking if it is a file
#            if os.path.isfile(f) and count < limit:
#                print(f)
#                idx = f.split("/")[5]
#                idx = idx.split(".")[0]
#                file.write("./data/FinnForest/rgb/val/" + str(idx) + ".jpg")
#                file.write('\n')
#                count = count + 1
#            elif os.path.isfile(f) and count >= limit:
#                idx = f.split("/")[5]
#                idx = idx.split(".")[0]
#                testfile.write("./data/FinnForest/rgb/val/" + str(idx) + ".jpg")
#                testfile.write('\n')
