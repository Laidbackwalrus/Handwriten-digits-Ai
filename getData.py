import gzip
import sys
import struct 
import binascii

def display_images(images):
    for array in images:
        for row in array:
            for item in row:
                print(''.join(['{:3}'.format(item)]), end="")
            print("")

def get_data(x):
    with gzip.open("f/train-labels-idx1-ubyte.gz", "rb") as f:
        f.seek(8)
        labels = []
        for i in range(x):
            label = int.from_bytes(f.read(1), sys.byteorder)
            labels.append(label)

    with gzip.open("f/train-images-idx3-ubyte.gz", "rb") as f:
        f.seek(16)
        images = []
        for i in range(x):
            image = []
            
            for j in range (0,28):
                temp_mat = []
                for k in range (0,28):
                    temp_mat.append(int.from_bytes(f.read(1), sys.byteorder))
                image.append(temp_mat)
            images.append(image) 

    return labels, images

labels, images = get_data(40)

display_images(images)