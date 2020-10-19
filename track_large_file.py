
import os

for dirPath, dirNames, fileNames in os.walk("./"):
    for f in fileNames:
        file_path = os.path.join(dirPath, f)
        b = os.path.getsize(file_path)
        if b / 1024 / 1024 > 50:
            #print(file_path, b)
            print(file_path)
            #os.system('git lfs track {}'.format(file_path))
