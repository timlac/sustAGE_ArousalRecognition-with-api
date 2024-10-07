import os
import module_arousal as arousal

video_dir = '/home/tim/Work/nexa/nexa-emotion-recognition-pipelines/data/videos/sentimotion/'

# List all files in the directory
files = os.listdir(video_dir)

# Iterate over each file and apply the arousal API
for file in files:
    file_path = os.path.join(video_dir, file)
    if os.path.isfile(file_path):
        ret = arousal.API(file_path)
        print(f'File: {file}, Arousal: {ret}')