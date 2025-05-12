import os
from glob import glob
import moviepy.video.io.ImageSequenceClip
image_files = sorted(glob("/home/fangqiang/RadarFlow/checkpoints/raflow/test_vis_2d/*.png"),key=lambda x:eval(x.split("/")[-1].split(".")[0]))
fps=10
duration = 300

frame = fps * duration

if len(image_files) > frame:
    image_files = image_files[:frame]

clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
clip.write_videofile('/home/fangqiang/RadarFlow/checkpoints/raflow/test_vis_2d.mp4')