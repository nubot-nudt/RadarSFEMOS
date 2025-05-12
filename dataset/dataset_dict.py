from .nuscenes import *
from .carla import *
from .saic import *
from .vod import *
from .vod_clip import vodClipDataset

dataset_dict = {
 'carlaDataset': carlaDataset,
 'saicDataset': saicDataset,
 'vodDataset': vodDataset,
 'vodClipDataset': vodClipDataset,

}