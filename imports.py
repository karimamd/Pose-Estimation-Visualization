import pandas as pd
from matplotlib import pyplot as plt
import cv2
from pcloud import PyCloud
import urllib
import hashlib
from fs import opener

import torch
from torchvision import models
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary

import os, time, sys, concurrent.futures
import datetime
from IPython.display import clear_output
import zipfile
import pandas as pd
import numpy as np
import math
import cv2
from tqdm import trange
from tqdm import tqdm

from scipy.ndimage import convolve
import scipy.misc