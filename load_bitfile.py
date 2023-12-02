#!/usr/bin/python3
from pynq import Overlay
from pynq import Xlnk
import numpy as np
import time
xlnk=Xlnk()

ol=Overlay("design_1_wrapper.bit")
for i in ol.ip_dict:
    print(i);
ol.download();
