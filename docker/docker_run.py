#!/usr/bin/env python
import os

if __name__=="__main__":
    cmd = "nvidia-docker run -it -v %s:/host hapticnet" % (os.path.join(os.getcwd(), '..'))
    code = os.system(cmd)
