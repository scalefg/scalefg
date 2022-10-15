import os
from os import path
import sys
test_src = path.abspath(path.join(path.dirname(__file__)))
src_dir = path.abspath(path.join(test_src, os.pardir, os.pardir))
sys.path.append(src_dir)
