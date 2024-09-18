# NOTE: For some reason have to import MjCassieSim first before LibCassieSim otherwise will get seg
# fault error. Seems like doing our own mujoco library loading in libcassiemujoco.so messes things
# if you try to import python mujoco later (`import mujoco`)
import sys

from .mj_cassiesim import MjCassieSim
if sys.platform in ["linux", "linux2"]:
    from .lib_cassiesim import LibCassieSim
