from sim.digit_sim import MjDigitSim
import numpy as np
import time
import mujoco as mj

class DigitFeetEstimator:
    def __init__(self):
        self.sim = MjDigitSim()

    def update_qpos(self, qpos):
        self.sim.set_qpos(qpos)

    def get_feet_position(self):
        return self.sim.get_feet_position_in_base()

if __name__ == "__main__":
    est = DigitFeetEstimator()
    # est.sim.reset()
    est.sim.hold()
    qpos = est.sim.reset_qpos
    qpos[0:7] = np.array([0, 0, 3, 0, 0, 0, 1])
    est.update_qpos(qpos)
    t0 = time.time()
    a = est.get_feet_position()
    print(time.time() - t0)
    jacp = np.zeros((3,self.sim.model.nv))
    site_id = mj.mj_name2id(self.sim.model, mj.mjtObj.mjOBJ_SITE, self.sim.feet_site_name[0])
    mj.mj_jacSite(self.sim.model, self.sim.data, jacp, None, site_id)
    # jacp[:,0:6] = 0.0
    # print(jacp)
    fpos = jacp@self.sim.data.qvel
    print(fpos)