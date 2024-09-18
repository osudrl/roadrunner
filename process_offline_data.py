import glob
import pickle
import numpy as np

# fill in this name
NAME_CONFIG = 'base'
TERRAIN_DIFF = 'stair-hard'

path = "offline_data/CassieHfield/stair_fix_bootstrap_oldlstm/0c953f/stair-diff-1/20230908-142940/*.pkl"
files = glob.glob(path)
cnt = 0
done = 0
both = 0
force = []
side = []
vel = []
for f in files:
    fn = open(f, "rb")
    try:
        data = pickle.load(fn)
        vel.append(data['cmd'][0][0])
        if any(data['scuff']):
            cnt += 1
            force.append(np.max(data['touch'][-1]))
        if any(data['done']):
            done += 1
        if any(data['done']) and any(data['scuff']):
            both += 1
    except:
        pass
    fn.close()
print("total eps", len(files), " eps with scuff, ", cnt, " failure eps, ", done, " eps with scuff and failures ,", both)
print(f"success rate {1-done/len(files)}, scuff rate {cnt/len(files)}, both rate {both/len(files)}, scuff in done rate {both/done}")

print("copy the following to the plot script")
print({'config': NAME_CONFIG, 'terrain': TERRAIN_DIFF, 'success': {1-done/len(files)}, 'scuff': {cnt/len(files)}, 'both': {both/len(files)}, 'scuff-done': {both/done}})