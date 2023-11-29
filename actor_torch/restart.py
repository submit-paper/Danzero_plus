import os
import psutil
import time 

def find_procs_by_name(name):
    "Return a list of processes matching 'name'."
    ls = []
    for p in psutil.process_iter(["name", "exe", "cmdline"]):
        if name == p.info['name'] or \
                p.info['exe'] and os.path.basename(p.info['exe']) == name or \
                p.info['cmdline'] and p.info['cmdline'][0] == name:
            ls.append(p)
    return ls

res = find_procs_by_name('/root/miniconda3/envs/guandan/bin/python')
print(res)
print(len(res)) 

while True:
    time.sleep(120)
    res = find_procs_by_name('/root/miniconda3/envs/guandan/bin/python')
    # res = find_procs_by_name('python')
    if len(res) < 10:
        print('restart actor')
        os.system("bash /home/zhaoyp/guandan_tog/actor_torch/restart.sh")
        time.sleep(300)
