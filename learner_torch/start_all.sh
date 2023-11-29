#!/bin/bash
# sshpass ssh root@172.15.15.3 "bash /home/zhaoyp/guandan_tog/actor_n/start.sh"
# nohup python -u learner.py > ./learner_out.log 2>&1 &
#sshpass ssh root@172.15.15.3 "bash /home/zhaoyp/guandan_tog/actor_torch/start.sh"
#sleep 0.1s
for i in {3..13}
do
        sshpass ssh root@172.15.15.$i "bash /home/zhaoyp/guandan_tog/actor_torch/start.sh"
        echo $i
        sleep 0.1s
done

nohup python -u learner.py > ./learner_out.log 2>&1 &

for i in {14..43}
do
        sshpass ssh root@172.15.15.$i "bash /home/zhaoyp/guandan_tog/actor_torch/start.sh"
        echo $i
        #sleep 3s
done
