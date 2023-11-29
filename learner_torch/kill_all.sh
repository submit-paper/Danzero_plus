#!/bin/bash
for i in {3..43}
do
	sshpass ssh root@172.15.15.$i "bash /home/zhaoyp/guandan_tog/actor_torch/kill.sh"
done

ps aux|grep python|grep -v grep|cut -c 9-15|xargs kill -9



rm /home/zhaoyp/guandan_tog/learner_torch/ckpt_bak/*.pth
