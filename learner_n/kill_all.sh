#!/bin/bash
for i in {3..83}
do
	sshpass ssh root@172.15.15.$i "bash /home/zhaoyp/guandan/actor_n/kill.sh"
done
ps aux|grep python|grep -v grep|cut -c 9-15|xargs kill -9