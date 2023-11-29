#!/bin/bash
nohup /home/zhaoyp/guandan_tog/actor_torch/danserver 100000 >/dev/null 2>&1 &
sleep 0.5s
nohup /root/miniconda3/envs/guandan/bin/python -u /home/zhaoyp/guandan_tog/actor_torch/actor.py > /home/zhaoyp/actor_out.log 2>&1 &
sleep 0.5s
nohup /root/miniconda3/envs/guandan/bin/python -u /home/zhaoyp/guandan_tog/actor_torch/game.py > /home/zhaoyp/game_out.log 2>&1 &
sleep 0.5s
nohup /root/miniconda3/envs/guandan/bin/python -u /home/zhaoyp/guandan_tog/actor_torch/restart.py > /home/zhaoyp/restart_out.log 2>&1 &
