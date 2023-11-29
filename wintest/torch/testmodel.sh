#!/bin/bash
nohup /home/zhaoyp/guandan_tog/wintest/torch/danserver 50 >/dev/null 2>&1 & 
sleep 0.5s
nohup /root/miniconda3/envs/guandan/bin/python /home/zhaoyp/guandan_tog/wintest/ai4/client1.py >/dev/null 2>&1 &
# /root/miniconda3/envs/guandan/bin/python /home/zhaoyp/guandan_tog/wintest/random_clien0.py 2>&1 &
sleep 0.5s
nohup /root/miniconda3/envs/guandan/bin/python /home/zhaoyp/guandan_tog/wintest/torch/client1.py --resfile res$1v4.log >/dev/null  2>&1 &
# /root/miniconda3/envs/guandan/bin/python /home/zhaoyp/guandan_tog/wintest/newversion/my/client1.py --resfile res$1v9.log 2>&1 &
sleep 0.5s
nohup /root/miniconda3/envs/guandan/bin/python /home/zhaoyp/guandan_tog/wintest/ai4/client3.py >/dev/null 2>&1 &
sleep 0.5s
nohup /root/miniconda3/envs/guandan/bin/python /home/zhaoyp/guandan_tog/wintest/torch/client3.py >/dev/null 2>&1 &
sleep 0.5s
nohup /root/miniconda3/envs/guandan/bin/python /home/zhaoyp/guandan_tog/wintest/torch/actor.py --iter $1 >/dev/null 2>&1 &
# /root/miniconda3/envs/guandan/bin/python /home/zhaoyp/guandan_tog/wintest/newversion/my/actor.py --model $1 2>&1 &
echo $1

