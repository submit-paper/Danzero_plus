#!/bin/bash
# docker network create --driver bridge --subnet=172.15.15.0/24 --gateway=172.15.15.1 guandanNet
#docker run -itd --gpus all --network=guandanNet --ip 172.15.15.2 --name guandan_learner -v /home/luyd/guandan_tog/:/home/luyd/guandan_tog -w /home/luyd/guandan_tog/ nvcr.io/nvidia/tensorflow:22.02-tf1-py3
for i in {3..13}
do
    docker run -itd --network=guandanNet --ip 172.15.15.$i --name guandan_actor_$i -v /home/zhaoyp/log/log$i:/home/root/log -v /home/zhaoyp/guandan_tog:/home/zhaoyp/guandan_tog -w /home/zhaoyp/guandan_tog  guandan_actor:v5 /bin/bash
done
for i in {14..43}
do
    docker run -itd --network=guandanNet --ip 172.15.15.$i --name guandan_actor_$i -v /home/zhaoyp/guandan_tog:/home/zhaoyp/guandan_tog -w /home/zhaoyp/guandan_tog  guandan_actor:v5 /bin/bash
done

