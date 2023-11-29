As this work is based on "DanZero: Mastering GuanDan Game with Reinforcement Learning", the code is also built on
the repository "https://github.com/AltmanD/guandan_mcc/tree/main".

## Install
lib needs(If you just train the DMC model, torch is not required):

linux20.04

python=3.8

tensorflow=1.15.5

numpy=1.18.5

websocket(ws4py)=0.5.1

pyarrow=5.0.0

pyzmq=22.3.0

torch=1.9.1+cpu(actor) or 1.13.1+cu116(learner)

To realize the communication between dockers, you can refer to https://cloud.tencent.com/developer/article/1013167.
If you use the docker, follow the create_containeder.sh to set docker network.
Then you can enter the learner and use "ssh-keygen -t rsa" to create the pub file and copy this to authorized_keys file in
the actors. After that, edit the /etc/ssh/ssh_config file to set "StrictHostKeyChecking" to be no.
In this way, the dockers can communicate directly.

## Run
The direct command to run the code is as below:

actor:
python actor_n/actor.py

learner:
python learner_n/learner.py

Here we offer a start shell file in the learner directory.

## Evaluation

The evaluation code is in the ./wintest directory and we give introduction in the directory.