for i in {50..52}
do
    docker run -itd --network=guandanNet --ip 172.15.15.$i --name guandan_actor_$i -v /home/zhaoyp/guandan_tog:/home/zhaoyp/guandan_tog -w /home/zhaoyp/guandan_tog  guandan_actor:v5 /bin/bash
done
