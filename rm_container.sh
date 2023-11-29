#!/bin/bash
for i in {3..43}
do
	docker rm -f guandan_actor_$i
done

