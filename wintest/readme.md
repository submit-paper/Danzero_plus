We provide the evaluation code in this directory.
For the rule-based bots, they can be used for the four positions.
For the DMC model, they need to be deployed at player0 and player2 while the PPO model is deployed at player1 and player3.

In ./torch, we give the evaluation shell file, which you can follow to conduct needed evaluation. Here we also give an 
example to show how to evaluate the model during the training process, that you can refer to the evaluate_xxx.py.
Because the interval between models to be saved is almost the same, you can adjust the model id to get the checkpoints you want.

Here we give an example to conduct the evaluation. After copying the tested models to the target dir, you can just
execute the command "bash testmodel.sh xx", where xx is the model_id. How it is set can be referred to in ./torch/actor.py.
If you want to execute the evaluation during training, maybe you can run "nohup python -u evaluate_xx.py > xx.log &".
In this way, you can just see the log file to see how many models have been tested.
