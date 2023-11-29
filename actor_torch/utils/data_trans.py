import datetime
import os
import io
import pickle
import time
from itertools import count
from pathlib import Path
from typing import Any, Tuple
import threading
import torch
import zmq


class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)


def find_new_weights(current_model_id: int, ckpt_path: Path) -> Tuple[Any, int]:
    try:
        ckpt_files = sorted(os.listdir(ckpt_path), key=lambda p: int(p.split('.')[0]))
        latest_file = ckpt_files[-1]
    except IndexError:
        # No checkpoint file
        return None, -1
    new_model_id = int(latest_file.split('.')[0])

    if int(new_model_id) > current_model_id:
        loaded = False
        while not loaded:
            try:
                with open(ckpt_path / latest_file, 'rb') as f:
                    new_weights = CPU_Unpickler(f).load()
                loaded = True
            except (EOFError, pickle.UnpicklingError):
                # The file of weights does not finish writing
                pass
        return new_weights, new_model_id
    else:
        return None, current_model_id


def create_experiment_dir(args, prefix: str) -> None:
    if args.exp_path is None:
        args.exp_path = prefix + datetime.datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d-%H-%M-%S')
    args.exp_path = Path(args.exp_path)

    if args.exp_path.exists():
        os.system(f'rm -rf {args.exp_path}')
        # raise FileExistsError(f'Experiment directory {str(args.exp_path)!r} already exists')

    args.exp_path.mkdir()


def run_weights_subscriber(args, unknown_args):
    """Subscribe weights from Learner and save them locally"""
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(f'tcp://{args.ip}:{args.param_port}')
    socket.setsockopt_string(zmq.SUBSCRIBE, '')  # Subscribe everything
    def recv_weight():
        try:
            weights = socket.recv(flags=zmq.NOBLOCK)
            # Weights received
            with open(args.ckpt_path / f'{model_id}.pth', 'wb') as f:
                f.write(weights)
            del weights
            
            if model_id > args.num_saved_ckpt:
                os.remove(args.ckpt_path / f'{model_id - args.num_saved_ckpt}.pth')
        except zmq.Again:
            pass
    for model_id in count(1):  # Starts from 1
        t = 3
        recv_weight_thread = threading.Timer(t, recv_weight)
        while True:
            recv_weight_thread.run()
            recv_weight_thread.finished.clear()
            # try:
            #     weights = socket.recv(flags=zmq.NOBLOCK)
            #     # Weights received
            #     with open(args.ckpt_path / f'{model_id}.ckpt', 'wb') as f:
            #         f.write(weights)

            #     if model_id > args.num_saved_ckpt:
            #         os.remove(args.ckpt_path / f'{model_id - args.num_saved_ckpt}.ckpt')
            #     break
            # except zmq.Again:
            #     pass

            # # For not cpu-intensive
            # time.sleep(1)

