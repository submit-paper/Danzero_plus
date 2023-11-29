#!/bin/bash
ps -ef | grep danserver | grep -v grep | awk '{print $2}' | xargs kill -9
ps -ef | grep actor.py | grep -v grep | awk '{print $2}' | xargs kill -9
ps -ef | grep game.py | grep -v grep | awk '{print $2}' | xargs kill -9