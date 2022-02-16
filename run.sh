#!/bin/bash

if [ "$1" = "vision" ]; then
	python emoji_embedding/train_cnn.py
else
	echo "Invalid Option Selected"
fi