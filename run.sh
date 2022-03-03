#!/bin/bash

if [ "$1" = "vision" ]; then
	python emoji_embedding/train_cnn.py
elif [ "$1" = "embedding" ]; then
	python emoji_embedding/create_embeddings.py
elif [ "$1" = "main" ]; then
	python train.py
elif [ "$1" = "sembert" ]; then
	python emoji_embedding/train_sembert.py
else
	echo "Invalid Option Selected"
fi