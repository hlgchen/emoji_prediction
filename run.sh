#!/bin/bash

if [ "$1" = "vision" ]; then
	python emoji_embedding/train_cnn.py
elif [ "$1" = "embedding" ]; then
	python emoji_embedding/create_embeddings.py
else
	echo "Invalid Option Selected"
fi