#!/bin/bash

if [ "$1" = "vision" ]; then
	python emoji_embedding/train_cnn.py
elif [ "$1" = "embedding" ]; then
	python emoji_embedding/create_embeddings.py
elif [ "$1" = "main" ]; then
	python train.py
elif [ "$1" = "sembert" ]; then
	python emoji_embedding/train_sembert.py
elif [ "$1" = "compare" ]; then
	python evaluate.py sembert_chunk51 "valid_v2" --text_col "text_no_emojis" --run_name main_run --function compare
	python evaluate.py sembert_chunk51 "valid_v2_min_2" --text_col "text_no_emojis" --run_name main_run --function compare
	python evaluate.py sembert_chunk51 "test_v2" --text_col "text_no_emojis" --run_name main_run --function compare
	python evaluate.py sembert_chunk51 "test_v2_min_2" --text_col "text_no_emojis" --run_name main_run --function compare
	python evaluate.py sembert_chunk51 "extra_zero_v2" --text_col "text_no_emojis" --run_name main_run --function compare
	python evaluate.py sembert_chunk51 "extra_zero_v2_min_2" --text_col "text_no_emojis" --run_name main_run --function compare
else
	echo "Invalid Option Selected"
fi