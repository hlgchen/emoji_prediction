#!/bin/bash

if [ "$1" = "vision" ]; then
	python emoji_embedding/train_cnn.py
elif [ "$1" = "embedding" ]; then
	python emoji_embedding/create_embeddings.py
elif [ "$1" = "main" ]; then
	python train.py
elif [ "$1" = "sembert" ]; then
	python emoji_embedding/train_sembert.py
elif [ "$1" = "evaluate" ]; then
	# sembert base on v2 of data (cased)
	python evaluate.py sembert_chunk14 "valid_v2" --k 1 --text_col "text_no_emojis"
	python evaluate.py sembert_chunk14 "valid_v2" --k 10 --text_col "text_no_emojis"
	python evaluate.py sembert_chunk14 "test_v2" --k 1 --text_col "text_no_emojis"
	python evaluate.py sembert_chunk14 "test_v2" --k 10 --text_col "text_no_emojis"
	# sembert cased but no training data removal
	python evaluate.py sembert_cased_chunk26 "valid_v2" --k 1 --text_col "text_no_emojis"
	python evaluate.py sembert_cased_chunk26 "valid_v2" --k 10 --text_col "text_no_emojis"
	python evaluate.py sembert_cased_chunk26 "test_v2" --k 1 --text_col "text_no_emojis"
	python evaluate.py sembert_cased_chunk26 "test_v2" --k 10 --text_col "text_no_emojis"
	# sembert cased min3 
	python evaluate.py sembert_cased_min3_chunk27 "valid_v2" --k 1 --text_col "text_no_emojis"
	python evaluate.py sembert_cased_min3_chunk27 "valid_v2" --k 10 --text_col "text_no_emojis"
	python evaluate.py sembert_cased_min3_chunk27 "test_v2" --k 1 --text_col "text_no_emojis"
	python evaluate.py sembert_cased_min3_chunk27 "test_v2" --k 10 --text_col "text_no_emojis"
	# sembert cased min3 cleaned
	python evaluate.py sembert_cased_min3_clean_chunk30 "valid_v2" --k 1 --text_col "text_no_emojis"
	python evaluate.py sembert_cased_min3_clean_chunk30 "valid_v2" --k 10 --text_col "text_no_emojis"
	python evaluate.py sembert_cased_min3_clean_chunk30 "test_v2" --k 1 --text_col "text_no_emojis"
	python evaluate.py sembert_cased_min3_clean_chunk30 "test_v2" --k 10 --text_col "text_no_emojis"

	### on different dataset 
	# sembert base on v2
	python evaluate.py sembert_chunk14 "valid_v2_min_2" --k 1 --text_col "text_no_emojis_clean"
	python evaluate.py sembert_chunk14 "valid_v2_min_2" --k 10 --text_col "text_no_emojis_clean"
	python evaluate.py sembert_chunk14 "test_v2_min_2" --k 1 --text_col "text_no_emojis_clean"
	python evaluate.py sembert_chunk14 "test_v2_min_2" --k 10 --text_col "text_no_emojis_clean"
	# sembert cased but no training data removal
	python evaluate.py sembert_cased_chunk26 "valid_v2_min_2" --k 1 --text_col "text_no_emojis_clean"
	python evaluate.py sembert_cased_chunk26 "valid_v2_min_2" --k 10 --text_col "text_no_emojis_clean"
	python evaluate.py sembert_cased_chunk26 "test_v2_min_2" --k 1 --text_col "text_no_emojis_clean"
	python evaluate.py sembert_cased_chunk26 "test_v2_min_2" --k 10 --text_col "text_no_emojis_clean"
	# sembert cased min3 
	python evaluate.py sembert_cased_min3_chunk27 "valid_v2_min_2" --k 1 --text_col "text_no_emojis_clean"
	python evaluate.py sembert_cased_min3_chunk27 "valid_v2_min_2" --k 10 --text_col "text_no_emojis_clean"
	python evaluate.py sembert_cased_min3_chunk27 "test_v2_min_2" --k 1 --text_col "text_no_emojis_clean"
	python evaluate.py sembert_cased_min3_chunk27 "test_v2_min_2" --k 10 --text_col "text_no_emojis_clean"
	# sembert cased min3 cleaned
	python evaluate.py sembert_cased_min3_clean_chunk30 "valid_v2_min_2" --k 1 --text_col "text_no_emojis_clean"
	python evaluate.py sembert_cased_min3_clean_chunk30 "valid_v2_min_2" --k 10 --text_col "text_no_emojis_clean"
	python evaluate.py sembert_cased_min3_clean_chunk30 "test_v2_min_2" --k 1 --text_col "text_no_emojis_clean"
	python evaluate.py sembert_cased_min3_clean_chunk30 "test_v2_min_2" --k 10 --text_col "text_no_emojis_clean"
else
	echo "Invalid Option Selected"
fi