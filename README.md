# emoji_prediction

This repository contains the code to train and test emoji recommendation models. 

To train models on a GPU install requirements with pip in `gpu_requirements.txt`

All required dependencies are in `requirements.txt`


The dataset used for training is aquired from the paper Cappallo et al. (2018) [1].

### Example predictions

Stop the war in Ukraine, we need peace!
['🙏', '😔', '😢', '💔', '🇺🇦', '☮️', '🇷🇺']

I came back home and we didn't have any food left
['🙃', '😭', '🙄', '😒', '🏠', '🏘️', '🔙']

send nudes
['😛', '👅', '😋', '😈', '😏', '👙', '🚚']

Why does my car always break down, I hate it!
['😤', '😭', '😒', '🙄', '🚗', '🏎️', '🚓']

sweet potato fries are overrated
['🙄', '😷', '😒', '😂', '🥔', '🍟', '🍠']

Whales are such majestic creatures.
['😍', '🐟', '🌊', '😭', '🐋', '🐬', '🐳']

I don't think he deserved to be treated like that
['😔', '😞', '😪', '😕', '😒', '💔', '😢']

Christopher Manning's papers are really lit.
['🔥', '😂', '👌', '😭', '📚', '📰', '🗞️']

love the way you lie
['🎶', '😍', '😏', '🎧', '😊', '😌', '🤥']



### References

[1] pencer Cappallo, Stacey Svetlichnaya, Pierre Garrigues, Thomas Mensink, and Cees GM Snoek.
New modality: Emoji challenges in prediction, anticipation, and retrieval. IEEE Transactions on
Multimedia, 21(2):402–415, 2018.
10
