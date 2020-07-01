# NLP_word_embedding_space_ML_segmentation
This is a project that aims to understand how semantic knowledge, our general knowledge of the world, influences our memory. I use word embeddings from Natual Language Processing GloVe to quantify semantic similariy of words. 

**pca_kmeans_clustering_BRM_common_words.py**: this script performs one approach I have tried to segment word embedding (GloVe) similarity matrix into sub-clusters by k-means clustering. Because this is an unsupervised method, I use silhouette coefficients to compare clustering results using different k.

**extract_bert_embeddings.ipynb**: this is a colab script. I use this script to obtain pre-trained word embeddings from BERT. The purpose of this exploratory analysis is to see how adjacent words affect semantic similarity between target words and studied words (I generated these words using GloVe for my study). 
