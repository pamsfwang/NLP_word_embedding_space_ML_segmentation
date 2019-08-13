#import packages
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as pltimg
import matplotlib.cm as cm
from mpl_toolkits.mplot3d import Axes3D 

#import glove

#for calculating similarity
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import pairwise_distances
#from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# for k means clustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score


#for importing  jason files
import json

########################load glove model########################
gloveFile = '../word_embed/glove.840B.300d.txt'
print("Loading Glove Model")
f = open(gloveFile,'r', encoding='utf8')
model = {}
for line in f:
    splitLine = line.split(' ')
    word = splitLine[0]
    embedding = np.asarray(splitLine[1:], dtype='float32')
    model[word] = embedding
print("Done.",len(model)," words loaded!")

########################specify variables########################
#specify variables
wordfname = 'BRM'

##t-sne
n_comp = 20
##k means
kmeans_int_method = 'random'
repetition_num = 10
clusters_from = 10
clusters_to = 100

##ploting variables
title_fontsize = 20
axis_fontsize = 15

##new fname
new_fname = 'Pereira_whole_list_pca_BRM_common_words'

########################START########################
common_words_df = pd.read_csv("../word_lists/concrete_words/final_common_words_BRM.csv")
common_words = common_words_df['Word']

##load word embeddings
vec = []
words_label = []

for iw in common_words:
    try:
        temp = model[iw]
        words_label.append(iw)
        vec.append(list(temp)) #append all vectors
    except:
        print("no word embedding for ",iw)
print("size of the matrix: rows are words ",len(vec),"; columns are features (word embedding dimensions) ", len(vec[0]))

fn = '../results/'+new_fname+'/word_embeddings_'+wordfname+'.csv'
np.savetxt(fn,vec,delimiter=",")


#####
CS = cosine_similarity(vec)
pca_vec = PCA(n_components=n_comp)
pca_vec_result = pca_vec.fit_transform(CS)

#vec_embedded_tsne = TSNE(n_components = n_comp, metric = 'cosine').fit_transform(vec)
print('Variance explained per principal component: {}'.format(pca_vec.explained_variance_ratio_))
print('Total variance explained',sum(pca_vec.explained_variance_ratio_))

fn = '../results/'+new_fname+'/pca_comonents'+str(n_comp)+'_'+wordfname+'.csv'
pd.DataFrame(pca_vec_result).reset_index().to_csv(fn, index = False, header=False,float_format='%g')

temp = pd.DataFrame(pca_vec_result)
temp['words'] = words_label

fn = '../results/'+new_fname+'/words_pca_comonents'+str(n_comp)+'_'+wordfname+'.csv'
pd.DataFrame(temp).reset_index().to_csv(fn, index = False, header=False,float_format='%g')

fn = '../results/'+new_fname+'/pca_variance_explain_comp_'+str(n_comp)+'_'+wordfname+'.csv'
pd.DataFrame(pca_vec.explained_variance_ratio_).reset_index().to_csv(fn, index = False, header=False,float_format='%g')

##ploting
fig, ax1 = plt.subplots(1)
fig.set_size_inches(18, 7)
plt.plot(pca_vec.explained_variance_ratio_)
plt.axhline(y = 0, linewidth=1, linestyle = '--', color='r')
ax1.set_title('PCA components', fontsize=title_fontsize)
ax1.set_xlabel("Components", fontsize=axis_fontsize)
ax1.set_ylabel("PCA", fontsize=axis_fontsize)
plt.savefig('../results/'+new_fname+'/figures/PCA_components_'+kmeans_int_method+'.jpg')

###K MEANS###
#5 k-means (k = 200) on D using squared Euclidean distance and the k-means ++ algorithm
range_n_clusters = range(clusters_from,clusters_to,10)
repetition = range(0, repetition_num)
silhouette_data = pd.DataFrame()

for cluster_num in range_n_clusters:
    for rp in repetition :
        kmeans = KMeans(n_clusters=cluster_num,init= kmeans_int_method).fit(pca_vec_result)
        temp_silhouette_avg = silhouette_score(pca_vec_result, kmeans.labels_)
        temp_info = {'n_clusters': cluster_num,
                     'silhouette_avg': temp_silhouette_avg,
                     }
        silhouette_data = silhouette_data.append(temp_info,ignore_index=True)

    print("For n_clusters =", cluster_num,
        "The average silhouette_score is :", silhouette_data.loc[silhouette_data['n_clusters']==cluster_num,['silhouette_avg']].mean())
    sample_silhouette_values = silhouette_samples(pca_vec_result, kmeans.labels_)
    
    ###########################
    temp_silhouette_values_df = pd.DataFrame(pd.np.column_stack([words_label, kmeans.labels_,sample_silhouette_values]))
    temp_silhouette_values_df.rename(columns={1:'clusters', 0:'words',2:'slihouette_values'}, inplace=True)
    temp_silhouette_values_df.to_csv ('../results/'+new_fname+'/silhouette_values_kmeans_'+kmeans_int_method+'_clusternum'+str(cluster_num)+'.csv', index = None, header=True,float_format='%g')
    ###########################
    centroid = np.array(kmeans.cluster_centers_)
    test = pd.DataFrame(pd.np.column_stack([words_label, kmeans.labels_]))
    test.rename(columns={1:'clusters', 0:'words'}, inplace=True)
    dist_dataframe = pd.DataFrame()
    for ic, ic_vec in enumerate(centroid):#ic: cluster number; ic_vec: centroid vector
        index = kmeans.labels_==ic
        temp_words = test.iloc[index,0]
        temp_list = pca_vec_result[index]
        temp_cent = ic_vec
        #print(sum(index.astype(int)),len(temp_words),temp_list.shape,len(temp_cent),ic)
        for i, iw in enumerate(temp_words):
            temp_word = temp_list[i,:]#word vector
            dist = np.linalg.norm(temp_word -temp_cent)#manually checked -- this is the same as euclidean distance
            dist_info = {'clusters':ic,
                         'words':iw,
                         'dist_to_cent':dist
                         }
            dist_dataframe = dist_dataframe.append(dist_info,ignore_index=True)
    
    dist_reorganize = dist_dataframe.groupby('clusters').apply(lambda x: x.sort_values(['dist_to_cent']))
    dist_reorganize.to_csv ('../results/'+new_fname+'/clusters_word_dist'+kmeans_int_method+'_clusternum'+str(cluster_num)+'.csv', 
                            index = None, header=True,float_format='%g')
    
    ###########################PLOT SILHOUETTE###########################
    fig, ax1 = plt.subplots(1)
    fig.set_size_inches(18, 7)
    ax1.set_xlim([-0.1, 0.5])
    # The (n_clusters+1)*10 is for inserting blank space between silhouette
    # plots of individual clusters, to demarcate them clearly.
    ax1.set_ylim([0, len(pca_vec_result) + (cluster_num + 1) * 10])
    y_lower = 10
    for i in range(cluster_num):
        # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
        ith_cluster_silhouette_values = \
        sample_silhouette_values[kmeans.labels_ == i]

        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i

        color = cm.nipy_spectral(float(i) / cluster_num)
        ax1.fill_betweenx(np.arange(y_lower, y_upper),
                  0, ith_cluster_silhouette_values,
                  facecolor=color, edgecolor=color, alpha=0.7)

        # Label the silhouette plots with their cluster numbers at the middle
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        # Compute the new y_lower for next plot
        y_lower = y_upper + 10  # 10 for the 0 samples

        #####

        ax1.set_title("The silhouette plot for the various clusters. n_clusters %d" % cluster_num,fontsize=title_fontsize)
        ax1.set_xlabel("The silhouette coefficient values",fontsize=axis_fontsize)
        ax1.set_ylabel("Cluster label",fontsize=axis_fontsize)
        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=temp_silhouette_avg, color="red", linestyle="--")
        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, -0.05, 0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5])
        
        ###save plots
        plt.savefig('../results/'+new_fname+'/figures/silhouetteplot_C_method_'+kmeans_int_method+'_clusternum'+str(cluster_num)+'.jpg')


silhouette_data.to_csv('../results/'+new_fname+'/silhouette_data_c_kmethod_'+kmeans_int_method+'.csv', index = None, header=True,float_format='%g')
