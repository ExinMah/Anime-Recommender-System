import pickle
import streamlit as st
import numpy as np
import pandas as pd

st.header("Anime Recommender System")
algo_items = pickle.load(open('artifacts/algo_items.pkl', 'rb'))
trainset = pickle.load(open('artifacts/trainset.pkl', 'rb'))
testset = pickle.load(open('artifacts/testset.pkl', 'rb'))
algo = pickle.load(open('artifacts/algo.pkl', 'rb'))
animeDF = pickle.load(open('artifacts/anime_dataframe.pkl', 'rb'))
anime_titles = pickle.load(open('artifacts/anime_title.pkl', 'rb'))

def fetch_poster(suggestion):
    anime_name = []
    ids_index = []
    poster_url = []
    
    for anime_id in suggestion['title']:
        anime_name.append(anime_id)
        
    for name in anime_name:
        ids = np.where(animeDF['title'] == name)[0][0]
        ids_index.append(ids)
    
    for ids in ids_index:
        url = animeDF.iloc[ids]['image_url']
        poster_url.append(url)
        
    return poster_url

def fetch_tag(suggestion):
    anime_name = []
    ids_index = []
    tag_list = []
    
    for anime_id in suggestion['title']:
        anime_name.append(anime_id)
        
    for name in anime_name:
        ids = np.where(animeDF['title'] == name)[0][0]
        ids_index.append(ids)
    
    for ids in ids_index:
        url = animeDF.iloc[ids]['genre']
        tag_list.append(url)
        
    return tag_list

def get_item_recommendations(anime_title, anime_id=100000, k=10):
    anime_list = []
    tag_list = []
    
    if anime_id == 100000:     
        anime_id = animeDF[animeDF['title'] == anime_title]['anime_id'].iloc[0]
    
    if trainset.knows_item(10):
        neighbors = algo_items.get_neighbors(10, k=k+1)
    else:    
        iid = algo_items.trainset.to_inner_iid(anime_id)
        neighbors = algo_items.get_neighbors(iid, k=k+1)
        
    raw_neighbors = (algo.trainset.to_raw_iid(inner_id) for inner_id in neighbors)
    df = pd.DataFrame(raw_neighbors, columns = ['Anime_ID'])
    df = pd.merge(df, animeDF, left_on = 'Anime_ID', right_on = 'anime_id', how = 'left')
        
    recommendations = df[['Anime_ID', 'image_url', 'title', 'genre', 'premiered']]
        
    poster_url = fetch_poster(recommendations)
    tag_list = fetch_tag(recommendations)
    for i in recommendations['title']:
            anime_list.append(i)
            
    return anime_list, poster_url, tag_list
          
column_width = 200

if st.button("Show Recommendation"):
    recommended_anime, poster_url, tags = get_item_recommendations(selected_anime, kValue)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    if kValue >= 1: 
        with col1:
            st.text(recommended_anime[1])
            st.image(poster_url[1], width=column_width, use_column_width='always', caption=tags[1])
            
            if kValue >= 6:
                st.text(recommended_anime[6])
                st.image(poster_url[6], width=column_width, use_column_width='always', caption=tags[6])
    
            if kValue >= 2:
                with col2:
                    st.text(recommended_anime[2])
                    st.image(poster_url[2], width=column_width, use_column_width='always', caption=tags[2])
                    
                    if kValue >= 7:
                        st.text(recommended_anime[7])
                        st.image(poster_url[7], width=column_width, use_column_width='always', caption=tags[7])
            
                    if kValue >= 3:
                        with col3:
                            st.text(recommended_anime[3])
                            st.image(poster_url[3], width=column_width, use_column_width='always', caption=tags[3])
                            
                            if kValue >= 8:
                                st.text(recommended_anime[8])
                                st.image(poster_url[8], width=column_width, use_column_width='always', caption=tags[8])
    
                            if kValue >= 4:
                                with col4:
                                    st.text(recommended_anime[4])
                                    st.image(poster_url[4], width=column_width, use_column_width='always', caption=tags[4])
                                    
                                    if kValue >= 9:
                                        st.text(recommended_anime[9])
                                        st.image(poster_url[9], width=column_width, use_column_width='always', caption=tags[9])
    
                                    if kValue >= 5:
                                        with col5:
                                            st.text(recommended_anime[5])
                                            st.image(poster_url[5], width=column_width, use_column_width='always', caption=tags[5])
                                            
                                            if kValue >= 10:
                                                st.text(recommended_anime[10])
                                                st.image(poster_url[10], width=column_width, use_column_width='always', caption=tags[10])
