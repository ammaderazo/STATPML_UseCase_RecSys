from re import M
import pandas as pd
import numpy as np 
import streamlit as st 
import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


st.set_page_config(layout="wide")
st.title("Amazon Prime Videos Recommender System")
st.caption("Recommender System Deployment as part of a technical use case for STAT280 Practical Machine Learning")

@st.cache
media_df = pd.read_pickle('Resources/AmazonPrimeDF_processed.pkl')
titles = media_df['title']
titles = titles.sort_values()

vectorizer = CountVectorizer()
mtx_trans = vectorizer.fit_transform(media_df["features"])
cos_sim = cosine_similarity(mtx_trans)


def generate_recos(input_media, num_reco):
    source_id = media_df[media_df["title"]== input_media].index.values[0]
    sim_list= list(enumerate(cos_sim[source_id]))
    sorted_sim = sorted(sim_list, key=lambda x:x[1], reverse=True)[1:]
    sorted_sim = sorted_sim[0: num_reco]
    sorted_sim = pd.DataFrame(sorted_sim)
    index_media = list(sorted_sim[0])
    conf_list = list(sorted_sim[1])
    selected_media = media_df.loc[index_media]
    selected_media['confidence'] = conf_list
    selected_media = selected_media.reset_index()
    selected_media = selected_media.drop(['index', 'features', 'runtime','age_certification', 'seasons', 'confidence'],axis =1 )
    return selected_media


selected_media = st.selectbox(
    "Which Amazon Prime Media would you like to find recommendations for?",
    titles
)



selected_num = st.number_input('Number of Recommendations:' , min_value=1, max_value=50, step=1)

if st.button("Show Recommendation"):
    rec_df = generate_recos(selected_media, selected_num)
    with st.spinner('Retrieving Amazon Prime Movies/Shows'):
        time.sleep(5)
        st.success('Done!')
    st.write(rec_df)


