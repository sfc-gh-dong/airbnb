import time

import streamlit as st
import snowflake.connector
from snowflake.snowpark import DataFrame
from snowflake.snowpark.session import Session
from snowflake.snowpark import functions as F
from snowflake.snowpark.functions import col, lit, udf
from snowflake.snowpark.session import Session
from snowflake.snowpark.exceptions import SnowparkSQLException
import pandas as pd
import numpy as np
import plotly.express as px
import pydeck as pdk
from datetime import datetime, date, timedelta

import re

st.set_page_config(layout="wide")
session = st.connection("snowflake").session()

@st.cache_data
def get_data():
    df = session.table("AIRBNB_CLEAN")
    return df.to_pandas()


def ask_llm(q):
    return session.sql(q)

def test_sql(q):
    try:
        session.sql("select * from AIRBNB_CLEAN where " +q +" limit 1").collect()
        return True
    except SnowparkSQLException as e:
        st.markdown("Using default SQL")
        return False

def format_output(s):
    formatted_output = re.sub(r'<q1>(.+)</q1>',r'<q1>:green[\1]</q1>', s)
    formatted_output = re.sub(r'<q2>(.+)</q2>',r'<q2>:green[\1]</q2>', formatted_output)
    formatted_output = formatted_output.replace("</q1>","</q1>  \n")
    return formatted_output


def get_stars(s):
    stars = ""
    ret = 0
    if s is not None:
        try:
            ret = float(s)
        except ValueError:
            ret = 0

        for i in range(0,int(ret)):
            if i>0:
                stars = stars +"⭐"

    return stars

# Custom color scale
COLOR_RANGE = [
[255, 0, 0],
[255, 128, 0],
[255, 255, 0],
[128, 255, 0],
[0, 255, 0],
]

BREAKS = [-0.5, 0.25, 0.5, 0.8, 1]


def color_scale(val):
    for i, b in enumerate(BREAKS):
        if val < b:
            return COLOR_RANGE[i]
    return COLOR_RANGE[i]

def form_question(decomposed, prompt2):

    q1_phrase = ""
    q2_phrase = ""
    q1 = ""
    q2 = ""
    q = ""

    q1_search = re.search('<q1>(.+)</q1>', decomposed, re.IGNORECASE)
    if q1_search: 
        q1_phrase = q1_search.group(1).strip()
    q2_search = re.search('<q2>(.+)</q2>', decomposed, re.IGNORECASE) 
    if q2_search: 
        q2_phrase = q2_search.group(1).strip()

    if not test_sql(q1_phrase): q1_phrase=""

    # both llm and sql
    if q1_phrase != "" and q2_phrase != "":
        q1embed = "select *, VECTOR_L2_DISTANCE(embedding, snowflake.cortex.embed_text('e5-base-v2', '{llm}')) as l2 from airbnb_clean \
                  where {where} order by l2 asc limit 10"
        q1 = q1embed.format(llm=q2_phrase, where=q1_phrase)
        q2part1 = "concat('based on these reviews <review>',reviews,'</review> of an airbnb listing, does it have {llm}? {prompt}"
        q2 = "select name, guests, price, summary, sentiment, image, lat, lon, url, snowflake.cortex.complete('llama2-70b-chat'," +q2part1.format(llm=q2_phrase, prompt=prompt2) +") as results, \
              case when left(trim(results), 9)='Part 1: 1' then True else False end as filter from "
        q = q2 +"(" +q1 +")"
    # only llm
    elif q1_phrase == "" and q2_phrase != "":
        q1embed = "select *, VECTOR_L2_DISTANCE(embedding, snowflake.cortex.embed_text('e5-base-v2', '{llm}')) as l2 from airbnb_clean order by l2 asc limit 10"
        q1 = q1embed.format(llm=q2_phrase)                    
        q2part1 = "concat('based on these reviews <review>',reviews,'</review> of an airbnb listing, does it have {llm}? {prompt}"
        q2 = "select name, guests, price, summary, sentiment, image, lat, lon, url, snowflake.cortex.complete('llama2-70b-chat'," +q2part1.format(llm=q2_phrase, prompt=prompt2) +") as results, \
              case when left(trim(results), 9)='Part 1: 1' then True else False end as filter from "
        q = q2 +"(" +q1 +")"
    # only sql
    elif q1_phrase != "" and q2_phrase == "":
        q1embed = "select * from airbnb_clean where {where} limit 50"
        q1 = q1embed.format(where=q1_phrase)
        q2 = "select name, guests, price, summary, sentiment, image, lat, lon, url from "
        q = q2 +"(" +q1 +")"
    else:
        q = ""

    return q


def main():

    st.title("🏖️ Wander : AirBnB LLM Chat Search")
    st.markdown('Hello, I\'m Wander, AirBnB newest chatbot! I will help you search for the right AirBnB listing based on what you want. 😎')
    st.markdown('Type something like this : :green[Show me listings with a great sea view! I have a minimum of 5 guests]')

    df = get_data()
    df = df[["NAME","CLEANLINESS_R","ACCURACY_R","COMMUNICATION_R","LOCATION_R","CHECKIN_R","VALUE_R","PRICE","GUESTS","REVIEWS","SUMMARY","SENTIMENT","IMAGE","LAT","LON"]]
    df["CLEANLINESS_S"] = df["CLEANLINESS_R"].apply(get_stars)
    df["ACCURACY_S"] = df["ACCURACY_R"].apply(get_stars)
    df["COMMUNICATION_S"] = df["COMMUNICATION_R"].apply(get_stars)
    df["LOCATION_S"] = df["LOCATION_R"].apply(get_stars)
    df["CHECKIN_S"] = df["CHECKIN_R"].apply(get_stars)
    df["VALUE_S"] = df["VALUE_R"].apply(get_stars)
    df["FILL_COLOR"] = df["SENTIMENT"].apply(color_scale)
    
    with st.expander("Preview data of selected fields."):
        sample = df[df['REVIEWS'].str.count('\s+').gt(500)]
        st.dataframe(sample[["NAME","PRICE","GUESTS","REVIEWS"]].sample(5),use_container_width=True)

    with st.expander("Have a look at our AirBnB locations worldwide, with added summarization and sentiment."):
        st.pydeck_chart(pdk.Deck(
            map_style='dark',
            initial_view_state=pdk.ViewState(
                latitude=df['LAT'].mean(),
                longitude=df['LON'].mean(),
                zoom=2,
                pitch=0,
            ),
            layers=[
                pdk.Layer(
                    'ScatterplotLayer',
                    data=df,
                    pickable=True,
                    filled=True,
                    opacity=0.5,
                    get_position=["LON", "LAT"],
                    get_fill_color="FILL_COLOR",
                    line_width_min_pixels=1,
                    get_radius=500,
                    stroked=True,
                    radius_scale=6,
                    radius_min_pixels=6,
                    radius_max_pixels=100,
                    auto_highlight=True,
                )
            ],
            tooltip={"html": "<b><u>{NAME}</u></b><br/>\
                              <b>Cleanliness : </b> {CLEANLINESS_S}<br/>\
                              <b>Accuracy : </b> {ACCURACY_S}<br/>\
                              <b>Communication : </b> {COMMUNICATION_S}<br/>\
                              <b>Location : </b> {LOCATION_S}<br/>\
                              <b>Check In : </b> {CHECKIN_S}<br/>\
                              <b>Value : </b> {VALUE_S}<br/>\
                              <b>Summary : </b><br/> {SUMMARY}<br/>\
                              <b>Sentiment : </b>{SENTIMENT}<br/>\
                              <img src='{IMAGE}' style='height: 160px; font-size : 14px; width: auto;'></img>", "style": {"color": "white"}},
        ))

    prompt1 = "select snowflake.cortex.complete('llama2-70b-chat',\
    '<Context>\
    You are Wander, a chatbot for airbnb. You need to decide based on the question, if a user is searching for listing, or just chatting in general. If the user is searching for a listing then do this : \
    \
    The question can be separated by comma,and,or. You need decompose the question into 2 parts. You are given a table with the columns : price, guests. \
    Part 1 provides the WHERE clause for querying the given table and you must only use the given column names. \
    Part 2 queries an LLM \
    Part 1 must only contain conditions based on the columns (price, guests). \
    The output format is :\
    \
    Example Question : \
    Show me the listings with good host, a sea view \
    \
    Example Output :\
    <q1>price >= 0 and guests >= 0</q1>\n\
    \
    <q2>good host and a sea view</q2>\
    \
    Example Question : \
    Show me the listings which are near town. I have a budget of 100 \
    \
    Example Output :\
    <q1>price <= 100 and guests >= 0</q1>\n\
    \
    <q2>near town</q2>\
    \
    Do provide repeated or overlapping criteria in each part. \
    Do not provide any other explanation outside of this format.\
    If the user is just chatting in general, then you do not need to follow the format mentioned.\
    \
    </Context>\
    \
    <Question>{0}</Question>\
    ')"

    prompt2 = " The output format is : \
    \
    Part 1: 1 if all criteria are met, and 0 otherwise \
    \
    Part 2: Provide explanation \
    \
    \
    Example Output: \
    Part 1: 1 \
    Part 2: {Explain} \
    \
    Do not provide any other explanation outside of this format\
    ')"


    if "messages" not in st.session_state:
        # system prompt includes table information, rules, and prompts the LLM to produce
        # a welcome message to the user.
        st.session_state.messages = [{"role": "system", "content": "Welcome"}]

    # Prompt for user input and save
    if prompt := st.chat_input():
        prompt = prompt.replace("'","\'")
        st.session_state.messages.append({"role": "user", "content": prompt})


    # display the existing chat messages
    for message in st.session_state.messages:
        if message["role"] == "system":
            continue
        with st.chat_message(message["role"]):
            st.markdown(format_output(message["content"]))


    if st.session_state.messages[-1]["role"] == "user":
        with st.chat_message("assistant"):

            prompt1 = prompt1.format(st.session_state.messages[-1]["content"])
            decomposed = ask_llm(prompt1).collect()[0][0]

            if "<q1>" in decomposed :
                decomposed = "Let's first decompose your question..  \n  \n" + decomposed + "  \n  \n" +"Hang on while I ask Snowflake Cortex.."
            else:
                decomposed = decomposed

            resp_container = st.empty()
            if decomposed is not None:
                resp_container.markdown(format_output(decomposed))
     
                q = form_question(decomposed, prompt2)

                if q!="":
                    results_container = st.empty()
                    df = ask_llm(q).to_pandas()
                    df = df[df["FILTER"]==True]
                    df["RESULTS"] = df["RESULTS"].str.replace("Part 1: 1", "")
                    df["RESULTS"] = df["RESULTS"].str.replace("Part 2:", "")
                    df["RESULTS"] = df["RESULTS"].str.strip()
 
                    st.pydeck_chart(pdk.Deck(
                        map_style='dark',
                        initial_view_state=pdk.ViewState(
                            latitude=df['LAT'].mean(),
                            longitude=df['LON'].mean(),
                            zoom=2,
                            pitch=0,
                        ),
                        layers=[
                            pdk.Layer(
                                'ScatterplotLayer',
                                data=df,
                                pickable=True,
                                filled=True,
                                opacity=0.6,
                                get_position=["LON", "LAT"],
                                get_fill_color=[255, 140, 0],
                                line_width_min_pixels=1,
                                get_radius=500,
                                stroked=True,
                                radius_scale=6,
                                radius_min_pixels=10,
                                radius_max_pixels=100,
                                auto_highlight=True,
                            )
                        ],
                        tooltip={"html": "<img src='{IMAGE}' style='height: 160px; width: auto;'></img><br/>\
                                <b><u>{NAME}</u></b><br/>\
                                <b>Number of Guests : </b> {GUESTS}<br/>\
                                <b>Price : </b> {PRICE}<br/>\
                                <b>LLM Reason : </b><br/> {RESULTS}<br/><br/>\
                                <a href>{URL}</a><br/>\
                        ", "style": {"color": "white", "font-size" : "16px"}},
                    ))
                
                    message = {"role": "assistant", "content": decomposed}
                    st.session_state.messages.append(message)



if __name__ == "__main__":
    main()


 
    


