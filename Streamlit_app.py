import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Ad data Analysis",
    page_icon="📊",
    layout="centered",
    initial_sidebar_state="collapsed"
)

df = pd.read_excel("HeadlineData-Drivers-Binary-Actual.xlsx")

df.drop(["delete"],axis=1,inplace=True)
df.columns=["Words","Coefficient"]
df.iloc[0,0]="1"
df["Words"] = df["Words"].apply(lambda a : "$"+str(a) if str(a)[0] in "0123456789" else a)
df["Words"] = df["Words"].str.lower()

st.subheader("Rate your Ad title")

input = st.text_input("Enter the Ad title to check the rating:")

submit = st.button("Check Rating!")

if submit:
    input = input.lower()
    rating = 0
    for word in input.split(" "):
        try:
            rating += df[df["Words"]==word]["Coefficient"].values[0]
        except:
            continue

    st.text("Rating for your Ad Title: "+str(rating))

st.subheader("Compare Ad titles")

if "titles" not in st.session_state:
   st.session_state.titles = []

ad_title = st.text_input("Enter your Ad title:")
title_added = st.button("Add title")

if title_added:
   st.session_state.titles.append(ad_title)
   st.text("Ad Titles:")
   for i in st.session_state.titles:
       st.text(i)

compare = st.button("Compare titles")

if compare:
    ratings = []
    print(st.session_state.titles)
    for title in st.session_state.titles:
        title = title.lower()
        rating = 0
        for word in title.split(" "):
            try:
                rating += df[df["Words"]==word]["Coefficient"].values[0]
            except:
                continue
        ratings.append(rating)

    result = pd.DataFrame({"Ad Title":st.session_state.titles,"Rating":ratings}).sort_values("Rating",ascending=False)
    result.index = np.arange(1,len(result)+1)
    st.subheader("Comparison Report")
    st.dataframe(result)
    st.text("Best Ad title among your inputs: ")
    st.text(str(result.iloc[0,0]))

        
           
       

