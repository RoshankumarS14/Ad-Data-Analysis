import streamlit as st
import pandas as pd
import numpy as np
from category_encoders import TargetEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

st.set_page_config(
    page_title="Ad data Analysis",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed"
)

data = pd.read_excel("HeadlineData.xlsx")
data.dropna(inplace=True)
df = data.copy()

st.header("Basic analysis using filter")
st.write("""
Welcome to our dataset filtering tool! This interactive component allows you to efficiently filter, sort, and explore your dataset based on various criteria. Whether you're searching for specific categories, narrowing down value ranges, or sorting columns, this tool provides a user-friendly interface to manipulate your dataset effortlessly.

**Instructions** 🤔
1. **Select Categories**: Begin by choosing the categories you want to filter. Utilize dropdown menus provided to narrow down your dataset.
2. **Set Value Ranges**: Once you've selected the categories, specify the desired value range for CPL. Sliders are be available to help define these ranges.
3. **Sorting Columns**: Click on the column headers to sort the dataset based on that particular column. A simple click can toggle between ascending and descending order.
4. Experiment and Refine: Feel free to adjust your filters and sorting options as needed. Explore different combinations to extract the exact information you're looking for.
    
After setting your filters and sorting preferences, observe the updated dataset. You'll see only the records that meet your specified criteria, sorted as per your preference.
         
""")


ad_title = st.selectbox("Ad Title",np.concatenate([["All"],df["AdTitle"].unique()]))
position = st.selectbox("Position",np.concatenate([["All"],df["Position"].unique()]))
ad_state = st.selectbox("Ad State",np.concatenate([["All"],df["AdState"].unique()]))
campaign = st.selectbox("Campaign Market",np.concatenate([["All"],df["CampaignMarket"].unique()]))
cpl = st.slider("CPL",df["CPL"].min(),df["CPL"].max(),(25.0,500.0))

query = []
if ad_title != "All":
    query.append("AdTitle==@ad_title")
if position != "All":
    query.append("Position==@position")
if ad_state != "All":
    query.append("AdState==@ad_state")
if campaign != "All":
    query.append("CampaignMarket==@campaign")
query.append("CPL>=@cpl[0] & CPL<=@cpl[1]")

query = " & ".join(query)
df_filter = data.query(query)

st.dataframe(df_filter)

st.header("Machine Learning Prediction")

st.write("""
Welcome to our CPL prediction tool! This component employs machine learning to forecast Cost Per Lead (CPL) values based on various factors such as ad title, ad state, market campaign, and position. By selecting specific categories within these columns, you can generate predictions for the CPL value associated with your ad campaigns.

**Instructions** 
1. **Select Prediction Categories:** Begin by choosing the categories within "ad title," "ad state," "market campaign," and "position" columns. These selections will form the basis for predicting the CPL value.
2. **Click on Predict:** Once you've set your chosen categories, click on the "Predict" button. This action triggers the machine learning model to generate a forecasted CPL value based on the provided selections.
3. **Review Predicted CPL:** After clicking "Predict," the tool will display the forecasted CPL value based on the chosen categories. This value gives you insight into the anticipated Cost Per Lead for your ad campaign under the specified conditions.
4. **Experiment with Different Categories:** Feel free to modify your category selections to observe how different combinations might affect the predicted CPL. This experimentation can help optimize your ad campaigns for better lead generation.    

**Refine and Optimize:** Use the predicted CPL values as insights to refine your ad strategies. Adjust categories or explore different scenarios to potentially improve your CPL and campaign effectiveness.
                 
""")

ad_title = st.selectbox("Ad Title",df["AdTitle"].unique())
position = st.selectbox("Position",df["Position"].unique())
ad_state = st.selectbox("Ad State",df["AdState"].unique())
campaign = st.selectbox("Campaign Market",df["CampaignMarket"].unique())

predict = st.button("Predict CPL")

Q1 = np.quantile(df["CPL"],0.25)
Q3 = np.quantile(df["CPL"],0.75)
IQR = Q3-Q1
ul = Q3+(IQR*1.5)
ll = Q1=(IQR*1.5)
for index in df.index:
    if df.loc[index,"CPL"]>ul:
        df.loc[index,"CPL"]=ul

cat_cols = ["CampaignMarket","AdState","Position","AdTitle"]  # replace with your actual column names

encoder = TargetEncoder()
values = encoder.fit_transform(df[cat_cols], df['CPL'])
df = df.drop(columns=cat_cols)
df = pd.concat([df, values], axis=1)

X = df[cat_cols]
y = df["CPL"]
gb = GradientBoostingRegressor().fit(X,y)

input = encoder.transform(pd.DataFrame({"CampaignMarket":campaign,"AdState":ad_state,"Position":position,"AdTitle":ad_title},index=[0]))

if predict:
    st.subheader("Prediction results")
    st.write("Campaign Market : "+campaign)
    st.write("Ad State : "+ad_state)
    st.write("Position : "+position)
    st.write("Ad Title : "+ad_title)
    st.write("CPL : "+str(gb.predict(input)[0]))

st.header("Text Analysis of Ad Title")

st.write("""
Welcome to our Ad Title Analysis tool! This component utilizes Natural Language Processing (NLP) techniques to analyze the impact of words within ad titles on the Cost Per Lead (CPL) values. By breaking down ad titles into individual words and computing coefficients, this tool provides insights into the significance of each word in influencing CPL.
         
**Instructions** 
1. **Select Categories:** Begin by filtering the dataset by choosing the categories you want. Utilize dropdown menus provided to narrow down your dataset.
2. **Initiate Analysis:** Once the ad titles are provided, trigger the analysis process. The tool will delve into each word within the titles and compute coefficients indicating their impact on the CPL value.
3. **Review Coefficients:** After the analysis is complete, the tool will display coefficients for each word found in the ad titles. These coefficients signify the level of influence each word has on the CPL value. Positive coefficients indicate a positive impact, while negative coefficients suggest the opposite.
4. **Interpret Results:** Use the coefficients as a guide to understand which words in your ad titles might be positively or negatively correlated with CPL. This insight can help in crafting more effective ad titles for better CPL outcomes.
5. **Experiment and Refine:** Try different ad titles or modify existing ones to observe changes in the coefficients. Experimentation allows you to refine your ad titles for potentially improved CPL performance.

**Utilize Insights:** Incorporate the insights gained from coefficient analysis into your advertising strategies. Optimize ad titles by emphasizing words positively correlated with CPL and possibly avoiding or adjusting those with negative correlations.                 
""")

position_text = st.selectbox("Position",np.concatenate([["All"],data["Position"].unique()]),key="text1")
ad_state_text = st.selectbox("Ad State",np.concatenate([["All"],data["AdState"].unique()]),key="text2")
campaign_text = st.selectbox("Campaign Market",np.concatenate([["All"],data["CampaignMarket"].unique()]),key="text3")
cpl_text = st.slider("CPL",data["CPL"].min(),data["CPL"].max(),(data["CPL"].min(),data["CPL"].max()),key="text4")

text_query = []
if position_text != "All":
    text_query.append("Position==@position_text")
if ad_state_text != "All":
    text_query.append("AdState==@ad_state_text")
if campaign_text != "All":
    text_query.append("CampaignMarket==@campaign_text")
text_query.append("CPL>=@cpl_text[0] & CPL<=@cpl_text[1]")

text_query = " & ".join(text_query)
df_filte_text = data.query(text_query)

adtitle_analyze = st.button("Analyze Ad Title")

if adtitle_analyze:
    # Assuming df is your DataFrame and it has columns 'AdTitle' and 'CPL'
    df_filte_text['AdTitle'] = df_filte_text['AdTitle'].str.lower() # Convert to lowercase

    X_text = df_filte_text['AdTitle']
    y_text = df_filte_text['CPL']

    # Create a pipeline that converts words to TF-IDF vectors, then applies linear regression
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english')),
        ('clf', LinearRegression()),
    ])

    # Train the model
    pipeline.fit(X_text, y_text)

    # Now you can inspect the coefficients of the model
    coef = pipeline.named_steps['clf'].coef_

    # Get the feature names (words)
    words = pipeline.named_steps['tfidf'].get_feature_names()

    # Create a DataFrame with words and coefficients
    word_coef = pd.DataFrame({'word': words, 'coef': coef})

    # Sort the DataFrame by coefficient value in descending order
    word_coef = word_coef.sort_values(by='coef', ascending=False)
    word_coef.columns=["Words","Coefficients"]

    st.dataframe(word_coef)


