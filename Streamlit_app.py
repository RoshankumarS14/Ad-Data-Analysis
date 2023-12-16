import streamlit as st
import pandas as pd
import numpy as np
from category_encoders import TargetEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import plotly.graph_objects as go
import plotly.express as px 
from sklearn.feature_selection import SelectFromModel

st.set_page_config(
    page_title="Ad data Analysis",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed"
)

data = pd.read_excel("HeadlineData.xlsx")
data = data[(data.Position=="Driver / Drivers") |  (data.Position=="Local City Driver/Forklift Operators") | (data.Position=="HE - Drivers")]
data.drop(["Position"],axis=1,inplace=True)
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
ad_state = st.selectbox("Ad State",np.concatenate([["All"],df["AdState"].unique()]))
campaign = st.selectbox("Campaign Market",np.concatenate([["All"],df["CampaignMarket"].unique()]))
cpl = st.slider("CPL",df["CPL"].min(),df["CPL"].max(),(25.0,500.0))

query = []
if ad_title != "All":
    query.append("AdTitle==@ad_title")
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

cat_cols = ["CampaignMarket","AdState","AdTitle"]  # replace with your actual column names

encoder = TargetEncoder()
values = encoder.fit_transform(df[cat_cols], df['CPL'])
df = df.drop(columns=cat_cols)
df = pd.concat([df, values], axis=1)

X = df[cat_cols]
y = df["CPL"]
gb = GradientBoostingRegressor().fit(X,y)

input = encoder.transform(pd.DataFrame({"CampaignMarket":campaign,"AdState":ad_state,"AdTitle":ad_title},index=[0]))

if predict:
    st.subheader("Prediction results")
    st.write("Campaign Market : "+campaign)
    st.write("Ad State : "+ad_state)
    st.write("Position : Driver")
    st.write("Ad Title : "+ad_title)
    st.write("CPL : "+str(gb.predict(input)[0]))

st.header("Analysis of impact of Ad Title on CPL")

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

ad_state_text = st.selectbox("Ad State",np.concatenate([["All"],data["AdState"].unique()]),key="text2")
campaign_text = st.selectbox("Campaign Market",np.concatenate([["All"],data["CampaignMarket"].unique()]),key="text3")
cpl_text = st.slider("CPL",data["CPL"].min(),data["CPL"].max(),(data["CPL"].min(),data["CPL"].max()),key="text4")

text_query = []
if ad_state_text != "All":
    text_query.append("AdState==@ad_state_text")
if campaign_text != "All":
    text_query.append("CampaignMarket==@campaign_text")
text_query.append("CPL>=@cpl_text[0] & CPL<=@cpl_text[1]")

text_query = " & ".join(text_query)
df_filte_text = data.query(text_query)

if len(df_filte_text)==0:
    st.write("Applied filter has no records in the table")

adtitle_analyze = st.button("Generate report")

# Assuming df is your DataFrame and it has columns 'AdTitle' and 'CPL'
df_filte_text['AdTitle'] = df_filte_text['AdTitle'].str.lower() # Convert to lowercase

X_text = df_filte_text['AdTitle']
y_text = df_filte_text['CPL']

# Create a pipeline that converts words to TF-IDF vectors, then applies linear regression
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('feature_selection', SelectFromModel(LinearRegression(), threshold=1e-5)),
    ('clf', LinearRegression()),
])

# Train the model
pipeline.fit(X_text, y_text)

# Now you can inspect the coefficients of the model
coef = pipeline.named_steps['clf'].coef_
intercept = pipeline.named_steps['clf'].intercept_

# Get the feature names (words)
words = pipeline.named_steps['tfidf'].get_feature_names_out()

# Create a DataFrame with words and coefficients
word_coef = pd.DataFrame({'word': words, 'coef': coef})

# Sort the DataFrame by coefficient value in descending order
word_coef = word_coef.sort_values(by='coef')
word_coef.columns=["Words","Coefficients"]

if adtitle_analyze:
    st.dataframe(word_coef)

st.header("Text Analysis of Ad Title")

st.write("""
Welcome to our Ad Title Analysis tool! This component employs Natural Language Processing (NLP) techniques to evaluate ad titles entered by users. By analyzing individual words within the title, it generates a rating on a scale of 0 to 100. A higher rating suggests a potentially lower Cost Per Lead (CPL), while a lower rating implies a higher CPL.
                  
**Instructions** 
1. **Enter Ad Title:** Input your ad title into the provided text field. This title will undergo NLP analysis to generate the CPL rating.
2. **Initiate Analysis:** Trigger the analysis process after entering the ad title. The NLP algorithm will scrutinize each word within the title to compute a CPL rating based on its content and language characteristics.
3. **Review CPL Rating:** Once the analysis is complete, the tool will display a CPL rating between 0 and 100. A higher rating indicates that the words used in the ad title are potentially associated with a lower CPL, while a lower rating suggests the opposite.         
4. **Interpret the Rating:** Use the generated CPL rating as a guideline. Higher-rated ad titles might correlate with better CPL outcomes, while lower-rated titles might need optimization to improve CPL performance.
5. **Modify and Experiment:** Try altering the ad title and observe how it affects the CPL rating. Experimentation allows you to refine the title for potentially improved CPL outcomes.

**Apply Insights:** Utilize the insights derived from the CPL rating to optimize your ad titles. Adjust the wording and structure to potentially enhance CPL performance based on the analysis.""")

title = st.text_input("Ad Title")
adtitle_rate = st.button("Rate your Ad Title!")

if adtitle_rate:
    predicted_cpl = pipeline.predict([title])
    # st.write("Predicted CPL with the entered Ad Title: ",str(predicted_cpl[0]))
    if predicted_cpl==intercept:
        rating=50
    elif predicted_cpl>100:
        rating=0
    elif predicted_cpl<0:
        rating=100
    else:
        rating = int(100-predicted_cpl)

    st.write("Ad Title Rating : ",str(rating),"/100")
    

    # Define your values
    current_price = rating
    ask_price = 100
    bid_price = 0
    spread = 10

    # Create the gauge chart
    fig = go.Figure()

    fig.add_trace(
        go.Indicator(
            mode="gauge+number+delta",
            title={'text': "Rating"},
            delta={'reference': ask_price, 'relative': False, 'increasing': {'color': "RebeccaPurple"}, 'decreasing': {'color': "RoyalBlue"}},
            value=current_price,
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'shape': 'angular',
                'axis': {'range': [bid_price - spread, ask_price + spread]},
                'bar': {'color': "darkblue"},
                'bgcolor': 'yellow',
                'borderwidth': 2,
                'bordercolor': 'black',
                'steps': [
                    {'range': [80, 100], 'color': 'green'},
                    {'range': [0, 20], 'color': 'red'}
                ],
                'threshold': {
                    'line': {'color': 'orange', 'width': 6},
                    'thickness': 0.75,
                    'value': current_price,
                }
            }
        )
    )

    # Display the chart in Streamlit
    st.plotly_chart(fig, use_container_width=True)

st.header("Data Visualization")

st.write("""
Welcome to our Data Visualization Tool! This Streamlit component is designed to provide insightful visualizations based on your dataset. You can explore and interact with various charts and graphs to gain a better understanding of your data. Additionally, the tool offers filtering options, allowing you to refine the displayed visualizations based on specific criteria.
                           
**Instructions** 
1. **Explore Visualization Options:** Explore the available visualization options. You might find charts like bar charts, pie charts.
2. **Apply Filters (Optional):** Use the filtering options provided to refine the data displayed in the visualizations. Filters could include selecting specific columns, setting value ranges, or applying categorical filters to focus on particular subsets of the data.
3. **Interact with Visualizations:** Engage with the visualizations by hovering over data points. This interaction can reveal specific details within the visual representation of your data.
4. **Save or Export Results:** All the charts offer options to save or export the visualizations. This could be helpful for presentations, reports, or further analysis outside of the tool.       

**Apply Insights:** Utilize the insights derived from the CPL rating to optimize your ad. 
       """)


ad_state_chart = st.selectbox("Ad State",np.concatenate([["All"],data["AdState"].unique()]),key="key 5")
campaign_chart = st.selectbox("Campaign Market",np.concatenate([["All"],data["CampaignMarket"].unique()]),key="key 6")
cpl_chart = st.slider("CPL",data["CPL"].min(),data["CPL"].max(),(25.0,500.0),key="key 7")

chart_query = []
if ad_state_chart != "All":
    chart_query.append("AdState==@ad_state_chart")
if campaign_chart != "All":
    chart_query.append("CampaignMarket==@campaign_chart")
chart_query.append("CPL>=@cpl_chart[0] & CPL<=@cpl_chart[1]")

chart_query = " & ".join(chart_query)
df_filter_chart = data.query(chart_query)

min_cpl = df_filter_chart["CPL"].min()
max_cpl = df_filter_chart["CPL"].max()
avg_cpl = round(df_filter_chart["CPL"].mean(),2)
no_of_jobs = len(df_filter_chart)

c1,c2,c3 = st.columns(3)

with c1:
    st.subheader("Average CPL:")
    st.subheader(f"US $ {avg_cpl:,}")
with c2:
    st.subheader("Minimum CPL:")
    st.subheader(f"US $ {min_cpl:,}")
with c3:
    st.subheader("Maximum CPL:")
    st.subheader(f"US $ {max_cpl:,}")

st.markdown("---")

df_cpl_state = pd.pivot_table(data=df_filter_chart,index="AdState",values="CPL").sort_values("CPL")
df_cpl_campaign = pd.pivot_table(data=df_filter_chart,index="CampaignMarket",values="CPL").sort_values("CPL")

if len(df_filter_chart)==0:
    st.warning("The applied filter has no records!")
    
else:
    
    fig_cpl_state = px.bar(
            df_cpl_state,
            x=df_cpl_state.index,
            y="CPL",
            orientation="v",
            title = "<b>Average CPL of all states</b>",
            color_discrete_sequence=["#0083B8"]*len(df_cpl_state),
            template = "plotly_white",
        )
    
    fig_cpl_campaign = px.bar(
        df_cpl_campaign,
        x=df_cpl_campaign.index,
        y="CPL",
        orientation="v",
        title = "<b>Average CPL of all Campaign Markets</b>",
        color_discrete_sequence=["#0083B8"]*len(df_cpl_state),
        template = "plotly_white",
    )

    c4,c5 = st.columns(2)

    with c4:
        st.plotly_chart(fig_cpl_state)
    
    with c5:
        st.plotly_chart(fig_cpl_campaign)

    st.markdown("---")

    c6,c7 = st.columns(2)

    with c6:
        st.subheader("Ad State Distribution")
        fig = px.pie(df_filter_chart, values = "CPL", names = "AdState", hole = 0.5)
        fig.update_traces(text = df_filter_chart["AdState"], textposition = "outside")
        st.plotly_chart(fig,use_container_width=True)

    with c7:
        st.subheader("Campaign Market Distribution")
        fig = px.pie(df_filter_chart, values = "CPL", names = "CampaignMarket", hole = 0.5)
        fig.update_traces(text = df_filter_chart["CampaignMarket"], textposition = "outside")
        st.plotly_chart(fig,use_container_width=True)
