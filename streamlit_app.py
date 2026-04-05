import streamlit as st
import pandas as pd
import PyPDF2
from textblob import TextBlob
import plotly.express as px
import plotly.graph_objects as go
from collections import Counter 
import re

st.set_page_config(page_title="Sentiment Analyzer", layout="wide")
st.title("📊 Sentiment Analysis App")
uploaded_file = st.file_uploader("Upload a PDF file", type=['pdf'])

def extract_text_from_pdf(pdf_file):
        pdf_reader = PyPDF2.PdfReader(pdf_file)
        text=""
        for page in pdf_reader.pages:
            text += page.extract_text()
        return text

def get_sentiment(text):
        analysis = TextBlob(text)
        polarity = analysis.sentiment.polarity
        if polarity > 0.1:
            return "Positive",polarity
        elif polarity < -0.1:
            return "Negative",polarity
        else:
            return "Neutral",polarity
        
def split_reviews(text):
     reviews=re.split(r'\n\s*\n', text)
     reviews= [r.strip() for r in reviews if r.strip() and len(r.strip())>20]
     return reviews

if uploaded_file is not None:
     text_content=extract_text_from_pdf(uploaded_file)

     reviews=split_reviews(text_content)
     st.success(f"Extracted {len(reviews)} reviews from the PDF file.")
     results=[]
     for i,review in enumerate(reviews):
         sentiment,polarity=get_sentiment(review)
         results.append({
        "Review_id":i+1,
        "Review":review[:200] + '...' if len(review)>200 else review,
        "Full_review":review,
        "Sentiment":sentiment,
        "Polarity":round(polarity,2)
})
     df=pd.DataFrame(results)
     col1,col2,col3=st.columns(3)

     positive_count=len(df[df['Sentiment']=='Positive'])
     negative_count=len(df[df['Sentiment']=='Negative'])
     neutral_count=len(df[df['Sentiment']=='Neutral'])  

     with col1:
          st.metric("positive reviews",positive_count,f"{(positive_count/len(df)*100):.2f}%")

     with col2:
          st.metric("negative reviews",negative_count,f"{(negative_count/len(df)*100):.2f}%")

     with col3:
         st.metric("neutral reviews",neutral_count,f"{(neutral_count/len(df)*100):.2f}%")

     st.divider()

     col1,col2=st.columns(2)
     
     sentiment_counts=df['Sentiment'].value_counts()

     with col1:
           fig_pie=px.pie(values=sentiment_counts.values, names=sentiment_counts.index,title='Sentiment Distribution',color=sentiment_counts.index,color_discrete_map={
                'Positive':'#00cc99','Negative':'#ff6b6b','Neutral':'#4ecdc4'})
           st.plotly_chart(fig_pie,use_container_width=True)

     with col2:
          fig_bar=px.bar(x=sentiment_counts.index,y=sentiment_counts.values,title='Count of Sentiments',color=sentiment_counts.index,color_discrete_sequence=['#00cc99','#ff6b6b','#4ecdc4'])
          st.plotly_chart(fig_bar,use_container_width=True)

     st.divider()

     col1,col2=st.columns(2)

     with col1:
          avg_polarity=df['Polarity'].mean()  
          st.metric("Average Polarity", round(avg_polarity, 3))

     with col2:
          most_common=df['Sentiment'].mode()[0]
          st.metric("Most common Sentiment",most_common)

     st.divider()

     fig_hist=px.histogram(df, x='Polarity', nbins=30, title='Distribution of Polarity Scores')
     st.plotly_chart(fig_hist, use_container_width=True)

     st.divider()

     st.subheader("All Reviews")

     filter_sentiment=st.multiselect("Filter by sentiment ",options=['Positive','Negative','Neutral'],default=['Positive','Negative','Neutral'])
     filtered_df=df[df['Sentiment'].isin(filter_sentiment)]

     st.dataframe(filtered_df[['Review_id','Review','Sentiment','Polarity']],use_container_width=True,height=400)

     st.divider()

     st.subheader("Individual review analyst")
     review_id=st.selectbox("select review id", options=df['Review_id'].tolist())

     selected_review=df[df['Review_id']==review_id].iloc[0]
     st.write(f"**Review ID:** {selected_review['Review_id']}")
     st.write(f"**Sentiment:** {selected_review['Sentiment']}")
     st.write(f"**Polarity:** {selected_review['Polarity']}")
     st.write(f"**Review:** {selected_review['Full_review']}")

     st.divider()

     csv=df.to_csv(index=False)
     st.download_button(label="Download analysis as csv",data=csv,file_name="sentiment_analysis_result.csv",mime="text/csv")

else:
     st.info("please upload a pdf file to start analysis")

st.markdown("""
1. Upload a PDF file with product reviews
2. The app will automatically extract and analyze each review
3. View sentiment distribution and detailed statistics
4. Download results as CSV

Sentiment classification (Positive/Negative/Neutral)
Polarity score calculation
Interactive visualizations
Detailed review breakdown
Export functionality
""")
#python -m streamlit run streamlit_app.py