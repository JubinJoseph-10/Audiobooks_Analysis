#import required libraries
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,confusion_matrix
import plotly.graph_objects as go
from PIL import Image
from sklearn.preprocessing import StandardScaler

ico = Image.open("Data/Audible.png")
#Setting page configs
st.set_page_config(layout='wide',page_title='Audiobook Analysis',initial_sidebar_state='collapsed',page_icon=ico)

#Setting the title
st.title('Audiobooks User Analyis')
req_data = pd.read_csv("Data/Audible_Dashboard_Data.csv")

#########################################################################################################################################################################################################################################################################################################################################################
#########################################################################################################################################################################################################################################################################################################################################################
######################################################################################################################################################################################################################################################################################################################################################### Habbits Across Users 
top_section = st.container(border=True)
state_photo,further = top_section.columns([.27,.73])
state_photo_ = state_photo.container(border=True) 
further_ = further.container(border=True)
further_1,further_2,further_3 = further_.columns([.33,.33,.33])


#select boxes for the variables that would be a part of the chart
state_name = further_1.selectbox('Select a State!',req_data['City'].unique(),key=69)
demograph = further_2.selectbox('Select a Demographic Variable!',['Age_Group','Gender','Commuting_Mode'],key=70)
demograph_specifc = further_3.selectbox(f'Select a {demograph}!',req_data[demograph].unique(),key=71)

top_data_city = req_data[req_data['City']==state_name]
top_data_city_req = top_data_city[top_data_city[demograph]==demograph_specifc]


image_loc = 'Data/{}.png'.format(state_name)
state_image = Image.open(image_loc)
state_photo_.image(image_loc,use_column_width=True)

###definition for data returner
def basis_data_sorter(nature,base):
  if base == 'Number of Listeners':
    apt_data = pd.DataFrame(top_data_city_req.groupby([nature])['Ref ID'].count().sort_values(ascending=False).head(5)).reset_index().rename(columns={'Ref ID':'Number of Users'})
    return apt_data
  elif base == 'Number of Reviews':
    apt_data = pd.DataFrame(top_data_city_req.groupby([nature])['Reviews_Written'].sum().sort_values(ascending=False).head(5)).reset_index().rename(columns={'Reviews_Written':'Number of Reviews'})
    return apt_data
  elif base == 'Best Ratings':
    apt_data = pd.DataFrame(top_data_city_req.groupby([nature])['Ratings_Given'].mean().round(2).sort_values(ascending=False).head(5)).reset_index().rename(columns={'Ratings_Given':'Average Rating'})
    return apt_data
  elif base == 'Most Shared':
    apt_data = pd.DataFrame(top_data_city_req.groupby([nature])['Social_Sharing'].sum().sort_values(ascending=False).head(5)).reset_index().rename(columns={'Social_Sharing':'Times Shared'})
    return apt_data


###Top audiobooks section

graphs_1 = top_section.container(border=True)
further_top_audiobooks,further_top_narrators = graphs_1.columns([.5,.5])
graphs_2 = top_section.container(border=True)
further_top_genre,further_top_publishers = graphs_2.columns([.5,.5])


#####further_top_audiobooks_
further_top_audiobooks_ = further_top_audiobooks.container(border=True)
basis_top_audiobooks = further_top_audiobooks_.selectbox('Select a Basis for Choosing Top Audiobooks',['Number of Listeners','Number of Reviews','Best Ratings','Most Shared'],key=89)
data_top_audiobooks = basis_data_sorter('Product_Name',basis_top_audiobooks)
top_audio_chart = px.bar(data_top_audiobooks,x='Product_Name',barmode='group',color='Product_Name',
                        y=data_top_audiobooks.columns[1],title=f'Top Audiobooks based on {data_top_audiobooks.columns[1]}',height=300,width=450)
top_audio_chart.update_layout(showlegend=False)
top_audio_chart.update_xaxes(showticklabels=False, title='')
further_top_audiobooks_.plotly_chart(top_audio_chart,use_column_width=True)


###further_top_narrators_
further_top_narrators_ = further_top_narrators.container(border=True)
basis_top_author = further_top_narrators_.selectbox('Select a Basis for Choosing Top Narrators',['Number of Listeners','Number of Reviews','Best Ratings','Most Shared'],key =49)
data_top_authors = basis_data_sorter('Product_Narrator',basis_top_author)
top_author_chart = px.bar(data_top_authors,x='Product_Narrator',barmode='group',color='Product_Narrator',
                        y=data_top_authors.columns[1],title=f'Top Audiobooks based on {data_top_authors.columns[1]}',height=300,width=450)
further_top_narrators_.plotly_chart(top_author_chart,use_column_width=True)

###further_top_genre_
further_top_genre_ = further_top_genre.container(border=True)
basis_top_genre = further_top_genre_.selectbox('Select a Basis for Choosing Top Genres',['Number of Listeners','Number of Reviews','Best Ratings','Most Shared'],key= 999)
data_top_genre = basis_data_sorter('Genre',basis_top_genre)
top_genre_chart = px.bar(data_top_genre,x='Genre',barmode='group',color='Genre',
                        y=data_top_genre.columns[1],title=f'Top Audiobooks based on {data_top_genre.columns[1]}',height=400,width=450)
further_top_genre_.plotly_chart(top_genre_chart,use_column_width=True)



further_top_publishers_ =  further_top_publishers.container(border=True)
basis_top_pub = further_top_publishers_.selectbox('Select a Basis for Choosing Top Publishers',['Number of Listeners','Number of Reviews','Best Ratings','Most Shared'],key =99)
data_top_pub = basis_data_sorter('Publisher_Name',basis_top_pub)
top_pub_chart = px.bar(data_top_pub,x='Publisher_Name',barmode='group',color='Publisher_Name',
                        y=data_top_pub.columns[1],title=f'Top Audiobooks based on {data_top_pub.columns[1]}',height=400,width=450)
top_pub_chart.update_layout(showlegend=False)
top_pub_chart.update_xaxes(showticklabels=False, title='')
further_top_publishers_.plotly_chart(top_pub_chart,use_column_width=True)


#######################################################################################
#######################################################################################
further_.divider()
top_book = data_top_audiobooks.iloc[0,0]
top_narrator = data_top_authors.iloc[0,0]
top_genre = data_top_genre.iloc[0,0]
top_pub = data_top_pub.iloc[0,0]
further_.markdown('<div style="text-align: justify; font-size: 14px">1. The most heard book in {} among users with {} as {} based on {} is <b>{}<b>.</div>'.format(state_name,demograph,demograph_specifc,basis_top_audiobooks,top_book),unsafe_allow_html=True) 
further_.write('\n')
further_.markdown('<div style="text-align: justify; font-size: 14px">2. The most most heard Narrator in {} among users with {} as {} based on {} is <b>{}<b>.</div>'.format(state_name,demograph,demograph_specifc,basis_top_author,top_narrator),unsafe_allow_html=True) 
further_.write('\n')
further_.markdown('<div style="text-align: justify; font-size: 14px">3. The most heard genre in {} among users with {} as {} based on {} is <b>{}<b>.</div>'.format(state_name,demograph,demograph_specifc,basis_top_genre,top_genre),unsafe_allow_html=True) 
further_.write('\n')
further_.markdown('<div style="text-align: justify; font-size: 14px">4. The most prominent publisher in {} among users with {} as {} based on {} is <b>{}<b>.</div>'.format(state_name,demograph,demograph_specifc,basis_top_pub,top_pub),unsafe_allow_html=True) 
further_.write('\n')




#########################################################################################################################################################################################################################################################################################################################################################
#########################################################################################################################################################################################################################################################################################################################################################
######################################################################################################################################################################################################################################################################################################################################################### Habbits Across Users 

#structuring the space for the analysis
demo_listening_habs = st.container(border=True)
demo_listening_habs.markdown('<div style="text-align: center; font-size: 24px">Impact of Demographics on User Habbits & Preferences</div>',unsafe_allow_html=True)
demo_listening_habs.write('\n')
demo_listening_habs_chart_space ,demo_listening_habs_des = demo_listening_habs.columns([.7,.3])
demo_listening_habs_des_ = demo_listening_habs_des.container(border =True)
demo_listening_habs_chart_space_ = demo_listening_habs_chart_space.container(border =True)

demo_listening_habs_des_.markdown('<div style="text-align: justify; font-size: 18px">User Habbits</div>',unsafe_allow_html=True)
demo_listening_habs_des_.write('\n')
demo_listening_habs_des_.markdown('<div style="text-align: justify; font-size: 14px"> Understanding the influence of demographics, including age, gender, and city, on audiobook listening habits and preferences is crucial in tailoring content and enhancing user experience.</div>',unsafe_allow_html=True)
demo_listening_habs_des_.write('\n')


#select boxes for the variables that would be a part of the chart
demo_Var_ = demo_listening_habs_des_.selectbox('Select a Demographic Variable to Study!',['Age_Group','City','Gender','Commuting_Mode'],key=2)
eval_Var_ = demo_listening_habs_des_.selectbox('Select a Variable to Study Habbits!',['No_of_Books_read_in_a_year_Number','Event_Duration_Minutes_2_Weeks','Event_Duration_Minutes_5_Weeks',
                                                'Product_Running_Time','Completion_Rate_5_Weeks','Membership_Duration','Number_of_Audiobooks_Completed','Number_of_Audiobooks_Purchased','Time_Spent_Browsing','Genre_Exploration',
                                                'Engagement_Rate','Listening_Speed_Numeric'])
demo_listening_habs_des_.write('\n')

#creating the datset that would be displayed
demo_eval = req_data.groupby([demo_Var_])[eval_Var_].mean().round(2)
demo_eval=pd.DataFrame(demo_eval)
demo_eval.reset_index(inplace=True)


habbit_chart = px.bar(demo_eval,color=demo_Var_,x=demo_Var_,
                        y=eval_Var_,title=f'Average {eval_Var_} over {demo_Var_}')
demo_listening_habs_chart_space_.plotly_chart(habbit_chart)


##### Habbits Across Users 

#structuring the space for the analysis
demo_listening_pref_des , demo_listening_pref_chart_space = demo_listening_habs.columns([.3,.7])
demo_listening_pref_des_ = demo_listening_pref_des.container(border =True)
demo_listening_pref_chart_space_ = demo_listening_pref_chart_space.container(border =True)


demo_listening_pref_des_.markdown('<div style="text-align: justify; font-size: 18px">User Preferences</div>',unsafe_allow_html=True)
demo_listening_pref_des_.write('\n')
demo_listening_pref_des_.markdown('<div style="text-align: justify; font-size: 14px">By analyzing these factors, we can gain insights into the diverse needs and behaviors of audiobook consumers across different segments. This section sets the stage for exploring the nuanced relationship between demographics and audiobook consumption patterns.</div>',unsafe_allow_html=True)
demo_listening_pref_des_.write('\n')


#select boxes for the variables that would be a part of the chart
demo_Var_pref = demo_listening_pref_des_.selectbox('Select a Demographic Variable to Study!',['Age_Group','City','Gender','Commuting_Mode'])
pref_Var_ = demo_listening_pref_des_.selectbox('Select a Variable to Study Preferences!',['Download_vs_Streaming','Language_Preference','Listening_Device_Preference',
                                                'Listening_Context','Preferred_Listening_Time','Subscription_Type','Average_Listening_Speed'])                                       
demo_listening_pref_des_.write('\n')

#creating the datset that would be displayed
pref_eval = req_data.groupby([demo_Var_pref,pref_Var_])['Ref ID'].count()
pref_eval=pd.DataFrame(pref_eval)
pref_eval.reset_index(inplace=True)
pref_eval.rename(columns={'Ref ID':'Number of Users'},inplace=True)

pref_chart = px.bar(pref_eval,color=pref_Var_,x=demo_Var_pref,barmode='group',
                        y='Number of Users',title=f'Preference Trends about {pref_Var_} over {demo_Var_pref}')
demo_listening_pref_chart_space_.plotly_chart(pref_chart)


########## Correaltion for commuting patterns and audiobook consumption

#selecting variables for the heatmap
corr_data = pd.DataFrame()
corr_data = pd.concat([corr_data,pd.get_dummies(req_data['Commuting_Mode'])],axis=1)
corr_data['Commuting_Time_Minutes'] = req_data['Commuting_Time_Minutes']
corr_data['Completion_Rate_5_Weeks'] = req_data['Completion_Rate_5_Weeks']
corr_data['Event_Duration_Minutes_2_Weeks'] = req_data['Event_Duration_Minutes_2_Weeks']
corr_data['Event_Duration_Minutes_3_Weeks'] = req_data['Event_Duration_Minutes_3_Weeks']
corr_data['Event_Duration_Minutes_5_Weeks'] = req_data['Event_Duration_Minutes_5_Weeks']
corr_data['Engagement_Rate'] = req_data['Engagement_Rate']
corr_data['Number_of_Audiobooks_Completed'] = req_data['Number_of_Audiobooks_Completed']
corr_data['Number_of_Audiobooks_Purchased'] = req_data['Number_of_Audiobooks_Purchased']
corr_data['No_of_Books_read_in_a_year_Number'] = req_data['No_of_Books_read_in_a_year_Number']
corr_data['Membership_Duration'] = req_data['Membership_Duration']
corr_data['Product_Running_Time'] = req_data['Product_Running_Time']


#plotting the correlation heatmap
demo_commute_corr = demo_listening_habs.container(border=True)  
heatmap = px.imshow(corr_data.corr(method='spearman').round(3),labels=dict(x="Features", y="Features", color="Correlation"),aspect=3,text_auto=True,color_continuous_scale ='jet',title='Correlation Heatmap for all the Variables Related to Commuting and Audiobook Consumption',height=700)
demo_commute_corr.plotly_chart(heatmap,use_container_width=True)



##### key takeaways from the analysis on user preferences and habits given demographics
user_hab_pref_analysis = demo_listening_habs.container(border=True) 
user_hab_pref_analysis.markdown('<div style="text-align: center; font-size: 24px">Analysis & Results for User Demographics, Habbits and Preferences</div>',unsafe_allow_html=True)
user_hab_pref_analysis.write('\n')
user_hab_pref_analysis.markdown('<div style="text-align: justify; font-size: 14px">1. Comparative analysis of different age groups along with user habit suggest that <i><b>the late adult population (over 50 years) form the biggest groups of users who consistently use the platform through their membership and read the greatest number of books. However, it must be noted that these people prefer shorter books (with a lesser PRT) as compared to younger adults and middle adults and are considerably conservative in exploring genres.<b></i></div>',unsafe_allow_html=True)
user_hab_pref_analysis.write('\n')
user_hab_pref_analysis.markdown('<div style="text-align: justify; font-size: 14px">2. While the <i><b>greatest number of books are being read and completed in Delhi, South Indian cities like Bangalore, Hyderabad and Chennai have the greatest number of readers who consistently listen over a 5-week period and have the longest duration of memberships.<b></i></div>',unsafe_allow_html=True)
user_hab_pref_analysis.write('\n')
user_hab_pref_analysis.markdown('<div style="text-align: justify; font-size: 14px">3. An analysis across genders suggests both genders follow similar trends in terms of engagement rate, genre exploration, time spent browsing, number of books completed and so on. <i><b>Women take the lead in their consistency of reading over a period of 5 weeks and for the number of books read.<b></i></div>',unsafe_allow_html=True)
user_hab_pref_analysis.write('\n')
user_hab_pref_analysis.markdown('<div style="text-align: justify; font-size: 14px">4. Bivariate analysis of commuting and general travel patterns is consistent and shows that <i><b>people generally prefer to listen to audiobooks while travelling in Public Transport, 4 wheelers (shared, pooled, personal or otherwise), and while walking.<b></i> Data shows people read longer materials while walking and have the highest number of completed books in the ‘other’ category.</div>',unsafe_allow_html=True)
user_hab_pref_analysis.write('\n')


#########################################################################################################################################################################################################################################################################################################################################################
#########################################################################################################################################################################################################################################################################################################################################################
######################################################################################################################################################################################################################################################################################################################################################### Audiobook Usage based on the nature of technology used
#structuring the space for the analysis
dev_pref_habbits = st.container(border=True)
dev_pref_habbits.markdown('<div style="text-align: center; font-size: 24px">Trends Related to Device and Technology Usage</div>',unsafe_allow_html=True)
dev_pref_habbits.write('\n')
dev_pref_habbits_des , dev_pref_chart_space = dev_pref_habbits.columns([.3,.7])
dev_pref_habbits_des_ = dev_pref_habbits_des.container(border =True)
dev_pref_chart_space_ = dev_pref_chart_space.container(border =True)


dev_pref_habbits_des_.markdown('<div style="text-align: justify; font-size: 18px">Choice of Technology and Audiobook Usage!</div>',unsafe_allow_html=True)
dev_pref_habbits_des_.write('\n')
dev_pref_habbits_des_.markdown('<div style="text-align: justify; font-size: 14px">This section delves into the impact of technology choices on audiobook usage and content delivery strategies to shed light on evolving consumer preferences and market dynamics. By examining factors such as the choice between Android and iOS devices, as well as preferences for smartphone versus tablet usage, we gain insights on how these decisions influence audiobook consumption habits.</div>',unsafe_allow_html=True)
dev_pref_habbits_des_.write('\n')

#select boxes for the variables that would be a part of the chart
tech_var_pref = dev_pref_habbits_des_.selectbox('Select a Technology Related Variable to Study!',['Smart_Phone','Tablet','Device_Type','Listening_Device_Preference'])
tech_pref_Var_ = dev_pref_habbits_des_.selectbox('Select a Variable to Study Usage!',['No_of_Books_read_in_a_year_Number','Event_Duration_Minutes_2_Weeks','Event_Duration_Minutes_5_Weeks',
                                                'Product_Running_Time','Completion_Rate_5_Weeks','Membership_Duration','Number_of_Audiobooks_Completed','Number_of_Audiobooks_Purchased','Time_Spent_Browsing','Genre_Exploration','Engagement_Rate','Listening_Speed_Numeric'])

dev_pref_habbits_des_.write('\n')

#creating the datset that would be displayed
tech_eval = req_data.groupby([tech_var_pref])[tech_pref_Var_].mean().round(2)
tech_eval=pd.DataFrame(tech_eval)
tech_eval.reset_index(inplace=True)

tech_usage_chart = px.bar(tech_eval,color=tech_var_pref,x=tech_var_pref,
                        y=tech_pref_Var_,title=f'Average {tech_pref_Var_} over {tech_var_pref}')
dev_pref_chart_space_.plotly_chart(tech_usage_chart)



###### Audiobook Usage based on the nature of technology used
#structuring the space for the analysis
dev_pref_chart_,dev_pref_des = dev_pref_habbits.columns([.7,.3])
dev_pref_des_ = dev_pref_des.container(border =True)
dev_pref_chart_ = dev_pref_chart_.container(border =True)


dev_pref_des_.markdown('<div style="text-align: justify; font-size: 18px">Implication of users preferences on content delivery startegies</div>',unsafe_allow_html=True)
dev_pref_des_.write('\n')
dev_pref_des_.markdown('<div style="text-align: justify; font-size: 14px">Furthermore, exploring user preferences for downloading versus streaming unveils implications for content delivery strategies, informing decision-making processes for audiobook platforms. Funneling our strategies and collaborations further into fine tuning the user experience that we provide to the customers.</div>',unsafe_allow_html=True)
dev_pref_des_.write('\n')




#select boxes for the variables that would be a part of the chart
tech_var_pref = dev_pref_des_.selectbox('Select a Technology Related Variable to Study!',['Smart_Phone','Tablet','Device_Type','Listening_Device_Preference'],key=4)
tech_pref_Var = dev_pref_des_.selectbox('Select a Variable to Study Usage!',['Download_vs_Streaming','Language_Preference','Listening_Device_Preference',
                                                'Listening_Context','Preferred_Listening_Time','Subscription_Type','Average_Listening_Speed'],key=8)                                       

dev_pref_des_.write('\n')

#creating the datset that would be displayed
tech_pre_val = req_data.groupby([tech_var_pref,tech_pref_Var])['Ref ID'].count()
tech_pre_val=pd.DataFrame(tech_pre_val)
tech_pre_val.reset_index(inplace=True)
tech_pre_val.rename(columns={'Ref ID':'Number of Users'},inplace=True)

tech_pref_chart = px.bar(tech_pre_val,color=tech_pref_Var,x=tech_var_pref,barmode='group',
                        y='Number of Users',title=f'Preference Trends about {tech_pref_Var} over {tech_var_pref}')
dev_pref_chart_.plotly_chart(tech_pref_chart)


##### key takeaways from the analysis on user preferences and habits given demographics
tech_pref_analysis = dev_pref_habbits.container(border=True) 
tech_pref_analysis.markdown('<div style="text-align: center; font-size: 24px">Analysis & Results for Device and Technology Usage and its Implications!</div>',unsafe_allow_html=True)
tech_pref_analysis.write('\n')
tech_pref_analysis.markdown('<div style="text-align: justify; font-size: 14px">1. The data indicates <i><b>smartphones are the most preferred device type for audible listeners with a majority leaning towards android over IOS. However, the most consistent use of the platform is made by IOS users, who also have a higher engagement rate.</i><b></div>',unsafe_allow_html=True)
tech_pref_analysis.write('\n')
tech_pref_analysis.markdown('<div style="text-align: justify; font-size: 14px">2. Smartphones, as the preferred device type for audiobooks, was further substantiated by a <i><b>Market Research report by Polaris which stated audiobook markets will witness rapid expansion over, primarily attributed to its role in democratizing audiobook accessibility.</i><b></div>',unsafe_allow_html=True)
tech_pref_analysis.write('\n')
tech_pref_analysis.markdown('<div style="text-align: justify; font-size: 14px">3. There were only minute differences for a preference for downloading or streaming across different demographics, such as age, city, gender and type of commuting. However, when it came to the <i><b>preferred device and choice between downloading or streaming content, we see that users that prefer PCs and Smart Speakers tend to stream content more as opposed to users that prefer, phones and tablets who usually download content more. Audible for tablets and smartphones should include some pre installed/downloaded popular audiobooks to increase probability of engagement right off the bat.</i><b></div>',unsafe_allow_html=True)
tech_pref_analysis.write('\n')


#########################################################################################################################################################################################################################################################################################################################################################
#########################################################################################################################################################################################################################################################################################################################################################
######################################################################################################################################################################################################################################################################################################################################################### Top Genres

top_g = st.container(border=True)
top_g.markdown('<div style="text-align: center; font-size: 24px">Content Engagment and Preferences</div>',unsafe_allow_html=True)
top_g.write('\n')
top_g_chart, top_g_des = top_g.columns([.7,.3])
top_g_des_des_ = top_g_des.container(border =True)
top_g_chart_ = top_g_chart.container(border =True)

top_g_des_des_.markdown('<div style="text-align: justify; font-size: 18px">Analysing top Genres given Demographics!</div>',unsafe_allow_html=True)
top_g_des_des_.write('\n')
top_g_des_des_.markdown('<div style="text-align: justify; font-size: 14px"> This section delves into the analysis of popular genres among diverse user segments and offers insights into how Audible can adjust its content acquisition strategy accordingly. In this first analysis you can explore top genres across different demographic user segments and also decide the metric you would like to use in order to decide the top genres!</div>',unsafe_allow_html=True)
top_g_des_des_.write('\n')

#select boxes for the variables that would be a part of the chart
basis_top_g = top_g_des_des_.selectbox('Select a Demographic Variable to Study!',['Age_Group','City','Gender','Commuting_Mode'],key=64)
metric_ = top_g_des_des_.selectbox('Select a Metric to Evaluate!',['Number of Users','Listening_Speed_Numeric','Completion_Rate_5_Weeks'])

top_g_des_des_.write('\n')

def analyssis_sel(met):
    if met=='Number of Users':
    #creating the datset that would be displayed
        top_g_by_dem = req_data.groupby([basis_top_g,'Genre'])['Ref ID'].count()
        top_g_by_dem=pd.DataFrame(top_g_by_dem)
        top_g_by_dem.reset_index(inplace=True)
        top_g_by_dem.rename(columns={'Ref ID':'Number of Users'},inplace=True)
        return top_g_by_dem
    else:
        top_g_by_dem = req_data.groupby([basis_top_g,'Genre'])[metric_].mean()
        top_g_by_dem=pd.DataFrame(top_g_by_dem)
        top_g_by_dem.reset_index(inplace=True)
        return top_g_by_dem

top_g_by_dem = analyssis_sel(metric_)

yop_g_chart = px.bar(top_g_by_dem,color='Genre',x=basis_top_g,barmode='group',
                        y=metric_,title=f'Top Genres given {basis_top_g} & {metric_}')
top_g_chart_.plotly_chart(yop_g_chart)



############ Genre Space 
top_genres_ , top_genres_var = top_g.columns([.3,.7])
top_genres_var_ = top_genres_var.container(border=True)
top_genres__ = top_genres_.container(border=True)

top_genres__.markdown('<div style="text-align: justify; font-size: 18px">Analysing top Genres given KPIs of your choice!</div>',unsafe_allow_html=True)
top_genres__.write('\n')
top_genres__.markdown('<div style="text-align: justify; font-size: 14px">Furthermore, it explores the variations in book completion rates and listening speeds across different genres and demographics. In this second analysis within this section you are free to explore the Top Genres given a KPI of your choice and also want measure of KPI.</div>',unsafe_allow_html=True)
top_genres__.write('\n')

#select boxes for the variables that would be a part of the chart
#demo_sel = top_genres_var_.selectbox('Select a Demographic Related Variable to Study!',['Age_Group','City','Gender','Commuting_Mode'],key=16)
demo_eval_var = top_genres__.selectbox('Select a Basis to Decide Top Genre!',['No_of_Books_read_in_a_year_Number','Event_Duration_Minutes_2_Weeks','Event_Duration_Minutes_5_Weeks',
                                                'Product_Running_Time','Completion_Rate_5_Weeks','Membership_Duration','Number_of_Audiobooks_Completed','Number_of_Audiobooks_Purchased','Time_Spent_Browsing','Genre_Exploration','Engagement_Rate'])
demo_eval_var_basis = top_genres__.selectbox('Select an Aritmetic Basis to Decide Top Genre!',['Average of Values','Sum of Values'])

top_genres__.write('\n')

## funtion to return apprpriate df for plotting
def count_top_g(eval_):
    if eval_ =='Average of Values':
        top_g = pd.DataFrame(req_data.groupby(['Genre'])[demo_eval_var].mean().round(2)).reset_index()
        return top_g
    elif eval_ == 'Sum of Values':
        top_g = pd.DataFrame(req_data.groupby(['Genre'])[demo_eval_var].sum()).reset_index()
        return top_g
    
req_g_top = count_top_g(demo_eval_var_basis)

top_g_basis_chart = px.bar(req_g_top,color='Genre',x='Genre',
                        y=demo_eval_var,title=f'Average {demo_eval_var_basis} over Genres')
top_genres_var_.plotly_chart(top_g_basis_chart)


#####
genre_pref_chart , genre_pref_des = top_g.columns([.7,.3])
genre_pref_chart_ = genre_pref_chart.container(border=True)
genre_pref_des_ = genre_pref_des.container(border=True)


genre_pref_des_.markdown('<div style="text-align: justify; font-size: 18px">Analysing Underlying Preferences wihin Genres and Segments!</div>',unsafe_allow_html=True)
genre_pref_des_.write('\n')
genre_pref_des_.markdown('<div style="text-align: justify; font-size: 14px">By understanding these dynamics, Audible can tailor its offerings to better meet the preferences and behaviors of its user base, thereby enhancing user satisfaction and engagement. In this third section on genres, users can drill down further to see how user preferences might differ in a given genre that they are exploring.</div>',unsafe_allow_html=True)
genre_pref_des_.write('\n')


#select boxes for the variables that would be a part of the chart
genre_pref_var = genre_pref_des_.selectbox('Select a Variable to Study Preferences across Genres!',['Download_vs_Streaming','Language_Preference','Listening_Device_Preference',
                                                'Listening_Context','Preferred_Listening_Time','Subscription_Type','Average_Listening_Speed','Top 100 in Respective Genre'],key=32)                                       

genre_pref_des_.write('\n')

#creating the datset that would be displayed
g_pref = req_data.groupby(['Genre',genre_pref_var])['Ref ID'].count()
g_pref=pd.DataFrame(g_pref)
g_pref.reset_index(inplace=True)
g_pref.rename(columns={'Ref ID':'Number of Users'},inplace=True)

g_pref_chart = px.bar(g_pref,color=genre_pref_var,x='Genre',barmode='group',
                        y='Number of Users',title=f'Preference Trends about {tech_pref_Var} over Genres')
genre_pref_chart_.plotly_chart(g_pref_chart)


top_g_analysis_pref = top_g.container(border=True)
##### key takeaways from the analysis on user preferences and habits given demographics
top_g_analysis_pref.markdown('<div style="text-align: center; font-size: 24px">Analysis & Results for Top Genres and its Implications!</div>',unsafe_allow_html=True)
top_g_analysis_pref.write('\n')
top_g_analysis_pref.markdown('<div style="text-align: justify; font-size: 14px">1. The <i><b>most popular genres of books across the platform for various age groups, regions and genders are Literature and fiction, Parenting and Relationships, and Religion and Spirituality.</i><b></div>',unsafe_allow_html=True)
top_g_analysis_pref.write('\n')
top_g_analysis_pref.markdown('<div style="text-align: justify; font-size: 14px">2. While <i><b>people from ages 18-40 prefer to listen at an average speed of 1.5 (approximately) for various genres, older adults like 46 years and older tend to listen to religious and spiritual material at twice the speed while preferring mysteries and thrillers, lifestyle and biographies and memoirs at normal.</i><b></div>',unsafe_allow_html=True)
top_g_analysis_pref.write('\n')
top_g_analysis_pref.markdown('<div style="text-align: justify; font-size: 14px">3. In terms of completion rate, different age groups show different preferences such as, <i><b>50 years old and above tend to complete mysteries and thrillers whereas 36-40 years old tend to complete materials of “teen and young adult” genre and even more, 31- 35-year-olds show a higher completion rate for arts and entertainment.</i><b></div>',unsafe_allow_html=True)
top_g_analysis_pref.write('\n')
top_g_analysis_pref.markdown('<div style="text-align: justify; font-size: 14px">4. An interesting note shows that <i><b>middle adults (36-40) not only show a comparatively high preference for ‘parenting and relationship’ genre in tandem with a tendency for the highest listening speed (over 1.5 times) and the highest completion rate for ‘teen and young adult’ genres. Similarly, while the number of users for the lifestyle genre stems low, it is preferred quite high among both the genders.</i><b></div>',unsafe_allow_html=True)
top_g_analysis_pref.write('\n')
top_g_analysis_pref.markdown('<div style="text-align: justify; font-size: 14px">5. <i><b>Content acquisition strategies tailored for different age groups and genders should be genre specific. Alongside must ensure that the popular books are part of the catalouge for example Top 100 from books from NY Times for a genre.</i><b></div>',unsafe_allow_html=True)
top_g_analysis_pref.write('\n')
top_g_analysis_pref.markdown('<div style="text-align: justify; font-size: 14px">6. <i><b>Audible sgould prioritize and incentivize popular authors who have multiple genres, to promote genre exploration among users.</i><b></div>',unsafe_allow_html=True)
top_g_analysis_pref.write('\n')



#########################################################################################################################################################################################################################################################################################################################################################Correlation
#########################################################################################################################################################################################################################################################################################################################################################
#########################################################################################################################################################################################################################################################################################################################################################
corr_data_engagement_ = st.container(border=True)
corr_data_engagement_.markdown('<div style="text-align: center; font-size: 24px">Correlation between type of Subscription and User Enagagement and Retention</div>',unsafe_allow_html=True)
corr_data_engagement_.write('\n')

corr_data_engagement=pd.DataFrame()
corr_data_engagement = pd.get_dummies(req_data['Subscription_Type'])
corr_data_engagement['Engagemrnt Rate'] = req_data['Engagement_Rate']
corr_data_engagement['Retention'] = req_data['Users_Retained_5Weeks']
corr_data_engagement['Membership_Duration'] = req_data['Membership_Duration']
corr_data_engagement['Number_of_Audiobooks_Completed'] = req_data['Number_of_Audiobooks_Completed']
corr_data_engagement['Number_of_Audiobooks_Purchased'] = req_data['Number_of_Audiobooks_Purchased']


#plotting the correlation heatmap
heatmap_user_engage = px.imshow(corr_data_engagement.corr(method='spearman').round(3),labels=dict(x="Features", y="Features", color="Correlation"),aspect='auto',text_auto=True,color_continuous_scale ='jet',title='Correlation Heatmap for all the Variables Related to User Enagagement and Retention',height=500)
corr_engagement = corr_data_engagement_.container(border=True)
corr_engagement.plotly_chart(heatmap_user_engage,use_container_width=True)



corr_data_engagement_des,corr_data_engagement_chart = corr_data_engagement_ .columns([.3,.7])
corr_data_engagement_des_ = corr_data_engagement_des.container(border=True)
corr_data_engagement_chart_ = corr_data_engagement_chart.container(border=True) 

corr_data_engagement_des_.markdown('<div style="text-align: justify; font-size: 18px">Analysing Subscription Preferences over Demographics!</div>',unsafe_allow_html=True)
corr_data_engagement_des_.write('\n')
corr_data_engagement_des_.markdown('<div style="text-align: justify; font-size: 14px">his section delves into the correlation between different subscription models (monthly, yearly, per-book) and user engagement as well as retention rates. It also examines insights derived from analyzing membership duration alongside the number of audiobooks purchased or completed.</div>',unsafe_allow_html=True)
corr_data_engagement_des_.write('\n')


#select boxes for the variables that would be a part of the chart
subs_var__ = corr_data_engagement_des_.selectbox('Select a Variable to Study Preferences across Type of Subscriptions!',['Age_Group','City','Gender','Commuting_Mode','Genre'],key=2048)                                       
corr_data_engagement_des_.write('\n')

#creating the datset that would be displayed
dem_subs = req_data.groupby([subs_var__,'Subscription_Type'])['Ref ID'].count()
dem_subs=pd.DataFrame(dem_subs)
dem_subs.reset_index(inplace=True)
dem_subs.rename(columns={'Ref ID':'Number of Users'},inplace=True)


dem_subs_chart = px.bar(dem_subs,color='Subscription_Type',x=subs_var__,barmode='group',
                        y='Number of Users',title=f'Preference Trends about Subscription Type over {subs_var__}')

corr_data_engagement_chart_.plotly_chart(dem_subs_chart)


subs_ret_en_chart ,subs_ret_en_des  = corr_data_engagement_ .columns([.7,.3])
subs_ret_en_chart_ = subs_ret_en_chart.container(border=True)
subs_ret_en_des_ = subs_ret_en_des.container(border=True)


subs_ret_en_des_.markdown('<div style="text-align: justify; font-size: 18px">Analysing Engagement Rates and Customers Retained over type of Subscription!</div>',unsafe_allow_html=True)
subs_ret_en_des_.write('\n')
subs_ret_en_des_.markdown('<div style="text-align: justify; font-size: 14px"> By exploring these metrics, valuable insights can be gained regarding how subscription models impact user behavior and loyalty, providing actionable information for optimizing Audible\'s subscription offerings and retention strategies.</div>',unsafe_allow_html=True)
subs_ret_en_des_.write('\n')


#select boxes for the variables that would be a part of the chart
subs_var__ret = subs_ret_en_des_.selectbox('Select a Variable to Study Preferences across Type of Subscriptions!',['Users_Retained_5Weeks','Engagement_Rate','Membership_Duration','Listening_Speed_Numeric','Number_of_Audiobooks_Completed','Number_of_Audiobooks_Purchased','No_of_Books_read_in_a_year_Number'],key=1024)                                       
subs_var__ret_eval = subs_ret_en_des_.selectbox('Select an Aritmetic Basis to Evaluate!',['Average of Values','Sum of Values'])
subs_ret_en_des_.write('\n')


## funtion to return apprpriate df for plotting
def subs_eval_(eval_):
    if eval_ =='Average of Values':
        top_g = pd.DataFrame(req_data.groupby(['Subscription_Type'])[subs_var__ret].mean().round(2)).reset_index()
        return top_g
    elif eval_ == 'Sum of Values':
        top_g = pd.DataFrame(req_data.groupby(['Subscription_Type'])[subs_var__ret].sum()).reset_index()
        return top_g

mon_ret_en = subs_eval_(subs_var__ret_eval)


mon_ret_en_chart = px.bar(mon_ret_en,y=subs_var__ret,x='Subscription_Type',color = 'Subscription_Type',
                    title=f'{subs_var__ret_eval} of {subs_var__ret} over Type of Subscription')
subs_ret_en_chart_.plotly_chart(mon_ret_en_chart)


user_ret_engage_analysis = corr_data_engagement_.container(border=True)
user_ret_engage_analysis.markdown('<div style="text-align: center; font-size: 24px">Analysis & Results for User Enagagement and Retention given Subscription Type</div>',unsafe_allow_html=True)
user_ret_engage_analysis.write('\n')
user_ret_engage_analysis.markdown('<div style="text-align: justify; font-size: 14px">1. The most popular <i><b>genres to have the greatest number of subscriptions are Literature and Fiction, Parenting and Relationships genres.</i><b></div>',unsafe_allow_html=True)
user_ret_engage_analysis.write('\n')
user_ret_engage_analysis.markdown('<div style="text-align: justify; font-size: 14px">2. Additionally, the data suggested <i><b>different age groups as well as smartphone users generally prefer a per-book based subscription. This has also been supported by market research by Polaris which stated that the segment dedicated to one-time downloads is anticipated to dominate the market, boasting the highest share. This is because it empowers users to procure individual audiobooks without obligating them to subscribe to a service. The only exception to this being the ages 25-30 who also show a strong inclination for monthly subscription.</i></b></div>',unsafe_allow_html=True)
user_ret_engage_analysis.write('\n')
user_ret_engage_analysis.markdown('<div style="text-align: justify; font-size: 14px">3. On the other hand, there was no stark difference between listening device preferences and membership duration.<i><b> Demographic details suggested that users over the age of 50 have the longest duration of membership, and Southern cities along with Delhi and Pune have users with longer membership duration.</i><b> There was no significant difference for membership duration among the genders.</div>',unsafe_allow_html=True)
user_ret_engage_analysis.write('\n')
#user_ret_engage_analysis.markdown('<div style="text-align: justify; font-size: 14px">4. An interesting note shows that middle adults (36-40) not only show a comparatively high preference for ‘parenting and relationship’ genre in tandem with a tendency for the highest listening speed (over 1.5 times) and the highest completion rate for ‘teen and young adult’ genres. Similarly, while the number of users for the lifestyle genre stems low, it is preferred quite high among both the genders.</div>',unsafe_allow_html=True)
#user_ret_engage_analysis.write('\n')
#user_ret_engage_analysis.markdown('<div style="text-align: justify; font-size: 14px">5. Content acquisition strategies tailored for different age groups and genders can be genre specific. For instance, parenting and young adult self-development genre for users in the middle adulthood age bracket or lifestyle genre books represent potential for both genders.</div>',unsafe_allow_html=True)
#user_ret_engage_analysis.write('\n')



#########################################################################################################################################################################################################################################################################################################################################################
#########################################################################################################################################################################################################################################################################################################################################################
##########################################################################################################################################################################################################################################################################################################################################################Determining the popularity and discovery
popularity_scores_ = st.container(border=True)
test=pd.DataFrame()
test['Ratings_Given'] = req_data['Ratings_Given'] 
test['Reviews_Written'] = req_data['Reviews_Written']
test['Top 100 in Respective Genre'] = pd.get_dummies(req_data['Top 100 in Respective Genre'],drop_first=True)
test['Social_Sharing'] = req_data['Social_Sharing']
test['Repeat_Listening'] = req_data['Repeat_Listening']
test['Recommendations_Followed'] = req_data['Recommendations_Followed']
test['Membership_Duration'] = req_data['Membership_Duration']

#plotting the correlation heatmap
heatmap_popu = px.imshow(test.corr(method='spearman').round(3),labels=dict(x="Features", y="Features", color="Correlation"),aspect='auto',text_auto=True,color_continuous_scale ='jet',title='Correlation Heatmap for Audiobook Popularity based on Social Sharing, Reviews and Ratings',height=500)
popularity_scores_.plotly_chart(heatmap_popu,use_container_width=True)


######### Model Development #########

ages = pd.get_dummies(req_data['Age_Group'],drop_first=True)
gender = pd.get_dummies(req_data['Gender'],drop_first=True)
city = pd.get_dummies(req_data['City'],drop_first=True)
smart_phone = pd.get_dummies(req_data['Smart_Phone'],drop_first=True)
tab = pd.get_dummies(req_data['Tablet'],drop_first=True)
commuting_mode = pd.get_dummies(req_data['Commuting_Mode'],drop_first=True)
top_100 = pd.get_dummies(req_data['Top 100 in Respective Genre'],drop_first=True)
device_pref = pd.get_dummies(req_data['Device_Type'],drop_first=True)
start_time = pd.get_dummies(req_data['Start_Time_Slots'],drop_first=True)
genre  = pd.get_dummies(req_data['Genre'],drop_first=True)
avg_listeing = pd.get_dummies(req_data['Average_Listening_Speed'],drop_first=True)
listening_pref = pd.get_dummies(req_data['Language_Preference'],drop_first=True)
Listen_devi = pd.get_dummies(req_data['Listening_Device_Preference'],drop_first=True)
down_vs_streaming = pd.get_dummies(req_data['Download_vs_Streaming'],drop_first=True)
#prepping the data for model
model_data = pd.DataFrame()
model_data = pd.concat([model_data,ages],axis=1)
model_data = pd.concat([model_data,gender],axis=1)
model_data = pd.concat([model_data,city],axis=1)
model_data = pd.concat([model_data,smart_phone],axis=1)
model_data = pd.concat([model_data,tab],axis=1)
model_data = pd.concat([model_data,commuting_mode],axis=1)
model_data =  pd.concat([model_data,top_100],axis=1)
model_data =  pd.concat([model_data,device_pref],axis=1)
model_data =  pd.concat([model_data,start_time],axis=1)
model_data =  pd.concat([model_data,genre],axis=1)
model_data =  pd.concat([model_data,avg_listeing],axis=1)
model_data =  pd.concat([model_data,listening_pref],axis=1)
model_data =  pd.concat([model_data,Listen_devi],axis=1)
model_data =  pd.concat([model_data,down_vs_streaming],axis=1)
model_data['Completion_Rate_2_Weeks'] = req_data['Completion_Rate_2_Weeks']
model_data['Completion_Rate_5_Weeks'] = req_data['Completion_Rate_5_Weeks']
model_data['Commuting_Time_Minutes'] = req_data['Commuting_Time_Minutes']
model_data['No_of_Books_read_in_a_year'] = req_data['No_of_Books_read_in_a_year_Number']
model_data['Time_Spent_Browsing'] = req_data['Time_Spent_Browsing']
model_data['Genre_Exploration'] = req_data['Genre_Exploration']
model_data['Number_of_Audiobooks_Purchased'] = req_data['Number_of_Audiobooks_Purchased'] 
model_data['Number_of_Audiobooks_Completed'] = req_data['Number_of_Audiobooks_Completed'] 
model_data['Social_Sharing'] = req_data['Social_Sharing'] 
model_data['Ratings_Given'] = req_data['Ratings_Given'] 
model_data['Recommendations_Followed'] = req_data['Recommendations_Followed'] 



######
model_retention_rate = st.container(border=True)
model_retention_rate.markdown('<div style="text-align: center; font-size: 24px">Predictive Model on Retention</div>',unsafe_allow_html=True)
model_retention_rate.divider()
model_retention_rate_class,model_retention_rate_chart = model_retention_rate.columns([.35,.65]) 
model_retention_rate_class_ = model_retention_rate_class.container(border=True)
model_retention_rate_chart_ = model_retention_rate_chart.container(border=True)


#from sklearn.ensemble import RandomForestClassfier
smt = SMOTE()
scaler = StandardScaler()
#Xrroin yrroin ± smr.irresompLe(Xrroin yrrin)
#X_train,X_test,y_train,y_test = train_test_split(model_data.drop(['Completion_Rate_2_Weeks','Completion_Rate_5_Weeks','Social_Sharing','Ratings_Given','Recommendations_Followed'],axis=1),req_data['Completion_Rate_5_Weeks'].apply(lambda x: 1 if x>=100 else 0),test_size=0.25)
@st.cache_resource
def sel_feat_returner():
    X_Scaled = scaler.fit_transform(model_data.drop(['Completion_Rate_2_Weeks','Completion_Rate_5_Weeks','Social_Sharing','Ratings_Given','Recommendations_Followed'],axis=1))
    X, y = smt.fit_resample(X_Scaled,req_data['Completion_Rate_5_Weeks'].apply(lambda x: 1 if x>=100 else 0))
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
    sel = SelectFromModel(RandomForestClassifier())
    sel.fit(X_train, y_train)
    sel.get_support()
    selected_feat = model_data.drop(['Completion_Rate_2_Weeks','Completion_Rate_5_Weeks','Social_Sharing','Ratings_Given','Recommendations_Followed'],axis=1).columns[sel.get_support()]
    return selected_feat,X_train,X_test,y_train,y_test

selected_feat,X_train,X_test,y_train,y_test = sel_feat_returner()

selected_vars = model_retention_rate_chart_.multiselect('Select Variables for the Model:',selected_feat,default=selected_feat.to_list())


#training the model on selected features and then 
model_rfc = RandomForestClassifier()
X_Scaled = scaler.fit_transform(model_data.drop(['Completion_Rate_2_Weeks','Completion_Rate_5_Weeks','Social_Sharing','Ratings_Given','Recommendations_Followed'],axis=1)[selected_feat])
X, y = smt.fit_resample(X_Scaled,req_data['Completion_Rate_5_Weeks'].apply(lambda x: 1 if x>=100 else 0))
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
model_rfc.fit(X_train,y_train)
y_pred_rfc = model_rfc.predict(X_test)

#visualisation of accracy
cm = confusion_matrix(y_test, y_pred_rfc)
# Create the heatmap plot using Plotly Express
con_mat = px.imshow(cm, labels=dict(x="Predicted", y="True"), x=['Not Retained', 'Retained'], y=['Not Retained', 'Retained'], color_continuous_scale='Blues',text_auto=True,title='Confusion Matrix Predicted Customer Churn')
# Update the color axis to hide the scale
con_mat.update_coloraxes(showscale=False)
#creating a container for model_accuracy
model_retention_rate_class_.plotly_chart(con_mat,use_container_width=True)
model_retention_rate_class_.divider()
model_retention_rate_class_.text(classification_report(y_test,y_pred_rfc))

# Get feature importances
importances = model_rfc.feature_importances_

# Get feature names
feature_names = selected_vars
# Create a horizontal bar chart for feature importance
fig = go.Figure(go.Bar(
    x=feature_names,
    y=importances,
))

# Customize layout
fig.update_layout(
    title='Feature Importance in Random Forest Classifier',
    xaxis_title='Feature Names',
    yaxis_title='Importance',
)
model_retention_rate_chart_.plotly_chart(fig,use_container_width=True)





###############################################################################################################################################################

model_ratings = st.container(border=True)
model_ratings.markdown('<div style="text-align: center; font-size: 24px">Predictive Model on User Ratings</div>',unsafe_allow_html=True)
model_ratings.divider()
model_ratings_class,model_ratings_chart = model_ratings.columns([.35,.65]) 
model_ratings_class_ = model_ratings_class.container(border=True)
model_ratings_chart_ = model_ratings_chart.container(border=True)



def rating_sorter(rate):
    if rate>=4:
        return 'Good'
    if rate==3:
        return 'Average'
    else:
        return 'Bad'
    

#from sklearn.ensemble import RandomForestClassfier
smt = SMOTE()
#Xrroin yrroin ± smr.irresompLe(Xrroin yrrin)
#X_train,X_test,y_train,y_test = train_test_split(model_data.drop(['Completion_Rate_2_Weeks','Completion_Rate_5_Weeks','Social_Sharing','Ratings_Given','Recommendations_Followed'],axis=1),req_data['Ratings_Given'].apply(lambda x:rating_sorter(x)),test_size=0.25)
@st.cache_resource
def sel_feat_returner():
    X_Scaled = scaler.fit_transform(model_data.drop(['Completion_Rate_2_Weeks','Completion_Rate_5_Weeks','Social_Sharing','Ratings_Given','Recommendations_Followed'],axis=1))
    X, y = smt.fit_resample(X_Scaled,req_data['Ratings_Given'].apply(lambda x:rating_sorter(x)))
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
    sel = SelectFromModel(RandomForestClassifier(),threshold=0.1)
    sel.fit(X_train, y_train)
    sel.get_support()
    selected_feat =  model_data.drop(['Completion_Rate_2_Weeks','Completion_Rate_5_Weeks','Social_Sharing','Ratings_Given','Recommendations_Followed'],axis=1).columns[sel.get_support()]
    return selected_feat,X_train,X_test,y_train,y_test 

selected_feat,X_train,X_test,y_train,y_test = sel_feat_returner()

selected_vars = model_ratings_chart_.multiselect('Select Variables for the Model:',selected_feat,default=selected_feat.to_list())


#training the model on selected features and then 
model_rfc = RandomForestClassifier()
X_Scaled = scaler.fit_transform(model_data.drop(['Completion_Rate_2_Weeks','Completion_Rate_5_Weeks','Social_Sharing','Ratings_Given','Recommendations_Followed'],axis=1)[selected_feat])
X, y = smt.fit_resample(X_Scaled,req_data['Ratings_Given'].apply(lambda x:rating_sorter(x)))
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)                                                               
model_rfc.fit(X_train,y_train)
y_pred_rfc = model_rfc.predict(X_test)


#visualisation of accracy
cm = confusion_matrix(y_test, y_pred_rfc)
# Create the heatmap plot using Plotly Express
con_mat = px.imshow(cm,labels=dict(x="Predicted", y="True"),y=['Average','Bad','Good'],x=['Average','Bad','Good'],color_continuous_scale='Blues',text_auto=True,title='Confusion Matrix Predicted User Ratings')
# Update the color axis to hide the scale
con_mat.update_coloraxes(showscale=False)
#creating a container for model_accuracy
model_ratings_class_.plotly_chart(con_mat,use_container_width=True)
model_ratings_class_.divider()
model_ratings_class_.text(classification_report(y_test,y_pred_rfc))

# Get feature importances
importances = model_rfc.feature_importances_

# Get feature names
feature_names = selected_vars
# Create a horizontal bar chart for feature importance
fig = go.Figure(go.Bar(
    x=feature_names,
    y=importances,
))

# Customize layout
fig.update_layout(
    title='Feature Importance in Random Forest Classifier',
    xaxis_title='Feature Names',
    yaxis_title='Importance',
)

model_ratings_chart_.plotly_chart(fig,use_container_width=True)


############################################################################################################################
model_recomm_follow = st.container(border=True)
model_recomm_follow.markdown('<div style="text-align: center; font-size: 24px">Predictive Model on Reccomendations Followed</div>',unsafe_allow_html=True)
model_recomm_follow.divider()
model_recomm_follow_class,model_recomm_follow_chart = model_recomm_follow.columns([.35,.65]) 
model_recomm_follow_class_ = model_recomm_follow_class.container(border=True)
model_recomm_follow_chart_ = model_recomm_follow_chart.container(border=True)


#from sklearn.ensemble import RandomForestClassfier
smt = SMOTE()
#Xrroin yrroin ± smr.irresompLe(Xrroin yrrin)
@st.cache_resource
def sel_feat_returner():
    X_Scaled = scaler.fit_transform(model_data.drop(['Completion_Rate_2_Weeks','Completion_Rate_5_Weeks','Social_Sharing','Ratings_Given','Recommendations_Followed'],axis=1))
    X, y = smt.fit_resample(X_Scaled,req_data['Recommendations_Followed'])
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
    sel = SelectFromModel(RandomForestClassifier(),threshold=0.1)
    sel.fit(X_train, y_train)
    sel.get_support()
    selected_feat = model_data.drop(['Completion_Rate_2_Weeks','Completion_Rate_5_Weeks','Social_Sharing','Ratings_Given','Recommendations_Followed'],axis=1).columns[(sel.get_support())]
    return selected_feat,X_train,X_test,y_train,y_test

selected_feat,X_train,X_test,y_train,y_test = sel_feat_returner()

selected_vars = model_recomm_follow_chart_.multiselect('Select Variables for the Model:',selected_feat,default=selected_feat.to_list())


#training the model on selected features and then 
model_rfc = RandomForestClassifier()
X_Scaled = scaler.fit_transform(model_data.drop(['Completion_Rate_2_Weeks','Completion_Rate_5_Weeks','Social_Sharing','Ratings_Given','Recommendations_Followed'],axis=1)[selected_feat])
X, y = smt.fit_resample(X_Scaled,req_data['Recommendations_Followed'])
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)  
model_rfc.fit(X_train,y_train)
y_pred_rfc = model_rfc.predict(X_test)


#visualisation of accracy
cm = confusion_matrix(y_test, y_pred_rfc)
# Create the heatmap plot using Plotly Express
con_mat = px.imshow(cm, labels=dict(x="Predicted", y="True"), x=['Not Followed', 'Followed'], y=['Not Followed', 'Followed'], color_continuous_scale='Blues',text_auto=True,title='Confusion Matrix Predicted Following Recommendations')
# Update the color axis to hide the scale
con_mat.update_coloraxes(showscale=False)
#creating a container for model_accuracy
model_recomm_follow_class_.plotly_chart(con_mat,use_container_width=True)
model_recomm_follow_class_.divider()
model_recomm_follow_class_.text(classification_report(y_test,y_pred_rfc))

# Get feature importances
importances = model_rfc.feature_importances_

# Get feature names
feature_names = selected_vars
# Create a horizontal bar chart for feature importance
fig = go.Figure(go.Bar(
    x=feature_names,
    y=importances,
))

# Customize layout
fig.update_layout(
    title='Feature Importance in Random Forest Classifier',
    xaxis_title='Feature Names',
    yaxis_title='Importance',
)

model_recomm_follow_chart_.plotly_chart(fig,use_container_width=True)

#########################################################################################################

model_social_sharing = st.container(border=True)
model_social_sharing.markdown('<div style="text-align: center; font-size: 24px">Predictive Model on Social Sharing</div>',unsafe_allow_html=True)
model_social_sharing.divider()
model_social_sharing_class,model_social_sharing_chart = model_social_sharing.columns([.35,.65]) 
model_social_sharing_class_ = model_social_sharing_class.container(border=True)
model_social_sharing_chart_ = model_social_sharing_chart.container(border=True)


#from sklearn.ensemble import RandomForestClassfier
smt = SMOTE()
#Xrroin yrroin ± smr.irresompLe(Xrroin yrrin)
#X_train,X_test,y_train,y_test = train_test_split(model_data.drop(['Completion_Rate_2_Weeks','Completion_Rate_5_Weeks','Social_Sharing','Ratings_Given','Recommendations_Followed'],axis=1),req_data['Social_Sharing'],test_size=0.25)
@st.cache_resource
def sel_feat_returner():
    X_Scaled = scaler.fit_transform(model_data.drop(['Completion_Rate_2_Weeks','Completion_Rate_5_Weeks','Social_Sharing','Ratings_Given','Recommendations_Followed'],axis=1))
    X, y = smt.fit_resample(X_Scaled,req_data['Social_Sharing'])
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)  
    sel = SelectFromModel(RandomForestClassifier(),threshold=0.1)
    sel.fit(X_train, y_train)
    sel.get_support()
    selected_feat = model_data.drop(['Completion_Rate_2_Weeks','Completion_Rate_5_Weeks','Social_Sharing','Ratings_Given','Recommendations_Followed'],axis=1).columns[(sel.get_support())]
    return selected_feat,X_train,X_test,y_train,y_test


selected_feat,X_train,X_test,y_train,y_test = sel_feat_returner()

selected_vars = model_social_sharing_chart_.multiselect('Select Variables for the Model:',selected_feat,default=selected_feat.to_list(),key=99999)


#training the model on selected features and then 
model_rfc = RandomForestClassifier()
X_Scaled = model_data.drop(['Completion_Rate_2_Weeks','Completion_Rate_5_Weeks','Social_Sharing','Ratings_Given','Recommendations_Followed'],axis=1)[selected_feat]
X, y = smt.fit_resample(X_Scaled,req_data['Social_Sharing'])
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)  
model_rfc.fit(X_train,y_train)
y_pred_rfc = model_rfc.predict(X_test)


#visualisation of accracy
cm = confusion_matrix(y_test, y_pred_rfc)
# Create the heatmap plot using Plotly Express
con_mat = px.imshow(cm, labels=dict(x="Predicted", y="True"), x=['Not Shared', 'Shared'], y=['Not Shared', 'Shared'], color_continuous_scale='Blues',text_auto=True,title='Confusion Matrix Predicted Social Sharing')
# Update the color axis to hide the scale
con_mat.update_coloraxes(showscale=False)
#creating a container for model_accuracy
model_social_sharing_class_.plotly_chart(con_mat,use_container_width=True)
model_social_sharing_class_.divider()
model_social_sharing_class_.text(classification_report(y_test,y_pred_rfc))

# Get feature importances
importances = model_rfc.feature_importances_

# Get feature names
feature_names = selected_vars
# Create a horizontal bar chart for feature importance
fig = go.Figure(go.Bar(
    x=feature_names,
    y=importances,
))

# Customize layout
fig.update_layout(
    title='Feature Importance in Random Forest Classifier Social Sharing',
    xaxis_title='Feature Names',
    yaxis_title='Importance',
)

model_social_sharing_chart_.plotly_chart(fig,use_container_width=True)


