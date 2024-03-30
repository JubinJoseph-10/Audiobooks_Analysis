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


ico = Image.open("Data/Audible.png")
#Setting page configs
st.set_page_config(layout='wide',initial_sidebar_state='collapsed',page_icon=ico)

#Setting the title
st.title('Audiobooks User Analyis')

req_data = pd.read_csv("Data/Audible_Dashboard_Data.csv")

##### Habbits Across Users 

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
                                                'Engagement_Rate'])

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
demo_listening_pref_des_.markdown('<div style="text-align: justify; font-size: 18px"></div>',unsafe_allow_html=True)



#select boxes for the variables that would be a part of the chart
demo_Var_pref = demo_listening_pref_des_.selectbox('Select a Demographic Variable to Study!',['Age_Group','City','Gender','Commuting_Mode'])
pref_Var_ = demo_listening_pref_des_.selectbox('Select a Variable to Study Preferences!',['Download_vs_Streaming','Language_Preference','Listening_Device_Preference',
                                                'Listening_Context','Preferred_Listening_Time','Subscription_Type','Average_Listening_Speed'])                                       

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

#linebefore heatmap
demo_listening_habs.divider()
#plotting the correlation heatmap
heatmap = px.imshow(corr_data.corr(method='spearman').round(3),labels=dict(x="Features", y="Features", color="Correlation"),aspect=3,text_auto=True,color_continuous_scale ='jet',title='Correlation Heatmap for all the Variables Related to Commuting and Audiobook Consumption',height=700)
demo_listening_habs.plotly_chart(heatmap,use_container_width=True)



###### Audiobook Usage based on the nature of technology used
#structuring the space for the analysis
dev_pref_habbits = st.container(border=True)
dev_pref_habbits.markdown('<div style="text-align: center; font-size: 24px">Trends Related to Device and Technology Usage</div>',unsafe_allow_html=True)
dev_pref_habbits.write('\n')
dev_pref_habbits_des , dev_pref_chart_space = dev_pref_habbits.columns([.3,.7])
dev_pref_habbits_des_ = dev_pref_habbits_des.container(border =True)
dev_pref_chart_space_ = dev_pref_chart_space.container(border =True)


#select boxes for the variables that would be a part of the chart
tech_var_pref = dev_pref_habbits_des_.selectbox('Select a Technology Related Variable to Study!',['Smart_Phone','Tablet','Device_Type','Listening_Device_Preference'])
tech_pref_Var_ = dev_pref_habbits_des_.selectbox('Select a Variable to Study Usage!',['No_of_Books_read_in_a_year_Number','Event_Duration_Minutes_2_Weeks','Event_Duration_Minutes_5_Weeks',
                                                'Product_Running_Time','Completion_Rate_5_Weeks','Membership_Duration','Number_of_Audiobooks_Completed','Number_of_Audiobooks_Purchased','Time_Spent_Browsing','Genre_Exploration','Engagement_Rate'])

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


#select boxes for the variables that would be a part of the chart
tech_var_pref = dev_pref_des_.selectbox('Select a Technology Related Variable to Study!',['Smart_Phone','Tablet','Device_Type','Listening_Device_Preference'],key=4)
tech_pref_Var = dev_pref_des_.selectbox('Select a Variable to Study Usage!',['Download_vs_Streaming','Language_Preference','Listening_Device_Preference',
                                                'Listening_Context','Preferred_Listening_Time','Subscription_Type','Average_Listening_Speed'],key=8)                                       


#creating the datset that would be displayed
tech_pre_val = req_data.groupby([tech_var_pref,tech_pref_Var])['Ref ID'].count()
tech_pre_val=pd.DataFrame(tech_pre_val)
tech_pre_val.reset_index(inplace=True)
tech_pre_val.rename(columns={'Ref ID':'Number of Users'},inplace=True)

tech_pref_chart = px.bar(tech_pre_val,color=tech_pref_Var,x=tech_var_pref,barmode='group',
                        y='Number of Users',title=f'Preference Trends about {tech_pref_Var} over {tech_var_pref}')
dev_pref_chart_.plotly_chart(tech_pref_chart)




###### Top Genres
top_g = st.container(border=True)
top_g_chart, top_g_des = top_g.columns([.7,.3])
top_g_des_des_ = top_g_des.container(border =True)
top_g_chart_ = top_g_chart.container(border =True)


#select boxes for the variables that would be a part of the chart
basis_top_g = top_g_des_des_.selectbox('Select a Demographic Variable to Study!',['Age_Group','City','Gender','Commuting_Mode'],key=64)
metric_ = top_g_des_des_.selectbox('Select a Metric to Evaluate!',['Number of Users','Listening_Speed_Numeric','Completion_Rate_5_Weeks'])

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
top_genres = st.container(border=True)
top_genres_ , top_genres_var = st.columns([.3,.7])
top_genres_var_ = top_genres_var.container(border=True)
top_genres__ = top_genres_.container(border=True)


#select boxes for the variables that would be a part of the chart
#demo_sel = top_genres_var_.selectbox('Select a Demographic Related Variable to Study!',['Age_Group','City','Gender','Commuting_Mode'],key=16)
demo_eval_var = top_genres__.selectbox('Select a Basis to Decide Top Genre!',['No_of_Books_read_in_a_year_Number','Event_Duration_Minutes_2_Weeks','Event_Duration_Minutes_5_Weeks',
                                                'Product_Running_Time','Completion_Rate_5_Weeks','Membership_Duration','Number_of_Audiobooks_Completed','Number_of_Audiobooks_Purchased','Time_Spent_Browsing','Genre_Exploration','Engagement_Rate'])
demo_eval_var_basis = top_genres__.selectbox('Select an Aritmetic Basis to Decide Top Genre!',['Average of Values','Sum of Values'])


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
genre_pref = st.container(border=True)
genre_pref_chart , genre_pref_des = genre_pref.columns([.7,.3])
genre_pref_chart_ = genre_pref_chart.container(border=True)
genre_pref_des_ = genre_pref_des.container(border=True)

#select boxes for the variables that would be a part of the chart
genre_pref_var = genre_pref_des_.selectbox('Select a Variable to Study Preferences across Genres!',['Download_vs_Streaming','Language_Preference','Listening_Device_Preference',
                                                'Listening_Context','Preferred_Listening_Time','Subscription_Type','Average_Listening_Speed'],key=32)                                       

#creating the datset that would be displayed
g_pref = req_data.groupby(['Genre',genre_pref_var])['Ref ID'].count()
g_pref=pd.DataFrame(g_pref)
g_pref.reset_index(inplace=True)
g_pref.rename(columns={'Ref ID':'Number of Users'},inplace=True)

g_pref_chart = px.bar(g_pref,color=genre_pref_var,x='Genre',barmode='group',
                        y='Number of Users',title=f'Preference Trends about {tech_pref_Var} over Genres')
genre_pref_chart_.plotly_chart(g_pref_chart)


################Correlation
corr_data_engagement=pd.DataFrame()
corr_data_engagement = pd.get_dummies(req_data['Subscription_Type'])
corr_data_engagement['Engagemrnt Rate'] = req_data['Engagement_Rate']
corr_data_engagement['Retention'] = req_data['Users_Retained_5Weeks']
corr_data_engagement['Membership_Duration'] = req_data['Membership_Duration']
corr_data_engagement['Number_of_Audiobooks_Completed'] = req_data['Number_of_Audiobooks_Completed']
corr_data_engagement['Number_of_Audiobooks_Purchased'] = req_data['Number_of_Audiobooks_Purchased']


#plotting the correlation heatmap
heatmap_user_engage = px.imshow(corr_data_engagement.corr(method='spearman').round(3),labels=dict(x="Features", y="Features", color="Correlation"),aspect='auto',text_auto=True,color_continuous_scale ='jet',title='Correlation Heatmap for all the Variables Related to Commuting and Audiobook Consumption',height=500)
corr_engagement = st.container(border=True)
corr_engagement.plotly_chart(heatmap_user_engage,use_container_width=True)


#Determining the popularity and discovery
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
heatmap_popu = px.imshow(test.corr(method='spearman').round(3),labels=dict(x="Features", y="Features", color="Correlation"),aspect='auto',text_auto=True,color_continuous_scale ='jet',title='Correlation Heatmap for all the Variables Related to Commuting and Audiobook Consumption',height=500)
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
#Xrroin yrroin ± smr.irresompLe(Xrroin yrrin)
#X_train,X_test,y_train,y_test = train_test_split(model_data.drop(['Completion_Rate_2_Weeks','Completion_Rate_5_Weeks','Social_Sharing','Ratings_Given','Recommendations_Followed'],axis=1),req_data['Completion_Rate_5_Weeks'].apply(lambda x: 1 if x>=100 else 0),test_size=0.25)
@st.cache_resource
def sel_feat_returner():
    X, y = smt.fit_resample(model_data.drop(['Completion_Rate_2_Weeks','Completion_Rate_5_Weeks','Social_Sharing','Ratings_Given','Recommendations_Followed',
                                             'Number_of_Audiobooks_Purchased','Number_of_Audiobooks_Completed'],axis=1),req_data['Completion_Rate_5_Weeks'].apply(lambda x: 1 if x>=100 else 0))
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
    sel = SelectFromModel(RandomForestClassifier())
    sel.fit(X_train, y_train)
    sel.get_support()
    selected_feat = X_train.columns[(sel.get_support())]
    return selected_feat,X_train,X_test,y_train,y_test

selected_feat,X_train,X_test,y_train,y_test = sel_feat_returner()

selected_vars = model_retention_rate_chart_.multiselect('Select Variables for the Model:',selected_feat,default=selected_feat.to_list())


#training the model on selected features and then 
model_rfc = RandomForestClassifier()
model_rfc.fit(X_train[selected_vars],y_train)
y_pred_rfc = model_rfc.predict(X_test[selected_vars])


#visualisation of accracy
cm = confusion_matrix(y_test, y_pred_rfc)
# Create the heatmap plot using Plotly Express
con_mat = px.imshow(cm, labels=dict(x="Predicted", y="True"), x=['Not Retained', 'Retained'], y=['Not Retained', 'Retained'], color_continuous_scale='Blues',text_auto=True,title='Confusion Matrix Random Forest Classifier Customer Churn')
# Update the color axis to hide the scale
con_mat.update_coloraxes(showscale=False)
#creating a container for model_accuracy
model_retention_rate_class_.plotly_chart(con_mat,use_container_width=True)
model_retention_rate_class_.divider()
model_retention_rate_class_.text(classification_report(y_test,y_pred_rfc))

# Get feature importances
importances = model_rfc.feature_importances_

# Get feature names
feature_names = X_test[selected_vars].columns.tolist()
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
    X, y = smt.fit_resample(model_data.drop(['Completion_Rate_2_Weeks','Completion_Rate_5_Weeks','Social_Sharing','Ratings_Given','Recommendations_Followed',
                                             'Number_of_Audiobooks_Purchased','Number_of_Audiobooks_Completed'],axis=1),req_data['Ratings_Given'].apply(lambda x:rating_sorter(x)))
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
    sel = SelectFromModel(RandomForestClassifier())
    sel.fit(X_train, y_train)
    sel.get_support()
    selected_feat = X_train.columns[(sel.get_support())]
    return selected_feat,X_train,X_test,y_train,y_test 

selected_feat,X_train,X_test,y_train,y_test = sel_feat_returner()

selected_vars = model_ratings_chart_.multiselect('Select Variables for the Model:',selected_feat,default=selected_feat.to_list())


#training the model on selected features and then 
model_rfc = RandomForestClassifier()
model_rfc.fit(X_train[selected_vars],y_train)
y_pred_rfc = model_rfc.predict(X_test[selected_vars])


#visualisation of accracy
cm = confusion_matrix(y_test, y_pred_rfc)
# Create the heatmap plot using Plotly Express
con_mat = px.imshow(cm,y=['Good','Average','Bad'],x=['Bad','Average','Good'],color_continuous_scale='Blues',text_auto=True,title='Confusion Matrix Random Forest Classifier Customer Churn')
# Update the color axis to hide the scale
con_mat.update_coloraxes(showscale=False)
#creating a container for model_accuracy
model_ratings_class_.plotly_chart(con_mat,use_container_width=True)
model_ratings_class_.divider()
model_ratings_class_.text(classification_report(y_test,y_pred_rfc))

# Get feature importances
importances = model_rfc.feature_importances_

# Get feature names
feature_names = X_test[selected_vars].columns.tolist()
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
X_train,X_test,y_train,y_test = train_test_split(model_data.drop(['Completion_Rate_2_Weeks','Completion_Rate_5_Weeks','Social_Sharing','Ratings_Given','Recommendations_Followed'],axis=1),req_data['Recommendations_Followed'],test_size=0.25)
@st.cache_resource
def sel_feat_returner():
    X, y = smt.fit_resample(model_data.drop(['Completion_Rate_2_Weeks','Completion_Rate_5_Weeks','Social_Sharing','Ratings_Given','Recommendations_Followed'],axis=1),req_data['Recommendations_Followed'])
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25)
    sel = SelectFromModel(RandomForestClassifier())
    sel.fit(X_train, y_train)
    sel.get_support()
    selected_feat = X_train.columns[(sel.get_support())]
    return selected_feat,X_train,X_test,y_train,y_test

selected_feat,X_train,X_test,y_train,y_test = sel_feat_returner()

selected_vars = model_recomm_follow_chart_.multiselect('Select Variables for the Model:',selected_feat,default=selected_feat.to_list())


#training the model on selected features and then 
model_rfc = RandomForestClassifier()
model_rfc.fit(X_train[selected_vars],y_train)
y_pred_rfc = model_rfc.predict(X_test[selected_vars])


#visualisation of accracy
cm = confusion_matrix(y_test, y_pred_rfc)
# Create the heatmap plot using Plotly Express
con_mat = px.imshow(cm, labels=dict(x="Predicted", y="True"), x=['Not Followed', 'Followed'], y=['Not Followed', 'Followed'], color_continuous_scale='Blues',text_auto=True,title='Confusion Matrix Random Forest Classifier Customer Churn')
# Update the color axis to hide the scale
con_mat.update_coloraxes(showscale=False)
#creating a container for model_accuracy
model_recomm_follow_class_.plotly_chart(con_mat,use_container_width=True)
model_recomm_follow_class_.divider()
model_recomm_follow_class_.text(classification_report(y_test,y_pred_rfc))

# Get feature importances
importances = model_rfc.feature_importances_

# Get feature names
feature_names = X_test[selected_vars].columns.tolist()
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
























