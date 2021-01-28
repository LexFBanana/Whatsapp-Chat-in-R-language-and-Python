#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import pandas as pd
sys.path.insert(0,'..')  # Needed to import package in parent dir, remove this if you pip installed the package
from soan.whatsapp import helper      # Helper to prepare the data
from soan.whatsapp import general     # General statistics
from soan.whatsapp import tf_idf      # To calculate TF-IDF
from soan.whatsapp import emoji       # To analyze emoji use
from soan.whatsapp import topic       # Topic modelling and summarization
from soan.whatsapp import sentiment   # Sentiment Analysis
import wordcloud   # Create Word Clouds

from soan.colors   import colors      # Frequent Color Visualization

get_ipython().run_line_magic('matplotlib', 'inline')

get_ipython().run_line_magic('load_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')


# In[2]:


df = helper.import_data('C:/PP/Test2.txt')
df = helper.preprocess_data(df)




if False:
    user_labels = {old: new for old, new in zip(sorted(df.User.unique()), ['Her', 'Me'])}
    df.User = df.User.map(user_labels)


# In[3]:


general.print_users(df)


# In[4]:


df1= helper.preprocess_data(df)
df1.to_csv('amd 2015.csv')
df1=pd.read_csv('amd 2015.csv',parse_dates=['Date'],index_col='Date')#,index_col='Date'
df1=pd.DataFrame(df1,columns=[ 'User', 'Message_Clean'])
def barChart(xValues, yValues, bTitle, xLable, yLable):
    

    plt.bar(xValues, yValues,width=0.8)
    plt.title(bTitle)
    plt.xlabel(xLable)
    plt.ylabel(yLable)
    plt.show()
df1


# In[5]:


df1.info()


# In[6]:


df1['Year']=df1.index.year
df1['Month']=df1.index.month
df1['Week']=df1.index.week
df1['Day']=df1.index.day
df1['Hour']=df1.index.hour
df1


# In[7]:


df1=df1.rename(columns = {'Message_Clean':'Message'})
df1


# In[ ]:





# In[8]:


df1


# In[9]:


import matplotlib.pyplot as plt
df2=df1.groupby(['User']).count()['Message']
df2.max()

df2.plot.barh( title='Number of Messages according to Year',figsize=(14, 10),width=0.7)
plt.xlabel('Message')
plt.show()


# In[ ]:





# In[10]:



df2=df1.groupby(['Year']).count()['Message'].to_frame()

df2.plot.barh( title='Number of Messages according to Year',figsize=(14, 7))
plt.xlabel('Message')
plt.show()
plt.savefig('2.png')


# In[11]:


df2=df1.groupby(['Month']).count()['Message'].to_frame()
df2.plot.barh( title='Number of Messages according to Month',figsize=(14, 7))
plt.xlabel('Message')
plt.show()
plt.savefig('3.png')


# In[12]:


df2=df1.groupby(['Day']).count()['Message'].to_frame()
df2.plot.barh( title='Number of Messages according to Day',figsize=(14, 7),width=0.7)
plt.xlabel('Message')
plt.show()
plt.savefig('4.png')


# In[13]:


df2=df1.groupby(['Hour']).count()['Message'].to_frame()
df2.plot.barh( title='Number of Messages according to Hour',figsize=(14, 7),width=0.7)
plt.xlabel('Message')
plt.show()
plt.savefig('5.png')


# In[14]:


df['User']


# In[ ]:





# In[ ]:





# In[15]:


df1=general.print_users(df)


# In[16]:


user = "All"
language = "english"


# In[17]:


#d=general.plot_messages(df, colors=None, trendline=False, savefig=False, dpi=100)


# In[18]:


#general.plot_day_spider(df, colors=None, savefig=False, dpi=100)


# In[19]:


general.plot_active_days(df, savefig=False, dpi=100, user='All')


# In[20]:


general.plot_active_hours(df, color='#ffdfba', savefig=False, dpi=100, user='To (Cs1)')
plt.savefig('7.png')


# In[21]:


years = set(pd.DatetimeIndex(df.Date.values).year)

for year in years:
    general.calendar_plot(df, year=year, how='count', column='index')
#plt.savefig('8.png')


# In[22]:


df2=general.print_stats(df)


# In[23]:


general.print_timing(df)


# In[24]:


#general.calendar_plot(df, year=2020, how='count', column = 'User', savefig=False, dpi=100)


# In[25]:



counts = tf_idf.count_words_per_user(df, sentence_column="Message_Only_Text", user_column="User")
counts = tf_idf.remove_stopwords(counts, language=language, column="Word")


# In[26]:


unique_words = tf_idf.get_unique_words(counts, df, version = 'A')


# In[ ]:





# In[27]:


tf_idf.plot_unique_words(unique_words, 
                         user='To (Cs1)', 
                         image_path='C:/PP/1.jpeg', # use '../images/mask.png' to use the standard image
                         image_url=None, 
                         title="To (Cs1)", 
                         title_color="green", 
                         title_background='#AAAAAA', 
                         width=25, 
                         height=50)
plt.savefig('8.png')


# In[ ]:


temp = df[['index', 'Message_Raw', 'User', 'Message_Clean', 'Message_Only_Text']].copy()
temp = emoji.prepare_data(temp)

# Count all emojis
counts = emoji.count_emojis(temp, non_unicode=True)

# Get unique emojis
list_of_words = [word for user in counts for word in counts[user]]
unique_emoji = emoji.get_unique_emojis(temp, counts, list_of_words)
del temp


# In[ ]:


emoji.print_stats(unique_emoji, counts)


# In[ ]:


emoji.get_unique_emojis(df, counts, list_of_words)


# In[ ]:



topic.topics(df, model='lda', language="english")


# In[ ]:





# In[ ]:


import re
import regex
import numpy as np
import emoji
import plotly.express as px
from collections import Counter
import matplotlib.pyplot as plt
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator
get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().system('git clone https://github.com/amueller/word_cloud.git')
get_ipython().run_line_magic('cd', 'word_cloud')
get_ipython().system(' pip install .')


# In[ ]:


def startsWithDateAndTimeAndroid(s):
    pattern = '^([0-9]+)(\/)([0-9]+)(\/)([0-9]+), ([0-9]+):([0-9]+)[ ]?(AM|PM|am|pm)? -' 
    result = re.match(pattern, s)
    if result:
        return True
    return False

def startsWithDateAndTimeios(s):
    pattern = '^\[([0-9]+)([\/-])([0-9]+)([\/-])([0-9]+)[,]? ([0-9]+):([0-9][0-9]):([0-9][0-9])?[ ]?(AM|PM|am|pm)?\]' 
    result = re.match(pattern, s)
    if result:
        return True
    return False


# In[ ]:


def FindAuthor(s):
  s=s.split(":")
  if len(s)==2:
    return True
  else:
    return False


# In[ ]:


def getDataPointAndroid(line):   
    splitLine = line.split(' - ') 
    dateTime = splitLine[0]
    date, time = dateTime.split(', ') 
    message = ' '.join(splitLine[1:])
    if FindAuthor(message): 
        splitMessage = message.split(':') 
        author = splitMessage[0] 
        message = ' '.join(splitMessage[1:])
    else:
        author = None
    return date, time, author, message

def getDataPointios(line):
	splitLine = line.split('] ')
	dateTime = splitLine[0]
	if ',' in dateTime:
		date, time = dateTime.split(',')
	else:
		date, time = dateTime.split(' ')
	message = ' '.join(splitLine[1:])
	if FindAuthor(message):
		splitMessage = message.split(':')
		author = splitMessage[0]
		message = ' '.join(splitMessage[1:])
	else:
		author = None
	if time[5]==":":
		time = time[:5]+time[-3:]
	else:
		if 'AM' in time or 'PM' in time:
			time = time[:6]+time[-3:]
		else:
			time = time[:6]
	return date, time, author, message


# In[ ]:


def dateconv(date):
  year=''
  if '-' in date:
    year = date.split('-')[2]
    if len(year) == 4:
      return datetime.datetime.strptime(date, "[%d-%m-%Y").strftime("%Y-%m-%d")
    elif len(year) ==2:
      return datetime.datetime.strptime(date, "[%d-%m-%y").strftime("%Y-%m-%d")
  elif '/' in date:
    year = date.split('/')[2]
    if len(year) == 4:
      return datetime.datetime.strptime(date, "[%d/%m/%Y").strftime("%Y-%m-%d")
    if len(year) ==2:
      return datetime.datetime.strptime(date, "[%d/%m/%y").strftime("%Y-%m-%d")


# In[ ]:


def split_count(text):

    emoji_list = []
    data = regex.findall(r'\X', text)
    for word in data:
        if any(char in emoji.UNICODE_EMOJI for char in word):
            emoji_list.append(word)

    return emoji_list


# In[ ]:


parsedData = [] # List to keep track of data so it can be used by a Pandas dataframe
conversationPath = 'C:/PP/Test2.txt' 
with open(conversationPath, encoding="utf-8") as fp:
    device=''
    first=fp.readline()
    print(first)
    if '[' in first:
      device='ios'
    else:
      device="android"
    fp.readline() 
    messageBuffer = [] 
    date, time, author = None, None, None
    while True:
        line = fp.readline() 
        if not line: 
            break
        if device=="ios":
          line = line.strip()
          if startsWithDateAndTimeios(line):
            if len(messageBuffer) > 0:
              parsedData.append([date, time, author, ' '.join(messageBuffer)])
            messageBuffer.clear()
            date, time, author, message = getDataPointios(line)
            messageBuffer.append(message)
          else:
            line= (line.encode('ascii', 'ignore')).decode("utf-8")
            if startsWithDateAndTimeios(line):
              if len(messageBuffer) > 0:
                parsedData.append([date, time, author, ' '.join(messageBuffer)])
              messageBuffer.clear()
              date, time, author, message = getDataPointios(line)
              messageBuffer.append(message)
            else:
              messageBuffer.append(line)
        else:
          line = line.strip()
          if startsWithDateAndTimeAndroid(line):
            if len(messageBuffer) > 0:
              parsedData.append([date, time, author, ' '.join(messageBuffer)])
            messageBuffer.clear()
            date, time, author, message = getDataPointAndroid(line)
            messageBuffer.append(message)
          else:
            messageBuffer.append(line)


# In[ ]:


if device =='android':
        df = pd.DataFrame(parsedData, columns=['Date', 'Time', 'Author', 'Message'])
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.dropna()
        df["emoji"] = df["Message"].apply(split_count)
        URLPATTERN = r'(https?://\S+)'
        df['urlcount'] = df.Message.apply(lambda x: re.findall(URLPATTERN, x)).str.len()
else:
        df = pd.DataFrame(parsedData, columns=['Date', 'Time', 'Author', 'Message']) # Initialising a pandas Dataframe.
        df = df.dropna()
        df["Date"] = df["Date"].apply(dateconv)
        df["Date"] = pd.to_datetime(df["Date"],format='%Y-%m-%d')
        df["emoji"] = df["Message"].apply(split_count)
        URLPATTERN = r'(https?://\S+)'
        df['urlcount'] = df.Message.apply(lambda x: re.findall(URLPATTERN, x)).str.len()


# In[ ]:





# In[ ]:


df.head(2)
df = df.dropna()
total_messages = df.shape[0]
print(total_messages)


# In[ ]:


media_messages = df[df['Message'] == '<Media omitted>'].shape[0]
print(media_messages)


# In[ ]:


emojis = sum(df['emoji'].str.len())
print(emojis)


# In[ ]:


URLPATTERN = r'(https?://\S+)'
df['urlcount'] = df.Message.apply(lambda x: re.findall(URLPATTERN, x)).str.len()


# In[ ]:


df


# In[ ]:


links = np.sum(df.urlcount)


# In[ ]:


df


# In[ ]:


f1=df['urlcount']
f1


# In[ ]:


print("Group Wise Stats")
print("Messages:",total_messages)
print("Media:",media_messages)
print("Emojis:",emojis)
print("Links:",links)


# In[ ]:


link_messages= df[df['urlcount']>0]
deleted_messages=df[(df["Message"] == " You deleted this message")| (df["Message"] == " This message was deleted.")|(df["Message"] == " You deleted this message.")]
media_messages_df = df[(df['Message'] == ' <Media omitted>')|(df['Message'] == ' image omitted')|(df['Message'] == ' video omitted')|(df['Message'] == ' sticker omitted')]
messages_df = df.drop(media_messages_df.index)
messages_df = messages_df.drop(deleted_messages.index)
messages_df = messages_df.drop(link_messages.index)


# In[ ]:


messages_df['Letter_Count'] = messages_df['Message'].apply(lambda s : len(s))
messages_df['Word_Count'] = messages_df['Message'].apply(lambda s : len(s.split(' ')))
messages_df["MessageCount"]=1


# In[ ]:


messages_df.head(2)


# In[ ]:



messages_df2=messages_df.groupby(['Author']).count()['emoji']
messages_df2


# In[ ]:


l = messages_df.Author.unique()

for i in range(len(l)):
  # Filtering out messages of particular user
  req_df= messages_df[messages_df["Author"] == l[i]]
  # req_df will contain messages of only one particular user
  print(f'Stats of {l[i]} -')
  # shape will print number of rows which indirectly means the number of messages
  print('Messages Sent', req_df.shape[0])
  #Word_Count contains of total words in one message. Sum of all words/ Total Messages will yield words per message
  words_per_message = (np.sum(req_df['Word_Count']))/req_df.shape[0]
  print('Words per message', words_per_message)
  #media conists of media messages
  media = media_messages_df[media_messages_df['Author'] == l[i]].shape[0]
  print('Media Messages Sent', media)
  # emojis conists of total emojis
  emojis = sum(req_df['emoji'].str.len())
  print('Emojis Sent', emojis)
  #links consist of total links
  links = sum(link_messages[link_messages['Author'] == l[i]]["urlcount"])   
  print('Links Sent', links)   
  print()


# In[ ]:


total_emojis_list = list(set([a for b in messages_df.emoji for a in b]))
total_emojis = len(total_emojis_list)
print(total_emojis)


# In[ ]:


total_emojis_list = list([a for b in messages_df.emoji for a in b])
emoji_dict = dict(Counter(total_emojis_list))
emoji_dict = sorted(emoji_dict.items(), key=lambda x: x[1], reverse=True)
print(emoji_dict)


# In[ ]:





# In[ ]:


emoji_df = pd.DataFrame(emoji_dict, columns=['emoji', 'count'])
emoji_df


# In[ ]:


df12=pd.DataFrame(df, columns=['Author'])


# In[ ]:



fig = px.pie(emoji_df, values='count', names='emoji')
fig.update_traces(textposition='inside', textinfo='percent+label')
fig.show()
plt.savefig('9.png')


# In[ ]:





# In[ ]:


l = messages_df.Author.unique()
for i in range(len(l)):
  dummy_df = messages_df[messages_df['Author'] == l[i]]
  total_emojis_list = list([a for b in dummy_df.emoji for a in b])
  emoji_dict = dict(Counter(total_emojis_list))
  emoji_dict = sorted(emoji_dict.items(), key=lambda x: x[1], reverse=True)
  print('Emoji Distribution for', l[i])
  author_emoji_df = pd.DataFrame(emoji_dict, columns=['emoji', 'count'])
  fig = px.pie(author_emoji_df, values='count', names='emoji')
  fig.update_traces(textposition='inside', textinfo='percent+label')
  fig.show()


# In[ ]:





# In[ ]:


def f(i):
  l = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
  return l[i];
day_df=pd.DataFrame(messages_df["Message"])
day_df['day_of_date'] = messages_df['Date'].dt.weekday
day_df['day_of_date'] = day_df["day_of_date"].apply(f)
day_df["messagecount"] = 1
day = day_df.groupby("day_of_date").sum()
day.reset_index(inplace=True)


# In[ ]:


fig = px.line_polar(day, r='messagecount', theta='day_of_date', line_close=True)
fig.update_traces(fill='toself')
fig.update_layout(
  polar=dict(
    radialaxis=dict(
      visible=True,
    )),
  showlegend=False
)
fig.show()
plt.savefig('11.png')


# In[ ]:


date_df = messages_df.groupby("Date").sum()
date_df.reset_index(inplace=True)
fig = px.line(date_df, x="Date", y="MessageCount")
fig.update_xaxes(nticks=20)
fig.show()
plt.savefig('12.png')


# In[ ]:


""""date_df["rolling"] = date_df["MessageCount"].rolling(30).mean()
fig = px.line(date_df, x="Date", y="rolling")
fig.update_xaxes(nticks=15)
fig.show()
plt.savefig('13.png')"""


# In[ ]:


auth = messages_df.groupby("Author").sum()
auth.reset_index(inplace=True)
fig = px.bar(auth, y="Author", x="MessageCount", color='Author', orientation="h",
             color_discrete_sequence=["red", "green", "blue", "goldenrod", "magenta"],
             title="Explicit color sequence"
            )

fig.show()
plt.savefig('14.png')


# In[ ]:


messages_df['Time'].value_counts().head(10).plot.barh() # Top 10 Times of the day at which the most number of messages were sent
plt.xlabel('Number of messages')
plt.ylabel('Time')
plt.savefig('15.png')


# In[ ]:


messages_df['Date'].value_counts().head(10).plot.barh();
print(messages_df['Date'].value_counts());
plt.xlabel('Number of Messages');
plt.ylabel('Date');


# In[ ]:


messages_df.iloc[messages_df['Word_Count'].argmax()]


# In[ ]:


text = " ".join(review for review in messages_df.Message)
print ("There are {} words in all the messages.".format(len(text)))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




