#!/usr/bin/env python
# coding: utf-8

# # Bag of Words 

# In[1]:


import nltk 
import re
import heapq
import numpy as np


# In[33]:


paragraph = """The rising average temperature of Earth's climate system, called global warming, is driving changes in rainfall patterns, extreme weather, arrival of seasons, and more. Collectively, global warming and its effects are known as climate change. While there have been prehistoric periods of global warming, observed changes since the mid-20th century have been unprecedented in rate and scale.[1]
Observed temperature from NASA[2] vs the 1850–1900 average as a pre-industrial baseline. The primary driver for increased global temperatures in the industrial era is human activity, with natural forces adding variability.[3]

The Intergovernmental Panel on Climate Change (IPCC) concluded that "human influence on climate has been the dominant cause of observed warming since the mid-20th century". These findings have been recognized by the national science academies of major nations and are not disputed by any scientific body of national or international standing.[4] The largest human influence has been the emission of greenhouse gases, with over 90% of the impact from carbon dioxide and methane.[5] Fossil fuel burning is the principal source of these gases, with agricultural emissions and deforestation also playing significant roles. Temperature rise is enhanced by self-reinforcing climate feedbacks, such as loss of snow cover, increased water vapour, and melting permafrost.

Land surfaces are heating faster than the ocean surface, leading to heat waves, wildfires, and the expansion of deserts.[6] Increasing atmospheric energy and rates of evaporation are causing more intense storms and weather extremes, damaging infrastructure and agriculture.[7] Surface temperature increases are greatest in the Arctic and have contributed to the retreat of glaciers, permafrost, and sea ice. Environmental impacts include the extinction or relocation of many species as their ecosystems change, most immediately in coral reefs, mountains, and the Arctic. Surface temperatures would stabilize and decline a little if emissions were cut off, but other impacts will continue for centuries, including rising sea levels from melting ice sheets, rising ocean temperatures, and ocean acidification from elevated levels of carbon dioxide.[8]
Some effects of climate change

    Agricultural changes. Droughts, rising temperatures, and extreme weather negatively impact agriculture.

Energy flows between space, the atmosphere, and Earth's surface. Current greenhouse gas levels are causing a radiative imbalance of about 0.9 W/m2.[9]

Countries work together on climate change under the umbrella of the United Nations Framework Convention on Climate Change (UNFCCC), which has near-universal membership. The goal of the convention is to "prevent dangerous anthropogenic interference with the climate system". The IPCC has told policy makers that there is much greater risk to human and natural systems if warming goes above 1.5 °C (2.7 °F) compared to pre-industrial levels.[10] Under the Paris Agreement, nations are making climate pledges to reduce greenhouse gas (GHG) emissions, but those promises - assuming nations follow through - would still allow global warming to reach about 2.8 °C (5.0 °F) by 2100.[11] To limit warming to 1.5 °C (2.7 °F), methane emissions would need to decrease to near-zero levels and carbon dioxide emissions would need to reach net-zero by the year 2050.[12]

Mitigation efforts to address global warming include the development and deployment of low carbon energy technologies, policies to reduce fossil fuel emissions, reforestation, forest preservation, as well as the development of potential climate engineering technologies. Societies and governments are also working to adapt to current and future global warming impacts, including improved coastline protection, better disaster management, and the development of more resistant crops"""


# In[34]:


data_set = nltk.sent_tokenize(paragraph)


# In[35]:


data_set


# In[36]:


for i in range(len(data_set)):
    data_set[i] = data_set[i].lower()
    data_set[i] = re.sub(r"\W"," ",data_set[i])
    data_set[i] = re.sub(r"\s"," ",data_set[i])


# In[37]:


data_set


# In[38]:


text_str=''.join(data_set)


# In[39]:


text_str


# In[41]:


from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
le=WordNetLemmatizer()


# In[42]:


for i in range(len(text_str)):
    words=nltk.word_tokenize(text_str)
    words_lemm=[le.lemmatize(w) for w in words]


# In[43]:


#stop words 
for i in range(len(words_lemm)):
    #print(words_lemm[i])
    words_stop=[w for w in words_lemm if w not in stopwords.words('english')]


# In[44]:


text_after_preprocess=' '.join(words_stop)


# In[49]:


text_after_preprocess


# In[45]:


#Bag of words model 
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer

cv=CountVectorizer(max_features=20)
x=cv.fit_transform(words_stop).toarray()


# In[46]:


x


# # TFIDF MODEL

# In[47]:


tfidf=TfidfVectorizer(max_features=10)
x1=tfidf.fit_transform(words_stop).toarray()


# In[48]:


x1


# In[ ]:





# # Word2vec

# In[2]:


import urllib
import bs4 as bs
from gensim.models import Word2Vec


# In[3]:


data = urllib.request.urlopen("https://en.wikipedia.org/wiki/Global_warming").read()


# In[4]:


soup = bs.BeautifulSoup(data,features="html.parser")


# In[5]:


text = ""


# In[6]:


for paragraph in soup.find_all('p'):
    text += paragraph.text 


# In[7]:


text


# In[8]:


text = re.sub(r"\[[0-9]*\]"," ",text)
text = re.sub(r"\s+"," ",text)
text = text.lower()
text = re.sub(r"\W"," ",text)
text = re.sub(r"\d"," ",text)
text = re.sub(r"\s+"," ",text)


# In[9]:


text


# In[10]:


sentences = nltk.sent_tokenize(text)


# In[11]:


sentences


# In[12]:


sentences = [nltk.word_tokenize(sentences) for sentences in sentences]


# In[13]:


model = Word2Vec(sentences,min_count=1)


# In[14]:


words = model.wv.vocab


# In[15]:


words


# In[23]:


vector = model.wv['temperature']


# In[24]:


vector


# In[27]:


similar = model.wv.most_similar("temperature")


# In[28]:


similar


# In[ ]:




