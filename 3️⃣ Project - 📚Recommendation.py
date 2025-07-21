#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[26]:


books = pd.read_csv("Books.csv")
users = pd.read_csv("Users.csv")
ratings = pd.read_csv("Ratings.csv")


# In[27]:


books.head()


# In[28]:


users.head()


# In[29]:


ratings.head()


# In[30]:


# dimension of dataset
print(f'''\t  books shape is {books.shape}
          ratings shape is {ratings.shape}
          users shape is {users.shape}''')


# In[31]:


books = books[['ISBN','Book-Title', 'Book-Author', 'Year-Of-Publication','Publisher','Image-URL-L']]
books.head(2)


# In[32]:


books.rename(columns={
    'Book-Title': 'title',
    'Book-Author': 'author',
    'Year-Of-Publication': 'year',
    'Publisher': 'publisher',
    'Image-URL-L': 'img_url'
},
             inplace=True)


# In[33]:


books.head(2)


# In[34]:


ratings.rename(columns={
    'User-ID' : 'user_id',
    'Book-Rating' : 'rating',
    
}, inplace=True)


# In[35]:


books.isnull().sum()


# In[36]:


users.isnull().sum()


# In[37]:


ratings.isnull().sum()


# In[38]:


books.duplicated().sum()


# In[39]:


ratings.duplicated().sum()


# In[40]:


users.duplicated().sum()


# ### Users_Dataset

# In[41]:


plt.hist(users['Age'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Count')
plt.grid()


# In[42]:


sns.boxplot(data=users['Age'])
plt.title('outliers in Age Column')


# In[43]:


print(sorted(users.Age.unique()))


# ***we have Outlier data in Age***

# In[44]:


users.Location.unique()


# In[45]:


users.Location.nunique()


# ***57339 unique Value it's really hard to understand***
# ***So lets create column Country***

# In[46]:


for i in users:
    users['Country']=users.Location.str.extract(r'\,+\s?(\w*\s?\w*)\"*$')  
    


# In[47]:


users.Country.nunique()


# In[48]:


#drop location column
users.drop('Location',axis=1,inplace=True)


# In[49]:


users.columns


# In[50]:


users.rename(columns={
    'User-ID' : 'user_id',
    'Age' : 'age',
    'Country' : 'country'
    
}, inplace=True)


# In[51]:


users.head(2)


# In[52]:


users.isnull().sum()


# In[53]:


users['country']=users['country'].astype('str')


# In[54]:


a=list(users.country.unique())
a = [x for x in a if x is not None]
a.sort()
print(a)


# In[61]:


users['country'].replace(['','01776','02458','19104','23232','30064','85021','87510','alachua','america','austria','autralia','cananda','geermany','italia','united kindgonm','united sates','united staes','united state','united states','us'],
                           ['other','usa','usa','usa','usa','usa','usa','usa','usa','usa','australia','australia','canada','germany','italy','united kingdom','usa','usa','usa','usa','usa'],inplace=True)


# In[62]:


users


# In[68]:


order=pd.value_counts(users['country'])


# In[69]:


plt.figure(figsize=(15, 8))
sns.countplot(y='country',
              data=users,
              order=pd.value_counts(users['country']).iloc[:10].index)
plt.title('Count of users Country wise')


# **Most number of users are from USA**
# ***Let's treat outliers in users age***

# In[70]:


sns.distplot(users['age'])
plt.title('Age Distribution Plot')


# ***Age value's below 5 and above 80 do not make much sense for our book rating case...hence replacing these by NaNs***

# In[71]:


# outlier data became NaN
users.loc[(users.age > 80) | (users.age < 5), 'age'] = np.nan


# In[72]:


users.isna().sum()


# ***Age has positive Skewness (right tail) so we can use median to fill Nan values, but for this we don't like to fill Nan value just for one range of age. To handle this we'll use country column to fill Nan***

# In[33]:


users['age'] = users['age'].fillna(users.groupby('country')['age'].transform('median'))


# In[34]:


users.isna().sum()


# ***Still we have 277 Nan values let's fill them with mean***

# In[35]:


users['age'].fillna(users.age.mean(),inplace=True)


# In[36]:


users.isna().sum()


# ### Books_Dataset

# In[37]:


books.head(2)


# In[77]:


order_b=pd.value_counts(books['title'])
order_b


# In[38]:


# Top 10 Title which have written the most books

plt.figure(figsize=(15, 8))
sns.countplot(data=books,
              y='title',
              order=pd.value_counts(books['title']).iloc[:10].index)
plt.title('Top 10 Title')
plt.show()


# In[39]:


# Top 10 Authors which have written the most books

plt.figure(figsize=(15, 8))
sns.countplot(data=books,
              y='author',
              order=pd.value_counts(books['author']).iloc[:10].index)
plt.title('Top 10 Authors')
plt.show()


# In[40]:


# Top 10 Publisher which have published the most books

plt.figure(figsize=(15, 8))
sns.countplot(data=books,
              y='publisher',
              order=pd.value_counts(books['publisher']).iloc[:10].index)
plt.title('Top 10 Publisher')
plt.show()


# In[41]:


books['year']=books['year'].astype('str')
a=list(books['year'].unique())

a = [x for x in a if x is not None]
a.sort()
print(a)


# In[42]:


#investigating the rows having 'DK Publishing Inc' as yearOfPublication
books.loc[books['year'] == 'DK Publishing Inc',:]


# ***As it can be seen from above that there are some incorrect entries in Year field. It looks like Publisher names 'DK Publishing Inc' and 'Gallimard' have been incorrectly loaded as Year-Of-Publication in dataset due to some errors in csv file***

# In[43]:


#From above, it is seen that bookAuthor is incorrectly loaded with bookTitle, hence making required corrections
#ISBN '0789466953'
books.loc[books.ISBN == '0789466953','year'] = 2000
books.loc[books.ISBN == '0789466953','author'] = "James Buckley"
books.loc[books.ISBN == '0789466953','publisher'] = "DK Publishing Inc"
books.loc[books.ISBN == '0789466953','title'] = "DK Readers: Creating the X-Men, How Comic Books Come to Life (Level 4: Proficient Readers)"

#ISBN '078946697X'
books.loc[books.ISBN == '078946697X','year'] = 2000
books.loc[books.ISBN == '078946697X','author'] = "Michael Teitelbaum"
books.loc[books.ISBN == '078946697X','publisher'] = "DK Publishing Inc"
books.loc[books.ISBN == '078946697X','title'] = "DK Readers: Creating the X-Men, How It All Began (Level 4: Proficient Readers)"

#rechecking
books.loc[(books.ISBN == '0789466953') | (books.ISBN == '078946697X'),:]


# In[44]:


#investigating the rows having 'Gallimard' as yearOfPublication
books.loc[books['year'] == 'Gallimard',:]


# In[45]:


#making required corrections as above, keeping other fields intact
books.loc[books.ISBN == '2070426769','year'] = 2003
books.loc[books.ISBN == '2070426769','author'] = "Jean-Marie Gustave Le ClÃ?Â©zio"
books.loc[books.ISBN == '2070426769','publisher'] = "Gallimard"
books.loc[books.ISBN == '2070426769','title'] = "Peuple du ciel, suivi de 'Les Bergers"


books.loc[books.ISBN == '2070426769',:]


# In[46]:


books['year']=pd.to_numeric(books['year'], errors='coerce')

print(sorted(books['year'].unique()))
#Now it can be seen that yearOfPublication has all values as integers


# ***The value 0 for Year is invalid  We have assumed that the years after 2021 to be invalid and setting invalid years as NaN***

# In[47]:


books.loc[(books['year'] > 2021) | (books['year'] == 0),'year'] = np.NAN

#replacing NaNs with median value of Year-Of-Publication
books['year'].fillna(round(books['year'].median()), inplace=True)


# In[48]:


books.isna().sum()


# In[49]:


#exploring 'publisher' column
books.loc[books.publisher.isnull(),:]
#two NaNs


# In[50]:


#Filling Nan of Publisher with others
books.publisher.fillna('Novelbooks Inc',inplace=True)


# In[51]:


#exploring 'Book-Author' column
books.loc[books['author'].isnull(),:]


# In[52]:


#Filling Nan of author with author's name
books['author'].fillna('Larissa Anne',inplace=True)


# In[53]:


books.isna().sum()


# ### Ratings_Dataset

# In[54]:


ratings.head(2)


# **Ratings dataset should have books only which exist in our books dataset**

# In[55]:


ratings_new = ratings[ratings.ISBN.isin(books.ISBN)]
ratings.shape,ratings_new.shape


# **It can be seen that many rows having book ISBN not part of books dataset got dropped off**
# 
# **Ratings dataset should have ratings from users which exist in users dataset.**

# In[56]:


print("Shape of dataset before dropping",ratings_new.shape)

ratings_new = ratings_new[ratings_new['user_id'].isin(users['user_id'])]
print("shape of dataset after dropping",ratings_new.shape)


# **It can be seen that no new user was there in ratings dataset.**

# In[57]:


ratings_new.isna().sum()


# In[58]:


plt.style.use('dark_background')
plt.figure(figsize=(10, 4))
sns.countplot(data=ratings, x='rating',palette='coolwarm')
plt.title('Book Rating Distribution')
plt.ylabel('Count')


# **Most of the Rating are '0'**

# ***The ratings are very unevenly distributed.As quoted in the description of the dataset -Book-Ratings contains the book rating information. Ratings are either explicit, expressed on a scale from 1-10 higher values denoting higher appreciation, or implicit, expressed by 0.Hence segragating implicit and explict ratings datasets***

# In[59]:


#Hence segragating implicit and explict ratings datasets
ratings_explicit = ratings_new[ratings_new['rating'] != 0]
ratings_implicit = ratings_new[ratings_new['rating'] == 0]


# In[60]:


print('ratings_explicit dataset shape',ratings_explicit.shape)
print('ratings_implicit dataset',ratings_implicit.shape)


# In[61]:


plt.figure(figsize=(10, 4))
sns.countplot(data=ratings_explicit , x='rating', palette='coolwarm')
plt.title('Book Rating Distribution from 1 - 10')
plt.ylabel('Count')


# **It can be observe that higher ratings are more common amongst users and rating 8 has been rated highest number of times**
# 
# 

# In[62]:


# Let's find the top 5 books which are rated by most number of users.
rating_count = pd.DataFrame(ratings_explicit.groupby('ISBN')['rating'].count())
rating_count.sort_values('rating', ascending=False).head()


# In[63]:


most_rated_books = pd.DataFrame(['0316666343', '0971880107', '0385504209', '0312195516', '0060928336'], index=np.arange(5), columns = ['ISBN'])
most_rated_books_summary = pd.merge(most_rated_books, books, on='ISBN')
most_rated_books_summary


# ***The book that received the most rating counts in this data set is Rich Shapero’s “Wild Animus”. And there is something in common among these five books that received the most rating counts — they are all novels. So it is conclusive that novels are popular and likely receive more ratings.***

# In[64]:


# Create column Rating average
ratings_explicit['Avg_Rating'] = ratings_explicit.groupby('ISBN')['rating'].transform('mean')
# Create column Rating sum
ratings_explicit['Total_No_Of_Users_Rated'] = ratings_explicit.groupby(
    'ISBN')['rating'].transform('count')


# In[65]:


ratings_explicit.head()


# In[66]:


ratings_explicit['user_id'].value_counts()


# In[67]:


ratings_explicit['user_id'].unique().shape


# In[68]:


x  = ratings_explicit['user_id'].value_counts() > 50


# In[69]:


x[x].shape


# In[70]:


x[x].index


# In[71]:


y = x[x].index


# In[72]:


ratings_explicit['user_id'].shape


# In[73]:


ratings_explicit = ratings_explicit[ratings_explicit['user_id'].isin(y)]


# In[74]:


ratings_explicit.head(2)


# In[75]:


ratings_explicit.shape


# In[76]:


ratings_explicit


# In[ ]:





# ## Merging All Dataset

# In[77]:


users.columns


# In[78]:


ratings_explicit.columns


# In[79]:


books.columns


# In[80]:


final_dataset = users.copy()
final_dataset = pd.merge(final_dataset, ratings_explicit, on='user_id')
final_dataset = pd.merge(final_dataset, books, on='ISBN')


# In[81]:


final_dataset.head(2)


# In[82]:


final_dataset.shape


# In[83]:


final_dataset.sample(10)


# In[84]:


final_dataset.isna().sum()


# In[85]:


final_dataset.shape


# In[86]:


df = final_dataset.copy()


# In[87]:


df.head(3)


# In[91]:


# Image printing for random books
import requests
from PIL import Image
from IPython.display import display
import random


# Shuffle the dataframe to get a random order
df = df.sample(frac=1).reset_index(drop=True)

# Select a subset of random books (e.g., 5 books)
random_books = df.sample(5)

# Display the images of the random books
for index, row in random_books.iterrows():
    isbn = row['ISBN']
    api_url = f'https://www.googleapis.com/books/v1/volumes?q=isbn:{isbn}'

    try:
        response = requests.get(api_url)
        response.raise_for_status()
        book_data = response.json()

        if 'items' in book_data and 'volumeInfo' in book_data['items'][0]:
            volume_info = book_data['items'][0]['volumeInfo']
            title = volume_info['title']
            image_links = volume_info.get('imageLinks')

            if image_links:
                image_url = image_links['thumbnail']
                img = Image.open(requests.get(image_url, stream=True).raw)
                display(img)
                print(f"Book: {title} | Rating: {row['rating']}")
            else:
                print(f"No image available for book: {title}")

    except requests.exceptions.HTTPError as err:
        print(f"Failed to retrieve book details for ISBN: {isbn} - Error: {err}")

print("Image printing complete!")


# In[90]:


# Image printing for Top rating books
import pandas as pd
import requests
from PIL import Image
from IPython.display import display


# Sort the dataframe by the 'rating' column in descending order
sorted_df = df.sort_values(by='rating', ascending=False)

# Select the top 10 rated books
top7_books = sorted_df.head(7)

# Display the images of the top 7 rated books
for index, row in top7_books.iterrows():
    isbn = row['ISBN']
    api_url = f'https://www.googleapis.com/books/v1/volumes?q=isbn:{isbn}'

    try:
        response = requests.get(api_url)
        response.raise_for_status()
        book_data = response.json()

        if 'items' in book_data and 'volumeInfo' in book_data['items'][0]:
            volume_info = book_data['items'][0]['volumeInfo']
            title = volume_info['title']
            image_links = volume_info.get('imageLinks')

            if image_links:
                image_url = image_links['thumbnail']
                img = Image.open(requests.get(image_url, stream=True).raw)
                display(img)
                print(f"Book: {title} | Rating: {row['rating']}")
            else:
                print(f"No image available for book: {title}")

    except requests.exceptions.HTTPError as err:
        print(f"Failed to retrieve book details for ISBN: {isbn} - Error: {err}")

print("Image printing complete!")


# In[89]:


# Image printing for Low rating books

# Sort the dataframe by the 'rating' column 
sorted_df = df.sort_values(by='rating', ascending= True)

# Select the top 10 rated books
top7_books = sorted_df.head(7)

# Display the images of the low rated books
for index, row in top7_books.iterrows():
    isbn = row['ISBN']
    api_url = f'https://www.googleapis.com/books/v1/volumes?q=isbn:{isbn}'

    try:
        response = requests.get(api_url)
        response.raise_for_status()
        book_data = response.json()

        if 'items' in book_data and 'volumeInfo' in book_data['items'][0]:
            volume_info = book_data['items'][0]['volumeInfo']
            title = volume_info['title']
            image_links = volume_info.get('imageLinks')

            if image_links:
                image_url = image_links['thumbnail']
                img = Image.open(requests.get(image_url, stream=True).raw)
                display(img)
                print(f"Book: {title} | Rating: {row['rating']}")
            else:
                print(f"No image available for book: {title}")

    except requests.exceptions.HTTPError as err:
        print(f"Failed to retrieve book details for ISBN: {isbn} - Error: {err}")

print("Image printing complete!")


# In[ ]:




