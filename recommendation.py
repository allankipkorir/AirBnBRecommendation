import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def get_id_from_name(name):
    return df[df['name'] == name]['id'].values[0]

def get_name_from_id(id):
    return df[df['id'] == id]['name'].values[0]

def get_combined_features_from_id(id):
    return df[df['id'] == id]['combined_features'].values[0]

def get_price_from_id(id):
    return df[df['id'] == id]['price'].values[0]

df = pd.read_csv('AB_NYC_2019.csv')

features = ['neighbourhood_group', 'neighbourhood', 'room_type', 'price_range', 'room_type']

for feature in features:
    df[feature] = df[feature].fillna('')

def combine_features(row):
    return row['neighbourhood_group']+" "+row['neighbourhood']+" "+row['room_type']+" "+str(row['price_range'])+" "+row['room_type']

df['combined_features'] = df.apply(combine_features, axis=1)

cv = CountVectorizer()

count_matrix = cv.fit_transform(df['combined_features'])

cosine_similarity = cosine_similarity(count_matrix)


listing_name = 'Beautiful 1 Bedroom in Nolita/Soho '
listing_id = get_id_from_name(listing_name)

similar_to_listing = list(enumerate(cosine_similarity[listing_id]))

similarity_sorted_similar_to_listing = sorted(similar_to_listing, key=lambda x:x[1], reverse=True)

# most similar to listing
print("\n\n\n") 
print(" SIMILAR TO LISTING") 
print("\n\n\n") 

i = 1
for listing in similarity_sorted_similar_to_listing:
    price = get_price_from_id(listing[0])
    print(get_name_from_id(listing[0]))
    if np.isnan(price) != True:
        print(" price:" + str(price)) 
    i=i+1
    if i > 200:
        break

# 200 listings most similar to listing, sorted by price

price_sorted_similar_to_listing = sorted(similar_to_listing, key=lambda x:x[1], reverse=True)
price_sorted_similar_to_listing = price_sorted_similar_to_listing[:200]

price_sorted_similar_to_listing = sorted(price_sorted_similar_to_listing, key=lambda x:get_price_from_id(x[0]), reverse=False)

print("\n\n\n") 
print(" 200 MOST SIMILAR TO LISTING, SORTED BY PRICE") 
print("\n\n\n") 

i = 1
for listing in price_sorted_similar_to_listing:
    price = get_price_from_id(listing[0])
    print(get_name_from_id(listing[0]))
    if np.isnan(price) != True:
        print(" price:" + str(price)) 