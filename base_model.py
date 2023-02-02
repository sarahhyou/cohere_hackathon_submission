import cohere
import numpy as np
import re
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
import umap
import altair as alt
from sklearn.metrics.pairwise import cosine_similarity
from annoy import AnnoyIndex
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_colwidth', None)

# SETTING UP THE QUESTION ARCHIVES:
# Get dataset
dataset = load_dataset("trec", split="train")
# Import into a pandas dataframe, take only the first 1000 rows
df = pd.DataFrame(dataset)[:1000]
# Preview the data to ensure it has loaded correctly
df.head(10)

# Paste your API key here. Remember to not share publicly
api_key = 'Nb1iI1R4zqDc84fhhRoOTs9nToz62DYgibtsakQa'

# Create and retrieve a Cohere API key from dashboard.cohere.ai/welcome/register
co = cohere.Client(api_key)

# Get the embeddings
embeds = co.embed(texts=list(df['text']),model='large',truncate='LEFT').embeddings

# Create the search index, pass the size of embedding
search_index = AnnoyIndex(embeds.shape[1], 'angular')
# Add all the vectors to the search index
for i in range(len(embeds)):
    search_index.add_item(i, embeds[i])
search_index.build(10) # 10 trees
search_index.save('test.ann')

#DOING THE ACTUAL SEARCHING:
query = "What is the tallest mountain in the world?"

# Get the query's embedding
query_embed = co.embed(texts=[query],
                  model="large",
                  truncate="LEFT").embeddings

# Retrieve the nearest neighbors
similar_item_ids = search_index.get_nns_by_vector(query_embed[0],10,
                                                include_distances=True)
# Format the results
results = pd.DataFrame(data={'texts': df.iloc[similar_item_ids[0]]['text'], 
                             'distance': similar_item_ids[1]})


print(f"Query:'{query}'\nNearest neighbors:")
print(results)