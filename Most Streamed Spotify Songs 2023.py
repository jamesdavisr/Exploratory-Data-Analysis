#!/usr/bin/env python
# coding: utf-8

# In[88]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# In[89]:


# Set style for seaborn
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

# Load dataset
file_path = 'spotify-2023.csv'
spotify_data = pd.read_csv(file_path, encoding='ISO-8859-1')


# In[90]:


# Initial Data Inspection
print("Dataset Shape:", spotify_data.shape)
print("\nDataset Info:")
spotify_data.info()

# Display first few rows
spotify_data.head()


# In[91]:


# Step 1: Data Cleaning and Preprocessing
# Convert 'streams' column to numeric by removing commas
spotify_data['streams'] = pd.to_numeric(spotify_data['streams'].str.replace(",", " "), errors='coerce')

# Check for missing values and drop rows with missing critical data (e.g., 'key')
missing_data = spotify_data.isnull().sum()
print("\nMissing Values in each Column:\n", missing_data[missing_data > 0])

spotify_data.dropna(subset=['key'], inplace=True)


# In[92]:


# Step 2: Descriptive Statistics
# Calculate summary statistics for numeric columns
numeric_summary = spotify_data.describe()
print("\nSummary Statistics:\n", numeric_summary)


# In[93]:


# Mean, Median, and Standard Deviation of Streams
streams_mean = spotify_data['streams'].mean()
streams_median = spotify_data['streams'].median()
streams_std = spotify_data['streams'].std()
print(f"\nStreams - Mean: {streams_mean}, Median: {streams_median}, Standard Deviation: {streams_std}")

# Distribution of released_year and artist_count
plt.figure()
sns.histplot(spotify_data['released_year'], bins=20, kde=True, color='skyblue')
plt.title("Distribution of Release Years")
plt.xlabel("Release Year")
plt.ylabel("Frequency")
plt.show()

plt.figure()
sns.histplot(spotify_data['artist_count'], bins=10, kde=True, color='coral')
plt.title("Distribution of Artist Count")
plt.xlabel("Number of Artists")
plt.ylabel("Frequency")
plt.show()


# In[94]:


# Step 3: Top Performers
# Track with highest streams
top_tracks = spotify_data[['track_name', 'streams']].sort_values(by='streams', ascending=False).head(5)
print("\nTop 5 Most Streamed Tracks:\n", top_tracks)


# In[95]:


# Top 5 most frequent artists
top_artists = spotify_data['artist(s)_name'].value_counts().head(5)
print("\nTop 5 Most Frequent Artists:\n",  top_artists )


# In[96]:


# Step 4: Temporal Trends
# Number of tracks released per year
yearly_release_count = spotify_data['released_year'].value_counts().sort_index()
plt.figure()
yearly_release_count.plot(kind='line', marker='o', color='green')
plt.title("Tracks Released Per Year")
plt.xlabel("Year")
plt.ylabel("Number of Tracks")
plt.show()

# Tracks released per month
monthly_release_count = spotify_data['released_month'].value_counts().sort_index()
plt.figure()
monthly_release_count.plot(kind='bar', color='purple')
plt.title("Tracks Released Per Month")
plt.xlabel("Month")
plt.ylabel("Number of Tracks")
plt.show()


# In[97]:


# Step 5: Correlations Between Streams and Music Characteristics
# Heatmap of correlations
correlation_matrix = spotify_data[['streams', 'bpm', 'danceability_%', 'valence_%', 'energy_%']].corr()
plt.figure()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix of Streams and Musical Attributes")
plt.show()

# Correlation between danceability and energy
plt.figure()
sns.scatterplot(data=spotify_data, x='danceability_%', y='energy_%', color='teal')
plt.title("Danceability vs Energy")
plt.xlabel("Danceability (%)")
plt.ylabel("Energy (%)")
plt.show()


# In[98]:


# Step 6: Platform Popularity
# Compare track counts in playlists and charts across platforms
platform_counts = spotify_data[['in_spotify_playlists', 'in_spotify_charts', 'in_apple_playlists']].sum()
platform_counts.plot(kind='bar', color=['green', 'black', 'pink'])
plt.title("Platform Popularity Based on Playlist and Chart Inclusions")
plt.ylabel("Count")
plt.show()


# In[99]:


# Step 7: Advanced Analysis - Key and Mode
# Analyzing patterns by key and mode
key_mode_streams = spotify_data.groupby(['key', 'mode'])['streams'].mean().unstack()
key_mode_streams.plot(kind='bar', stacked=True, colormap='viridis')
plt.title("Average Streams by Key and Mode")
plt.ylabel("Average Streams")
plt.show()

# Analyzing genre and artist playlist appearances
# Top artists in Spotify playlists
top_spotify_artists = spotify_data.groupby('artist(s)_name')['in_spotify_playlists'].sum().sort_values(ascending=False).head(5)
print("\nTop Artists by Spotify Playlists Inclusion:\n", top_spotify_artists)

