# Content-Based Movie Recommendation System

## Overview
This project implements a simple content-based recommendation system using TF-IDF vectorization and cosine similarity. Given a short text description of a user's movie preferences, the system returns the top 5 similar movies from a small dataset.

## Dataset
- **Source:** You can get a public movie dataset here https://www.kaggle.com/rounakbanik/the-movies-dataset. The original dataset has around 50K rows.
- I Randomly choose 500 rows in the original data and save to movies_dataset.csv.
- The CSV file (`movies_dataset.csv`) is the final dataset that I use which includes 500 rows with movie title and description

## Setup
- **Python Version:** Python 3.10
- **Setup Environment:** pip install -r requirements.txt

## Running the code:
- **Sample:** python recommend.py  "I like action movies set in space"
- You can replace the sentence in the ""

## Demo video
Can be found in this link: https://drive.google.com/file/d/1meRBvVC2EonjKQLzdcDZvRCmJPKzQ3J1/view?usp=sharing
