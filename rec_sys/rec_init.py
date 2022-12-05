import pandas as pd
import pickle
from sklearn.metrics.pairwise import cosine_similarity
from typing import Tuple


def read_dataset() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Read the datasets.
    """
    books = pd.read_csv("rec_sys/dataset/Books.csv", low_memory=False)
    ratings = pd.read_csv("rec_sys/dataset/Ratings.csv")

    return books, ratings


def save_data(rating_matrix, similarity_scores, books):
    """
    Saves rating matrix, books, book names, and similarity scores in pickle format.
    Parameters:
        rating_matrix: pandas.DataFrame - User rating for each book
        similarity_scores: 
        books: pandas.DataFrame - Details of the books
    """
    pickle.dump(
        list(rating_matrix.index), open("rec_sys/rec_data/book_names.pkl", "wb")
    )
    pickle.dump(rating_matrix, open("rec_sys/rec_data/books.pkl", "wb"))
    pickle.dump(books, open("rec_sys/rec_data/books.pkl", "wb"))
    pickle.dump(similarity_scores, open("rec_sys/rec_data/similarity_scores.pkl", "wb"))


def rec_init():
    """
    Computes the similarity scores based on collaborative filtering.
    Users that reviewed more than 200 books and books with equal or more than 50 ratings 
    are considered to improve the quality of recommendations. Similarity is measured based
    on cosine-similarity.
    """
    books, ratings = read_dataset()

    ratings_with_books = ratings.merge(books, on="ISBN")

    # Finding users with more than 200 reviews.
    ratings_group = ratings_with_books.groupby("User-ID").count()["Book-Rating"]
    ratings_group = ratings_group[ratings_group > 200]

    ratings_filtered = ratings_with_books[
        ratings_with_books["User-ID"].isin(ratings_group.index)
    ]

    # Finding books with equal or more than 50 ratings.
    filtered_group = ratings_filtered.groupby("Book-Title").count()["Book-Rating"]
    filtered_group = filtered_group[filtered_group >= 50]

    final_filtered_ratings = ratings_filtered[
        ratings_filtered["Book-Title"].isin(filtered_group.index)
    ]

    rating_matrix = final_filtered_ratings.pivot_table(
        index="Book-Title", columns="User-ID", values="Book-Rating"
    )
    rating_matrix.fillna(0, inplace=True)

    similarity_scores = cosine_similarity(rating_matrix)
    save_data(rating_matrix, similarity_scores, books)


if __name__ == "__main__":
    rec_init()
