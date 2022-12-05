import pickle
from .book import Book
import numpy as np
from typing import List

class Recommender:
    def __init__(self):
        """
        Creates an instance of the recommender system after loading the data.
        """
        self.rating_matrix = pickle.load(
            open("rec_sys/rec_data/rating_matrix.pkl", "rb")
        )
        self.books = pickle.load(open("rec_sys/rec_data/books.pkl", "rb"))
        self.similarity_scores = pickle.load(
            open("rec_sys/rec_data/similarity_scores.pkl", "rb")
        )
        self.book_names = pickle.load(open("rec_sys/rec_data/book_names.pkl", "rb"))

    def recommend(self, book_name: str) -> List[Book]:
        """
        Find similar books to the given book name.
        Parameters:
            book_name: str - name of the book.
        If the book is not found an exception will be raised.
        """
        book_idx = np.where(self.rating_matrix.index == book_name)[0][0]
        similar_items = list(enumerate(self.similarity_scores[book_idx]))
        similar_items.sort(reverse=True, key=lambda x: x[1])

        recommended_books = []
        for i in similar_items[1:6]:
            book_match_df = self.books[
                self.books["Book-Title"] == self.rating_matrix.index[i[0]]
            ].drop_duplicates("Book-Title")

            book = Book(
                book_match_df["Book-Title"].values[0],
                book_match_df['ISBN'].values[0],
                book_match_df["Book-Author"].values[0],
                book_match_df["Year-Of-Publication"].values[0],
                book_match_df["Publisher"].values[0],
                book_match_df["Image-URL-L"].values[0],
            )
            recommended_books.append(book)

        return recommended_books


recommender = Recommender()
