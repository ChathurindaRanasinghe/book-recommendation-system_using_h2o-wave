from dataclasses import dataclass


@dataclass
class Book:
    def __init__(self, title: str, isbn: str,author: str, year: int, publisher: str, image: str):
        """
        Parameters:
            titile: str - Title of the book
            isbn: str - ISBN of the book
            author: str - Author of the book
            year: int - Year the book was published
            publisher: str - Publisher of book
            image: str - Link to the cover page of the book
        """
        self.title = title
        self.isbn = isbn
        self.author = author
        self.year = year
        self.publisher = publisher
        self.image = image
