from rec_sys.rec_func import recommender
from h2o_wave import Q, main, app, ui
from fuzzywuzzy import process
from typing import Any


def search_books(book_name: str) -> list[tuple[Any, Any | int, Any] | tuple[Any, Any | int]]:
    """
    Find similar books name wise (using Levenshtein distance).
    """
    return process.extract(book_name, recommender.book_names)


@app("/recommender")
async def serve(q: Q):
    """
    Displays the recommended books according to the input.
    If the user cannot find the book, user can find the books that matches to the given input.
    """
    msg = ""
    
    if not q.client.initialized:
        q.client.initialized = True
    
    if q.args.search:
        del q.page["books"]
        q.args.search_box_input = q.args.search_box_input.strip()
        if q.args.search_box_input in recommender.book_names:
            result = recommender.recommend(q.args.search_box_input)

            msg = f"If you read {q.args.search_box_input} you may also like!"
            add_book_cards(result, q)

        elif q.args.search_box_input is None or q.args.search_box_input == "":
            msg = "Book name cannot be blanked."

        else:
            msg = f'"{q.args.search_box_input}" is not in our database or an invalid book name.\
            Use the "Find Book" button to find books'

    if q.args.find_books:
        q.args.search_box_input = q.args.search_box_input.strip()
        if q.args.search_box_input is None or q.args.search_box_input == "":
            msg = "Book name cannot be blanked."
        else:
            for i in range(1, 6):
                del q.page[f"book{i}"]
            add_similar_books(q)

    add_search_box(q, msg)
    add_header(q)
    add_footer(q)

    await q.page.save()


def add_similar_books(q: Q):
    similar_books = search_books(q.args.search_box_input)
    q.page["books"] = ui.form_card(
        box="2 4 10 7",
        items=[
            ui.copyable_text(
                value=book[0],
                name=f"book_match{i+1}",
                label=f"{book[1]}% match",
            )
            for i, book in enumerate(similar_books)
        ],
    )


def add_header(q: Q):
    q.page["header"] = ui.header_card(
        box="2 1 10 1",
        title="Book Recommendation System",
        subtitle="Collaborative filtering based recommender system",
        icon="Document",
        items=[
            ui.link(
                name="github_btn",
                path="https://github.com/ChathurindaRanasinghe/book-recommendation-system_using_h2o-wave.git",
                label="GitHub",
                button=True,
            )
        ],
    )


def add_book_cards(result, q: Q):
    for i in range(1, 6):
        q.page[f"book{i}"] = ui.tall_article_preview_card(
            box=f"{2*i} 4 2 7",
            title=f"{result[i-1].title}",
            subtitle=f"{result[i-1].author}",
            value=f"{result[i-1].year}",
            name="tall_article",
            image=f"{result[i-1].image}",
            items=[
                ui.text(f"{result[i-1].publisher}", size="l"),
                ui.text(f"ISBN: {result[i-1].isbn}", size="m"),
            ],
        )


def add_search_box(q: Q, msg):
    q.page["search_box"] = ui.form_card(
        box="2 2 10 2",
        items=[
            ui.textbox(
                name="search_box_input",
                label="Book Name",
                value=q.args.search_box_input,
            ),
            ui.buttons(
                items=[
                    ui.button(
                        name="search",
                        label="Search",
                        primary=True,
                        icon="BookAnswers",
                    ),
                    ui.button(name="find_books", label="Find Book", primary=False),
                ]
            ),
            ui.text(msg, size="m", name="msg_text"),
        ],
    )


def add_footer(q: Q):
    caption = """__Made with 💛 by Chathurinda Ranasinghe__ <br /> using __[h2o Wave](https://wave.h2o.ai/docs/getting-started).__"""
    q.page["footer"] = ui.footer_card(
        box="2 11 10 2",
        caption=caption,
        items=[
            ui.inline(
                justify="end",
                items=[
                    ui.links(
                        label="Contact Me",
                        width="200px",
                        items=[
                            ui.link(
                                name="github",
                                label="GitHub",
                                path="https://github.com/ChathurindaRanasinghe/book-recommendation-system_using_h2o-wave.git",
                                target="_blank",
                            ),
                            ui.link(
                                name="linkedin",
                                label="LinkedIn",
                                path="https://www.linkedin.com/in/chathurindaranasinghe/",
                                target="_blank",
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )
