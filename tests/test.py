from h2o_wave import cypress


@cypress("body_test")
def body_test(cy):
    cy.visit("/recommender")
    cy.locate("search_box_input").type("The Notebook")
    cy.locate("search").click()
    cy.locate("msg_text").should(
        "contain.text", "If you read The Notebook you may also like!"
    )

    for i in range(1, 6):
        cy.locate(f"book{i}").should("exist")

    cy.locate("find_books").click()

    for i in range(1, 6):
        cy.locate(f"book{i}").should("not.exist")
        cy.locate(f"book_match{i}").should("exist")

    cy.locate("search_box_input").clear().type("abcdef")
    cy.locate("search").click()
    cy.locate("msg_text").should(
        "contain.text",
        "“abcdef” is not in our database or an invalid book name. Use the “Find Book” button to find books\n",
    )


@cypress("header_test")
def header_test(cy):
    cy.visit("/recommender")
    cy.locate("github_btn").click()


@cypress("footer_test")
def footer_test(cy):
    cy.visit("/recommender")
    cy.locate("linkedin").click()
    cy.locate("github").click()
