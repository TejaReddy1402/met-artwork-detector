cd C:\Users\tejak\three_face_app

py app.py

We will make the following changes:

Public Registration: The main /register page will only create Homeowner accounts. The dropdown for roles will be removed.

Public Login: The main /login page will be for Homeowners only.

Company Login: We will create a new, "secret" URL (/company-login) for the single company user to log in.

Agency Rep Creation: The existing /create_agency_rep page will serve as the secret admin page for you to create agency accounts.
