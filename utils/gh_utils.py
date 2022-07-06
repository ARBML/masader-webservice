from flask import current_app as app
from github import Github

from constants import MASADER_GH_REPO


def create_issue(title: str, body: str) -> str:
    git_con = Github(app.config['GH_SECRET_KEY'])

    repo = git_con.get_repo(MASADER_GH_REPO)

    return repo.create_issue(title, body=body).html_url
