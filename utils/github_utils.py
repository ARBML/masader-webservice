import requests

from github import Github
from flask import current_app as app
from constants import MASADER_GH_REPO

def report_issue(title: str, message:str):
    git_con = Github(app.config['GH_ACCESS_TOKEN'])
    
    repo = git_con.get_repo(MASADER_GH_REPO)
    repo.create_issue(title, body = message);