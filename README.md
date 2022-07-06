# Masader Web Service

## Development

### Requirements

#### Python

This code tested locally using Python `3.8` and deployed on production using Python `3.10`. So, you can assume that any Python version `>= 3.8` should work fine without any issue.

#### Redis

You should install Redis locally to run the application. You just need to follow the [official documentation](https://redis.io/docs/getting-started/installation) to install Redis. To make sure that the installation is working fine, you should run `redis-cli ping`, and it should return `PONG`.

#### Packages

You need to run the following command to install the required packages:

```
pip install -r requirements.txt -r requirements_dev.txt
```

#### Run the Server

To run the server and be able to work with all endpoints, you should set two environment variables:
- `HF_SECRET_KEY`: Secret key for HuggingFace APIs. You can get one by creating an account on huggingface.co and go to the [Access Tokens](https://huggingface.co/settings/tokens) page.
- `GH_SECRET_KEY`: Secret key for GitHub APIs. You can get one by creating an account on github.com and go to [Personal Access Tokens](https://github.com/settings/tokens) page.

Then, you can run `flask run` to start the server. Also, you can set the environment variables within the same command as follows:

```
HF_SECRET_KEY=<HuggingFace secret key here> GH_SECRET_KEY=<GitHub secret key here> flask run
```

To run the server with auto reloading, you can run the command with `FLASK_DEBUG` environment variable set to `1`. For example:

```
FLASK_DEBUG=1 flask run
```

#### Pull Requests

Make sure to apply `pre-commit` hooks before submitting any pull request by running `pre-commit run --all-files`. It should be ran automatically when you commit your changes, just double check that it ran before submitting the PR.

## Endpoints

### /datasets/schema

Entry: None
Returns the structure of the dataset in JSON format. The returned data is a list that descibe the columns of the dataset.

### /datasets

Entry: None
Returns the complete dataset in JSON format. The returned data is a list of lists. Each item on the list is an entry on the dataset that contains all the existing information.

### /datasets/[index]

Entry:
[index] Must be an integer from 1 to the size of the dataset (Inclusive).
Returns the entry with the specific index from the dataset.

### /datasets/tags

### /refresh

Entry: None
This point refresh the dataset.
Return: None
