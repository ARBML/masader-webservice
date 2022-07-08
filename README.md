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

- Method: `GET`
- Description: Returns the list of available features for the datasets.
- Path Arguments: N/A
- Parameters: N/A
- Data: N/A
- Return Type: `JSON`
- Example Link: https://masader-web-service.herokuapp.com/datasets/schema
- Example Output:

```json
[
    "Name",
    "Subsets",
    "HF Link",
    ...
]
```

### /datasets

- Method: `GET`
- Description: Returns the list of available datasets based on the passed `query` and the requested `features`.
- Path Arguments: N/A
- Parameters:
  - `query` (Optional): Filtration query will be applied on the dataset before selecting the required features and returning the output (e.g. `query=Year>2003 and Year<2008 and Unit=='tokens'`). The query language should follow [Pandas](https://pandas.pydata.org) query language, for more information see [here](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html).
  - `features` (Optional): The list of required features to be returned for each dataset (e.g. `features=Name,Year,Unit`).
- Data: N/A
- Return Type: `JSON`
- Example Link: [https://masader-web-service.herokuapp.com/datasets?features=Name,Year,Unit&query=Year>2003 and Year<2008 and Unit=='tokens'](https://masader-web-service.herokuapp.com/datasets?features=Name,Year,Unit&query=Year>2003%20and%20Year<2008%20and%20Unit=='tokens')
- Example Output:

```json
[
    {
        "Name": "LC-STAR: Standard Arabic Phonetic lexicon",
        "Unit": "tokens",
        "Year": 2007
    },
    {
        "Name": "NEMLAR: Written Corpus",
        "Unit": "tokens",
        "Year": 2006
    },
    {
        "Name": "Arabic Treebank: Part 3",
        "Unit": "tokens",
        "Year": 2005
    },
    ...
]
```

### /datasets/[index]

- Method: `GET`
- Description: Returns specific dataset from the available datasets based on its `index`.
- Path Arguments:
  - `index`: The index of the required dataset. The `index` should be within range `[1, maximum number of datasets in Masader]`.
- Parameters:
  - `features` (Optional): The list of required features to be returned for each dataset (e.g. `features=Name,Year`).
- Data: N/A
- Return Type: `JSON`
- Example Link: https://masader-web-service.herokuapp.com/datasets/1?features=Name,Year
- Example Output:

```json
{
    "Name": "Shami",
    "Year": 2018
}
```

### /datasets/tags

- Method: `GET`
- Description: Returns the unique values of the requested features.
- Path Arguments: N/A
- Parameters:
  - `features` (Optional): The list of required features to return their unique values (e.g. `features=Dialect,Year`).
- Data: N/A
- Return Type: `JSON`
- Example Link: https://masader-web-service.herokuapp.com/datasets/tags?features=Dialect,Year
- Example Output:

```json
{
    "Dialect": [
        "Algeria",
        "Bahrain",
        "Classic",
        ...
    ],
    "Year": [
        2001,
        2002,
        2003,
        ...
    ]
}
```

### /datasets/[index]/issues

- Method: `POST`
- Description: Creates a new GitHub issue related to the dataset that assoiated with `index`.
- Path Arguments:
  - `index`: The index of the required dataset. The `index` should be within range `[1, maximum number of datasets in Masader]`.
- Parameters: N/A
- Data:
  - `title`: The issue's title. This will be prefixed with the dataset name.
  - `body`: The issue's body.
- Return Type: `JSON`
- Example Link: https://masader-web-service.herokuapp.com/datasets/1/issues
- Example Output:

```json
{
    "issue_url": "https://github.com/ARBML/masader/issues/64"
}
```

### /refresh/[password]

- Method: `GET`
- Description: Refreshes the in-memory datasets and their tags, embeddings, and clusters.
- Path Arguments:
  - `password`: Simple string authentication to prevent anonymous actors from requesting this endpoint.
- Parameters: N/A
- Data: N/A
- Return Type: `JSON`
- Example Link: https://masader-web-service.herokuapp.com/refresh/123456
- Example Output:

```json
"The datasets updated successfully! The current number of available datasets is 590."
```
