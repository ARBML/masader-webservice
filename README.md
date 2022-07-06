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
- Return Type: `JSON`
- Example Link: https://masader-web-service.herokuapp.com/datasets/schema
- Example Output:

```json
["Name", "Subsets", "HF Link", "Link", "License", "Year", ...]
```

### /datasets

- Method: `GET`
- Description: 
- Path Arguments: N/A
- Parameters: 
- Return Type: `JSON`
- Example Link: [https://masader-web-service.herokuapp.com/datasets?features=Name,Year,Unit&query=Year>2003 and Year<2008 and Unit=='tokens'](https://masader-web-service.herokuapp.com/datasets?features=Name,Year,Unit&query=Year>2003%20and%20Year<2008%20and%20Unit=='tokens')
- Example Output:

```json
```

### /datasets/[index]

- Method: `GET`
- Description: 
- Path Arguments: 
- Parameters: 
- Return Type: `JSON`
- Example Link: https://masader-web-service.herokuapp.com/datasets/1?features=Name,Year
- Example Output:

```json
```

### /datasets/tags

- Method: `GET`
- Description: 
- Path Arguments: N/A
- Parameters: 
- Return Type: `JSON`
- Example Link: https://masader-web-service.herokuapp.com/datasets/tags?features=Dialect,Year
- Example Output:

```json
```

### /datasets/[index]/issues

- Method: `POST`
- Description: 
- Path Arguments: 
- Parameters: N/A
- Return Type: `JSON`
- Example Link: https://masader-web-service.herokuapp.com/datasets/1/issues
- Example Output:

```json
```

### /refresh/[password]

- Method: `GET`
- Description: 
- Path Arguments: 
- Parameters: N/A
- Return Type: `JSON`
- Example Link: https://masader-web-service.herokuapp.com/refresh/[password]
- Example Output:

```json
```
