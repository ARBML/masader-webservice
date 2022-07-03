# masader-webservice

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
