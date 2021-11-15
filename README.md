# ECS 171 Group Project - CT Scan (COVID-19)
## Repository Guidelines

### Directory Structure

```
├── model
│   └── con_model.py # Trains model using preprocessed dataset
├── processed_data
│   ├── preprocessing.py # Preprocesses the raw dataset in raw_data
├── raw_data # NOTE: Generated Folder. Not in the repository by default
│   ├── original # Original dataset
│   └── modified # Modified dataset
└── README.md
└── fetch-data.sh # Downloads the raw dataset
```

### Running Model on CSIF

1. Log into CSIF via the command line on any of the following computers: PC21 - PC25. These computers are the only ones with libraries that allow Tensorflow to work with the GPU.
2. Clone this repository by running: `git clone https://github.com/ECS171-Team-15/ECS171-Team-15`. Enter your Github username and password when prompted. If git asks you to create a [token](https://github.com/settings/tokens), you can generate one as your password that only works on CSIF. Also, if you don't want git to prompt you to enter your username and password every time, you may run `git config --global credential.helper store` to [save](https://stackoverflow.com/a/12240995) your password in a file located in your `home` directory.
3. Once the repository has been cloned, download the original and modified datasets by running:
```
cd ECS171-Team-15
bash fetch-data.sh
```
4. Preprocess the data by running:
```
cd processed-data
python3 preprocessing.py modified|original
```
This will generate a csv file with our preprocessed data.

4. Train the model. The source code contains the hyperparameters. By default, it will use `original.csv` as its preprocessed dataset.
```
python3 con_model.py
```

### Making Changes to Code

Changes to the repository are not made directly. Instead:

1. Create a fork of this repository. This is a copy of the repository that you may edit yourself!

![Screenshot from 2021-10-15 13-10-15](https://user-images.githubusercontent.com/72328335/137548021-7484b22c-ec27-404e-85b6-ba074361549d.png)

2. Clone your fork to your local system.
3. Make and commit your changes to your fork.
4. Push your changes to your remote repository on Github.
5. To apply your changes, submit a pull request to this ![repository](https://github.com/jndnguyen123/ECS171-Team-15). Before doing this, make sure your fork is updated by fetching (retrieving the latest changes from this repository) and rebasing those changes to your fork. That way, your changes will be applied to the latest version of this repository.
6. Wait for the team leader to approve and merge your changes. (Merging means that your changes are finally applied to this repository)

**Modifying Your Pull Request**

If your changes aren't approved, you can create more commits (steps 3 and 4 above) and they will automatically show up in your commit. (*You don't have to close your pull request and create a new one.*)

**Commits**

By convention, your commit messages should be in the present tense. For example, the commit message:

`Implemented gradient descent`

should instead be:

`Implement gradient descent`
