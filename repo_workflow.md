# Repository Workflow

## Repo Setup
The instructions say not to fork the repo, only clone it. `main` has been set up as a protected branch with the following settings: 

Only the repo owner has access to change these settings. Contact them via Slack if there is an issue.

an organized workflow, including disabling pushing directly to main (i.e., main is a protected)

### A. Initial Setup
**Step 1**: Go to repo on GitHub
- [https://github.com/Rajesh-Detroja/Breast_Cancer_Diagnostics](https://github.com/sarahcreighton/breast-cancer-classifier)
- click dropdown green button that says <> Code and copy the https link
	- https://github.com/sarahcreighton/breast-cancer-classifier.git

**Step 2**: Clone to local machine using web url
- using bash/zsh navigate to the folder you want to clone the repo to
- clone the repository using the web url
- navigate to the cloned repo folder
```zsh
cd [your_folder]
git clone https://github.com/sarahcreighton/breast-cancer-classifier.git
cd ./breast-cancer-classifier
git status
```

**Step 3**: Sync local clone with remote repo
- while in the project folder `breast-cancer-classifier` check what branch is active
- make sure your local repo has the most up to date version of `main`
```zsh
git status              # check what branch is active
git checkout main       # or git switch main
git pull origin main    # git fetch origin main + git merge origin/main
```

**Step 4**: Set up your branch 
- first, make sure you are on the main branch
```zsh
git checkout main
git status          # check active branch, should be main
git switch -c ab-modeling # create and switch branch

```


## Commits
- commit often
- short, informative message
- more detail can be included in a description section

## Branching
- Only one person works on a given branch 
- It should start with your initials and then indicate what feature you're working on. For example, `ab-eda`. 

## Pull Requests
- There is a custom template for the PRs. 

## Merging