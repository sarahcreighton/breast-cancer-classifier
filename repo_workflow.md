# Repository Workflow

## Contents
[Repo Information](#repo-information) \
[Initial Setup](#initial-setup)\
[Branching](#branching)\
[Commits](#commits)\
[Pull Requests](#pull-requests)\
[Merging](#merging)\
[Typical Workflow](#doing-work)

## Repo Information
Do not to fork the repo, only [clone](#initial-setup) it. `main` has been set up as a protected branch with settings that restrict certain actions. In particular, merges to the main branch are only allowed after a [pull request](#pull-requests) has been approved by at least 1 other team member. No one (including the repo owner) is able to force [merge](#merging) to the main branch. If you try is should throw an error. Only the repo owner has access to change these settings. Contact them via Slack if there is an issue.

## Initial Setup
**Step 1**: Go to the project [repo](https://github.com/sarahcreighton/breast-cancer-classifier) on GitHub 
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

**Step 4**: Set up your `wdbc-env` environment
- open a terminal window
- navigate to the `breast-cancer-classifier` folder
- run the commands listed in [SETUP.md](./SETUP.md)
- make sure `wdbc-env` is active each time you are working on the repo
- the first time you run it, it will download any required packages
- a list of these packages can be found in [pyproject.toml](./pyproject.toml)

## Branching
- each feature should have it's own branch
- Only one person works on a given branch (unless otherwise specified)
- naming should start with your initials and then indicate what feature you're working on e.g., `ab-modeling` 
- once you are done working on that feature and all PRs have been merged with `main`, the branch can be deleted

**Set up a branch**:  
- first, make sure you are on the `main` branch
- check the status, pull updates to local if needed
- create a branch that has your initials and the feature you are working on
- do work on your feature branch
```zsh
git checkout main
git status          		# check active branch, should be main
git switch -c ab-modeling 	# create and switch branch
```

## Commits
- commit often
- short, informative message
- more detail can be included in a description section

## Pull Requests
- There is a custom template for the PRs that will automatically load any time you create a PR
- fill it in, tag group members if needed/desired
- you can include additional comments or screenshots/attachments if needed

## Merging
- all PRs to `main` must be reviewed by at least one team member and all comments on PRs must be resolved before a merge can happen
- if you attempt to force push from your local machine to main, it should throw an error

## Typical Workflow
Here's an example workflow. I did not try to cover multiple scenarios. Each time you go to work on your feature branch, check for updates to main first. If there are changes, pull to any of your local machine branches (e.g., local main and feature branch). Check for updates to the remote main branch before you push any changes to your feature branch to avoid merge conflicts. 
```zsh
git status						# active branch
git checkout main				# switch to main
git status						# check for updates
git pull origin main			# merge updates from remote to local main
git git checkout ab-modeling	# switch to feature branch
git pull origin main			# merge updates from remote main to local branch

# ... do work ...
git add "01_eda.ipynb"			# stage a file
git commit -m "added new plots"	# commit all staged changes

# check for updates to main before pushing your feature branch to remote
git switch main 	
git status			 
git pull origin main 			# update local main if needed
git switch ab-modeling
git pull origin main			# merge main to local feature branch if needed
git push origin ab-modeling		# push feature branch from local to remote
```

---
_Last Updated: 2026-02-28_