# Setup
## Environment Setup Guide
Before using this repo, make sure you’ve completed the [environment setup guide](https://github.com/UofT-DSI/onboarding/blob/main/environment_setup/README.md), which installs the core tools you’ll need for this repository, such as:

- Git  
- Git Bash (for Windows)  
- Visual Studio Code
- UV

## Necessary Packages
This repo uses its own isolated environment called `wdbc-env` so that packages don’t conflict with other projects. 
We use UV to create this environment, activate it, and install the required packages listed in the repo’s `pyproject.toml`.  
This setup only needs to be done **once**, after that, you just activate the environment whenever you want to work in this repo.  


Open a terminal (macOS/Linux) or Git Bash (Windows) in this repo, and run the following commands in order:

1. Create a virtual environment called `wdbc-env`:
    ```
    uv venv wbcd-env --python 3.11
    ```

2. Activate the environment:
    - for macOS/Linux:
        ```
        source wdbc-env/bin/activate
        ```
        
    - for windows (git bash):    
        ```
        source wdbc-env/Scripts/activate
        ```

3. Install all required packages from the [pyproject.toml](./pyproject.toml)
    ```bash
    uv sync --active
    ```

## Environment Usage
In order to run any code in this repo, you must first activate its environment.
- for macOS/Linux:
    ```
    source wdbc-env/bin/activate
    ```
    
- for windows (git bash):    
    ```
    source wdbc-env/Scripts/activate
    ```

When the environment is active, your terminal prompt will change to show:  
```
(wdbc-env) $
```
This is your **visual cue** that you’re working inside the right environment.  

When you’re finished, you can deactivate it with:  
```bash
deactivate
```

> **👉 Remember**   
> Only one environment can be active at a time. If you switch to a different repo, first deactivate this one (or just close the terminal) and then activate the new repo’s environment.

---

If you have issues, view the `pyproject.toml` to see which libraries and dependencies are required. You may need to install them yourself. Good luck!

---
_Last Updated: 2026-02-27_