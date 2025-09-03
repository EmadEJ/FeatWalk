# fair_graph_embedding

*Emad says: this set-up guide was written with Gemini*

This guide provides the basic steps to set up a virtual environment and install the necessary dependencies for this project.

---

### ## 1. Create a Virtual Environment

First, navigate to the project's root directory. Then, create an isolated virtual environment using Python's built-in `venv` module.

```bash
python3 -m venv .venv
```

---

### ## 2. Activate the Virtual Environment

Before installing packages, you must activate the environment.

* **On macOS / Linux:**
    ```bash
    source .venv/bin/activate
    ```

* **On Windows (Command Prompt):**
    ```bash
    .venv\Scripts\activate
    ```

Your terminal prompt should now be prefixed with `(.venv)`.

---

### ## 3. Install Requirements

With the environment active, install the project's requirements.

*If you have a `requirements.txt` file, you can install it with the following command:*

```bash
pip install -r requirements.txt
```

---

### ## 4. Exiting the Environment

When you are finished working on the project, you can deactivate the environment with a single command.

```bash
deactivate
```