
# Installing pymoors from source


## Setting Up

Before you proceed, make sure you have **Rust** installed. We recommend using [rustup](https://rustup.rs/) for an easy setup:

```bash
# For Linux/Mac:
curl --proto '=https' --tlsv1.2 https://sh.rustup.rs -sSf | sh

# For Windows, download and run the installer from the official website:
# https://rustup.rs/
```

Also `pymoors` uses [uv](https://github.com/astral-sh/uv) . Make sure it's available on your PATH so the `make` commands can run properly.

Then

* **Clone the repository**
```sh
git clone https://github.com/andresliszt/pymoors
cd pymoors
```

* **Create a virtual environment**
```sh
# Create the virtual environment
uv venv

# Activate it (Linux/Mac)
source .venv/bin/activate

# On Windows, you would typically do:
# .venv\Scripts\activate
```

* **Install and compile in dev mode**
```sh
make build-dev
```

This command will install all the necessary dependencies for development and compile using [maturin](https://github.com/PyO3/maturin).

* **Compile in prod mode**
```bash
make build-prod
```
This command will run `maturin develop --release`, i.e compiling in a optimized way. This is needed when you want to check the performance of the different algorithms.

## Code Formatting and Linting

To ensure code quality and consistency, `pymoors` uses various tools for formatting and linting. You can run these tools using the provided `make` commands.

* **Format the code**
```sh
make format
```
This command will format the codebase using tools like `black` for Python and `rustfmt` for Rust.

* **Lint the code**
```sh
make lint
```
This command will run linters like `flake8` for Python and `clippy` for Rust to check for potential issues and enforce coding standards.

Additionally, `pymoors` uses `pre-commit` to automatically run these checks before each commit. To set up `pre-commit`, run the following command:

```sh
pre-commit install
```

This will ensure that your code is formatted and linted before every commit, helping maintain code quality throughout the development process.
