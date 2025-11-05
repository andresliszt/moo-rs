def define_env(env):
    """
    Defines a macro `docs_rs(item_type, item_name)` that returns
    the URL to docs.rs/moors/<version>/moors/<type>.<item>.html
    """
    version = env.variables.get("moors_crate_version")
    base = f"https://docs.rs/moors/{version}/moors/"

    @env.macro
    def docs_rs(item_type: str, path: str, label: str | None = None, method: str | None = None, tymethod: str | None = None) -> str:
        parts = path.split(".")
        name, *rest = parts[::-1]
        # reconstruct the URL path
        if rest:
            parts_url = "/".join(parts[:-1]) + f"/{item_type}.{name}.html"
        else:
            parts_url = f"{item_type}.{name}.html"
        url = base + parts_url
        if method:
            url = url + f"#method.{method}"

        if tymethod:
            url = url + f"#tymethod.{method}"

        text = label or name
        # return raw HTML
        return f'<a href="{url}" target="_blank" rel="noopener">{text}</a>'
