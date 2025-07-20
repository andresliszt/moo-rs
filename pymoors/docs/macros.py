def define_env(env):
    """
    Defines a macro `docs_rs(item_type, item_name)` that returns
    the URL to docs.rs/moors/<version>/moors/<type>.<item>.html
    """
    version = env.variables.get("moors_crate_version")
    base    = f"https://docs.rs/moors/{version}/moors/"

    @env.macro
    def docs_rs(item_type: str, path: str) -> str:
        parts = path.split(".")
        item_name, parts = parts[-1], parts[:-1]
        if parts:
            parts_url = "/".join(parts) + f"/{item_type}.{item_name}.html"
        else:
            parts_url = f"{item_type}.{item_name}.html"
        return base + parts_url
