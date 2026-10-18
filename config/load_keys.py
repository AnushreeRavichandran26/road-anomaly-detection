"""Load secrets from config/keys.env into os.environ.

Usage from anywhere in the project:

    from config.load_keys import load_keys
    load_keys()
    api_key = os.environ["OPENWEATHER_API_KEY"]
"""
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_ENV_PATH = os.path.join(_HERE, "keys.env")


def load_keys(path=_ENV_PATH):
    """Read KEY=VALUE lines into os.environ. Existing env vars win."""
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{path} not found. Copy config/keys.env.example to config/keys.env "
            "and add your OpenWeatherMap key."
        )
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip())
    return os.environ
