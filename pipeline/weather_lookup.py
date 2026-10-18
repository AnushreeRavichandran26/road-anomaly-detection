import requests


def get_weather(lat, lon, api_key):
    """Fetch current weather for given latitude and longitude using OpenWeatherMap API."""
    if not api_key:
        return {"condition": "Unknown", "description": "No API key provided", "temp_c": None, "humidity": None}
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {"lat": lat, "lon": lon, "appid": api_key, "units": "metric"}
    try:
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        return {
            "condition": data["weather"][0]["main"],
            "description": data["weather"][0]["description"],
            "temp_c": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
        }
    except Exception as e:
        return {"condition": "Unknown", "description": f"fetch failed: {e}", "temp_c": None, "humidity": None}


if __name__ == "__main__":
    w = get_weather(13.0827, 80.2707, "")
    print("Weather lookup smoke test result (no key):", w)
