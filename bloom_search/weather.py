"""No-key current-weather lookup backed by Open-Meteo."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Callable
from urllib.parse import urlencode
from urllib.request import Request, urlopen


JsonGetter = Callable[[str], dict[str, object]]


def get_json(url: str) -> dict[str, object]:
    request = Request(url, headers={"User-Agent": "AdaptiveBloomSearch/0.3"})
    with urlopen(request, timeout=15) as response:
        return json.loads(response.read().decode("utf-8"))


WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    56: "Light freezing drizzle",
    57: "Dense freezing drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snowfall",
    73: "Moderate snowfall",
    75: "Heavy snowfall",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


@dataclass(frozen=True, slots=True)
class WeatherResult:
    location: str
    observed_at: str
    condition: str
    temperature_f: float
    apparent_temperature_f: float
    humidity_percent: int
    precipitation_in: float
    wind_speed_mph: float
    latitude: float
    longitude: float

    @property
    def summary(self) -> str:
        return (
            f"{self.condition}. Temperature {self.temperature_f:.1f}°F, feels like "
            f"{self.apparent_temperature_f:.1f}°F. Humidity {self.humidity_percent}%. "
            f"Precipitation {self.precipitation_in:.2f} in. Wind {self.wind_speed_mph:.1f} mph. "
            f"Updated {self.observed_at}."
        )


class WeatherService:
    def __init__(self, json_getter: JsonGetter = get_json) -> None:
        self.json_getter = json_getter

    def current(self, location: str) -> WeatherResult:
        location = location.strip()
        if not location:
            raise ValueError("enter a city or postal code for a weather search")
        geo_url = "https://geocoding-api.open-meteo.com/v1/search?" + urlencode(
            {"name": location, "count": 1, "language": "en", "format": "json"}
        )
        geocoded = self.json_getter(geo_url)
        results = geocoded.get("results")
        if not isinstance(results, list) or not results or not isinstance(results[0], dict):
            raise ValueError(f"weather location not found: {location}")
        place = results[0]
        latitude = float(place["latitude"])
        longitude = float(place["longitude"])
        label_parts = [str(place.get("name", location))]
        if place.get("admin1"):
            label_parts.append(str(place["admin1"]))
        if place.get("country"):
            label_parts.append(str(place["country"]))
        forecast_url = "https://api.open-meteo.com/v1/forecast?" + urlencode(
            {
                "latitude": latitude,
                "longitude": longitude,
                "current": (
                    "temperature_2m,apparent_temperature,relative_humidity_2m,"
                    "precipitation,weather_code,wind_speed_10m"
                ),
                "temperature_unit": "fahrenheit",
                "wind_speed_unit": "mph",
                "precipitation_unit": "inch",
                "timezone": "auto",
            }
        )
        forecast = self.json_getter(forecast_url)
        current = forecast.get("current")
        if not isinstance(current, dict):
            raise ValueError("weather provider returned no current conditions")
        code = int(current["weather_code"])
        return WeatherResult(
            location=", ".join(label_parts),
            observed_at=str(current["time"]),
            condition=WEATHER_CODES.get(code, f"Weather code {code}"),
            temperature_f=float(current["temperature_2m"]),
            apparent_temperature_f=float(current["apparent_temperature"]),
            humidity_percent=int(current["relative_humidity_2m"]),
            precipitation_in=float(current["precipitation"]),
            wind_speed_mph=float(current["wind_speed_10m"]),
            latitude=latitude,
            longitude=longitude,
        )


def is_weather_query(query: str) -> bool:
    words = {word.strip("?!.,").lower() for word in query.split()}
    return bool(words & {"weather", "temperature", "forecast"})


def weather_location(query: str, fallback: str) -> str:
    """Extract an explicit location from a weather query, or use the form value."""
    match = re.search(r"\b(?:in|for|at)\s+(.+?)(?:\s+(?:today|now|currently))?[?!.]*$", query, re.I)
    if not match:
        return fallback.strip()
    location = match.group(1).strip(" \t?!.,")
    return location or fallback.strip()
