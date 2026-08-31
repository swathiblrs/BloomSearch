import unittest

from bloom_search.weather import WeatherService, is_weather_query, weather_location


class FakeWeatherAPI:
    def __call__(self, url):
        if "geocoding-api" in url:
            return {
                "results": [
                    {
                        "name": "Knoxville",
                        "admin1": "Tennessee",
                        "country": "United States",
                        "latitude": 35.9606,
                        "longitude": -83.9207,
                    }
                ]
            }
        return {
            "current": {
                "time": "2026-08-31T12:00",
                "temperature_2m": 82.4,
                "apparent_temperature": 84.1,
                "relative_humidity_2m": 61,
                "precipitation": 0.0,
                "weather_code": 2,
                "wind_speed_10m": 5.2,
            }
        }


class WeatherTests(unittest.TestCase):
    def test_detects_weather_queries(self):
        self.assertTrue(is_weather_query("What is the weather today?"))
        self.assertTrue(is_weather_query("Current temperature"))
        self.assertFalse(is_weather_query("What is a Bloom filter?"))

    def test_returns_current_conditions(self):
        result = WeatherService(FakeWeatherAPI()).current("Knoxville, TN")
        self.assertEqual(result.condition, "Partly cloudy")
        self.assertIn("82.4°F", result.summary)
        self.assertIn("Knoxville", result.location)

    def test_extracts_location_from_query(self):
        self.assertEqual(
            weather_location("What is the weather in India?", "Knoxville, Tennessee"),
            "India",
        )
        self.assertEqual(
            weather_location("weather today", "Knoxville, Tennessee"),
            "Knoxville, Tennessee",
        )


if __name__ == "__main__":
    unittest.main()
