import os
import unittest
from unittest.mock import patch

from app.config.settings import Settings
from app.domain.aggregation.services import (
    AttributeAggregator,
    CountAggregator,
)
from app.domain.detection.services import HumanDetectionService


class FakeDetector:
    def __init__(self, results=None):
        self.calls = 0
        self.results = results or []

    def detect(self, frame):
        self.calls += 1
        return list(self.results)

    def describe(self):
        return {"loaded": True}


class DetectionServiceTests(unittest.TestCase):
    def test_empty_results_do_not_force_every_frame(self):
        detector = FakeDetector()
        service = HumanDetectionService(detector, 0.5, 0, False, 2, 10.0)
        for _ in range(10):
            service.detect(None)
        self.assertEqual(detector.calls, 6)

    def test_stale_forces_rerun(self):
        results = [{"kind": "human", "box": [0, 0, 10, 10], "conf": 0.9}]
        detector = FakeDetector(results)
        service = HumanDetectionService(detector, 0.5, 0, False, 100, 0.0)
        for _ in range(3):
            service.detect(None)
        self.assertEqual(detector.calls, 3)


class SettingsTests(unittest.TestCase):
    def test_bool_parsing(self):
        with patch.dict(os.environ, {"FACE_ENABLE": "true",
                                     "HEADPOSE_ENABLE": "false"}):
            settings = Settings.from_env()
        self.assertTrue(settings.face_enable)
        self.assertFalse(settings.headpose_enable)

    def test_invalid_bool(self):
        with patch.dict(os.environ, {"FACE_ENABLE": "maybe"}):
            with self.assertRaises(ValueError):
                Settings.from_env()


class CountAggregatorTests(unittest.TestCase):
    def test_stats(self):
        aggregator = CountAggregator(window_sec=0.0)
        for value in (1, 2, 3):
            aggregator.add(value, 0)
        result = aggregator.current
        self.assertEqual(result["human"]["mode"], 3)
        self.assertEqual(result["samples"], 1)


class AttributeAggregatorTests(unittest.TestCase):
    def test_person_wise_mean_age_and_majority_gender(self):
        aggregator = AttributeAggregator(window_sec=1.0)
        aggregator.add([
            {"human_box": [0, 0, 100, 100], "box": [10, 10, 30, 30],
             "gender": "Male", "age": 30},
            {"human_box": [200, 0, 300, 100], "box": [210, 10, 230, 30],
             "gender": "Female", "age": 50},
        ])
        second = [
            {"human_box": [1, 0, 101, 100], "box": [11, 10, 31, 30],
             "gender": "Male", "age": 32},
            {"human_box": [201, 0, 301, 100], "box": [211, 10, 231, 30],
             "gender": "Female", "age": 48},
            {"human_box": [1, 0, 101, 100], "box": [12, 10, 32, 30],
             "gender": "Female", "age": 34},
        ]
        aggregator.add(second)
        result = aggregator.fill([dict(face) for face in second])
        self.assertEqual(result[0]["gender"], "Male")
        self.assertEqual(result[0]["age"], 32)
        self.assertEqual(result[1]["gender"], "Female")
        self.assertEqual(result[1]["age"], 49)

    def test_window_expiry(self):
        aggregator = AttributeAggregator(window_sec=1.0)
        face = {"human_box": [0, 0, 100, 100], "box": [10, 10, 30, 30],
                "gender": "Male", "age": 30}
        with patch("app.domain.aggregation.services.time.time",
                   return_value=0.0):
            aggregator.add([face])
        with patch("app.domain.aggregation.services.time.time",
                   return_value=2.0):
            result = aggregator.fill([dict(face)])
        self.assertEqual(result[0]["gender"], "Male")
        self.assertEqual(result[0]["age"], 30)
        self.assertEqual(aggregator.tracks, [])


if __name__ == "__main__":
    unittest.main()
