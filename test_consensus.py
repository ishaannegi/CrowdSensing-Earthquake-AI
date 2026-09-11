import sys
import unittest
from datetime import datetime, timezone
import os

# Add paths to import detector logic
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "final ai implementation"))
import ai_enhanced_detector as detector

class TestSpatialConsensusLogic(unittest.TestCase):

    def setUp(self):
        # Reset event buffer before each test
        detector.event_buffer = []

    def test_haversine_distance(self):
        # Delhi CP (28.6139, 77.2090) to CP East (28.6200, 77.2200) ~ 1.28 km
        d = detector.haversine(28.6139, 77.2090, 28.6200, 77.2200)
        self.assertAlmostEqual(d, 1.28, delta=0.5)

        # Gurugram (28.4595, 77.0266) to Greater Noida (28.4744, 77.5040) ~ 46.7 km
        d_far = detector.haversine(28.4595, 77.0266, 28.4744, 77.5040)
        self.assertGreater(d_far, 40.0)

    def test_lead_time_calculation(self):
        # Distance = 0km -> Lead time = 0.0s
        self.assertEqual(detector.calculate_lead_time(0.0), 0.0)

        # Distance = 24.6km (Ghaziabad) -> Lead time = 24.6 * (1/3.5 - 1/6.0) ~ 2.9s
        lead_time = detector.calculate_lead_time(24.6)
        self.assertAlmostEqual(lead_time, 2.9, delta=0.2)

    def test_single_node_false_alarm(self):
        now_ts = datetime.now(timezone.utc).timestamp()
        alert, lead_time, epi = detector.compute_consensus("Node-1", 28.6139, 77.2090, "real_quake", now_ts)
        self.assertEqual(alert, "LOCAL_VIBRATION")
        self.assertIsNone(epi)

    def test_distant_nodes_spatial_filtering(self):
        now_ts = datetime.now(timezone.utc).timestamp()
        # Node-3 Gurugram
        alert1, _, _ = detector.compute_consensus("Node-3", 28.4595, 77.0266, "real_quake", now_ts)
        # Node-7 Greater Noida (46.7 km away)
        alert2, _, _ = detector.compute_consensus("Node-7", 28.4744, 77.5040, "real_quake", now_ts + 0.5)
        
        self.assertEqual(alert1, "LOCAL_VIBRATION")
        self.assertEqual(alert2, "LOCAL_VIBRATION")

    def test_confirmed_cluster_consensus(self):
        now_ts = datetime.now(timezone.utc).timestamp()
        # Node-1 (Delhi CP)
        alert1, _, _ = detector.compute_consensus("Node-1", 28.6139, 77.2090, "real_quake", now_ts)
        self.assertEqual(alert1, "LOCAL_VIBRATION")

        # Node-9 (CP East) - <2km away
        alert2, _, _ = detector.compute_consensus("Node-9", 28.6200, 77.2200, "real_quake", now_ts + 0.5)
        self.assertEqual(alert2, "LOCAL_VIBRATION")

        # Node-10 (Karol Bagh) - <4km away -> 3rd node triggers CONFIRMED_EARTHQUAKE_ALERT!
        alert3, lead_time, epi = detector.compute_consensus("Node-10", 28.6500, 77.1900, "real_quake", now_ts + 1.0)
        self.assertEqual(alert3, "CONFIRMED_EARTHQUAKE_ALERT")
        self.assertIsNotNone(epi)
        self.assertIn("epicenter", epi)
        self.assertEqual(len(epi["cluster_nodes"]), 3)

    def test_retroactive_alert_upgrade(self):
        now_ts = datetime.now(timezone.utc).timestamp()
        
        # Node 1 sends anomaly -> LOCAL_VIBRATION
        alert1, _, _ = detector.compute_consensus("Node-1 (Delhi CP)", 28.6139, 77.2090, "real_quake", now_ts)
        self.assertEqual(alert1, "LOCAL_VIBRATION")
        self.assertEqual(detector.event_buffer[0]["alert_level"], "LOCAL_VIBRATION")

        # Node 9 sends anomaly -> LOCAL_VIBRATION
        alert2, _, _ = detector.compute_consensus("Node-9 (CP East)", 28.6200, 77.2200, "real_quake", now_ts + 0.5)
        self.assertEqual(alert2, "LOCAL_VIBRATION")
        self.assertEqual(detector.event_buffer[1]["alert_level"], "LOCAL_VIBRATION")

        # Node 10 sends anomaly -> 3rd node triggers CONFIRMED_EARTHQUAKE_ALERT!
        alert3, _, _ = detector.compute_consensus("Node-10 (Karol Bagh)", 28.6500, 77.1900, "real_quake", now_ts + 1.0)
        self.assertEqual(alert3, "CONFIRMED_EARTHQUAKE_ALERT")

        # Verify retroactive upgrade in event_buffer memory for earlier Node-1 and Node-9!
        self.assertEqual(detector.event_buffer[0]["alert_level"], "CONFIRMED_EARTHQUAKE_ALERT")
        self.assertEqual(detector.event_buffer[1]["alert_level"], "CONFIRMED_EARTHQUAKE_ALERT")
        self.assertGreater(detector.event_buffer[0]["lead_time_sec"], 0.0)

if __name__ == "__main__":
    unittest.main()
