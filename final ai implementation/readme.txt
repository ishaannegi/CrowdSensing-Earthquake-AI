========================================================================
       EARTHQUAKE DETECTION CROWDSENSING NETWORK & LIVE DASHBOARD
========================================================================

📁 Project Structure
---------------------
final ai implementation/
│
├── run_servers.py                  # 🌐 Persistent servers launcher (Ports 5000 & 9000)
├── run_pipeline.py                 # ⚡ Automated end-to-end pipeline test runner
├── ai_trainer.py                   # Feature extraction & Random Forest ML training (STEAD data)
├── ai_enhanced_detector.py         # Multi-client TCP Server + 15km Consensus + S-Wave Lead Time Calculator
├── web_server.py                   # Multithreaded SSE HTTP Web Dashboard Server (Port 5000)
├── dashboard.html                  # OLED Dark Mode Leaflet Map & Chart.js Dashboard UI
├── simulate_crowd.py               # Modular Multi-Node Simulator Engine
├── run_scenario1_false_alarm.py    # Standalone Scenario 1 Runner (False Alarm)
├── run_scenario2_confirmed_alert.py # Standalone Scenario 2 Runner (Confirmed Alert)
├── run_scenario3_spatial_filtering.py# Standalone Scenario 3 Runner (Spatial Filter >40km)
├── run_scenario4_normal_reset.py   # Standalone Scenario 4 Runner (System Reset)
├── earthquake_client.py            # Individual TCP Client for AI inference & telemetry transmission
├── sta_lta_detector.py             # STA/LTA waveform visualization graph plotter
├── generate_data.py                # Optional fallback synthetic data generator
│
├── earthquake_ai_model.pkl         # Trained Random Forest ML model (97.33% accuracy)
├── training_features.csv           # Extracted 11-feature dataset
├── received_log.csv                # Master live server log (includes lead_time_sec & alert_level)
├── received_log_scenario1.csv      # Scenario 1 log file (False Alarm)
├── received_log_scenario2.csv      # Scenario 2 log file (Confirmed Alert & Lead Times)
├── received_log_scenario3.csv      # Scenario 3 log file (Spatial Filtering >40km)
└── received_log_scenario4.csv      # Scenario 4 log file (System Reset)

========================================================================
🚀 QUICK START
========================================================================

1. Launch Live Servers (Terminal 1):
   python run_servers.py

2. Open Live Web Dashboard in Browser:
   http://127.0.0.1:5000

3. Run Individual Test Scenarios (Terminal 2):
   python run_scenario1_false_alarm.py       # Test Single Node False Alarm
   python run_scenario2_confirmed_alert.py    # Test Confirmed Quake Consensus & Lead Times
   python run_scenario3_spatial_filtering.py # Test Spatial Filtering (>40km)
   python run_scenario4_normal_reset.py      # Test System Normal Reset

   Or run all scenarios sequentially:
   python simulate_crowd.py --scenario all

========================================================================
📊 Scenario Output Logs
========================================================================
All server received telemetry packets are saved in received_log.csv and
dedicated scenario log files (received_log_scenario1.csv ... scenario4.csv):
- timestamp, node_id, lat, lon
- file
- prediction (real_quake / fake_vibration)
- confidence score
- peak magnitude & STA/LTA ratio
- lead_time_sec (individual S-wave warning lead time in seconds)
- alert_level (NORMAL / LOCAL_VIBRATION / CONFIRMED_EARTHQUAKE_ALERT)
