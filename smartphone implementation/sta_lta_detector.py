import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import smtplib
import ssl
from email.message import EmailMessage
import sys

SAMPLING_RATE = 100
T_short = 0.1
T_long = 2.0
TRIGGER_ON = 10.0
TRIGGER_OFF = 2.0

SEND_EMAIL_ALERTS = True
SENDER_EMAIL = "rishunegi2005@gmail.com"
RECEIVER_EMAIL = "ishaan.negi2023@vitstudent.ac.in"

EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD")

def send_alert_email(subject, body_content):
    if not EMAIL_PASSWORD:
        print("\n--- EMAIL ERROR ---", file=sys.stderr)
        print("CRITICAL: 'EMAIL_PASSWORD' environment variable not set.", file=sys.stderr)
        print("Email alert NOT sent.", file=sys.stderr)
        return

    msg = EmailMessage()
    msg.set_content(body_content)
    msg['Subject'] = subject
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECEIVER_EMAIL

    print(f"\nSending email alert to {RECEIVER_EMAIL}...")

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
            server.login(SENDER_EMAIL, EMAIL_PASSWORD)
            server.send_message(msg)
        print("Email alert sent successfully!")
    except Exception as e:
        print(f"\n--- EMAIL ERROR ---", file=sys.stderr)
        print(f"Failed to send email: {e}", file=sys.stderr)

def run_sta_lta_detector(signal, sampling_rate, t_short, t_long, trig_on, trig_off):
    nsta = int(t_short * sampling_rate)
    nlta = int(t_long * sampling_rate)
    
    if nsta >= nlta or nlta >= len(signal):
        return [], [], np.array([])
        
    squared_signal = signal ** 2
    csta = np.cumsum(squared_signal)
    sta = (csta[nsta:] - csta[:-nsta]) / nsta
    lta = (csta[nlta:] - csta[:-nlta]) / nlta
    lta = np.where(lta == 0, 1e-10, lta)
    ratio = sta[nlta - nsta:] / lta
    
    on, off = [], []
    triggered = False
    
    for i, val in enumerate(ratio):
        if not triggered and val > trig_on:
            on.append(i + nlta)
            triggered = True
        elif triggered and val < trig_off:
            off.append(i + nlta)
            triggered = False
            
    if triggered and len(on) > len(off):
        off.append(len(signal) - 1)
        
    return on, off, ratio

def estimate_magnitude(peak_amplitude):
    if peak_amplitude >= 10:
        return "7.1+ (Major)"
    elif peak_amplitude >= 7:
        return "6.1 - 7.0 (Strong)"
    elif peak_amplitude >= 4:
        return "5.0 - 6.0 (Moderate)"
    else:
        return "Below 5.0 (Minor)"

def classify_event(duration, peak_amplitude):
    if duration > 0.5 and peak_amplitude > 4:
        return "REAL EARTHQUAKE DETECTED"
    elif duration > 0 or peak_amplitude > 0:
        return "LOW-LEVEL VIBRATION"
    else:
        return "NO EVENT DETECTED"

def process_file(filename, color):
    try:
        data = pd.read_csv(filename) 
    except FileNotFoundError:
        print(f"\n--- ERROR: File not found: {filename}", file=sys.stderr)
        print("Did you run 'python generate_data.py' first?", file=sys.stderr)
        return
    except Exception as e:
        print(f"\n--- ERROR reading {filename}: {e}", file=sys.stderr)
        return

    if 'magnitude' not in data.columns:
        print(f"--- ERROR in {filename}: Column 'magnitude' not found.", file=sys.stderr)
        return

    signal = data['magnitude'].values
    
    on, off, ratio = run_sta_lta_detector(signal, SAMPLING_RATE, T_short, T_long, TRIGGER_ON, TRIGGER_OFF)
    
    if on and off and on[0] < off[0]:
        start_index = on[0]
        end_index = off[0]
        
        if start_index < len(signal) and end_index < len(signal):
            start_sec = start_index / SAMPLING_RATE
            end_sec = end_index / SAMPLING_RATE
            duration_sec = end_sec - start_sec
            peak_amplitude = max(abs(signal[start_index:end_index]))
        else:
            start_sec, end_sec, duration_sec, peak_amplitude = 0, 0, 0, 0
    else:
        start_sec, end_sec, duration_sec, peak_amplitude = 0, 0, 0, 0

    classification = classify_event(duration_sec, peak_amplitude)
    estimated_magnitude = estimate_magnitude(peak_amplitude)
    
    print(f"\n--- {filename} ---")
    print(f"Duration: {duration_sec:.2f}s | Peak Amplitude: {peak_amplitude:.2f}")
    print(f"Result: {classification}")
    if "REAL EARTHQUAKE" in classification:
        print(f"Estimated Magnitude: {estimated_magnitude}")

    if SEND_EMAIL_ALERTS and "REAL EARTHQUAKE" in classification:
        
        email_subject = f"URGENT: SEISMIC ALERT - Est. Magnitude {estimated_magnitude}"
        email_body = f"""
        **URGENT SEISMIC ALERT**

        A significant seismic event has been detected by the monitoring system.

        **Event Classification:** {classification}
        **Data Source:** {filename}

        **EVENT METRICS:**
        --------------------------------
        - **ESTIMATED MAGNITUDE:** {estimated_magnitude}
        - **Peak Ground Motion Amplitude:** {peak_amplitude:.2f} (at sensor location)
        - **Detected Shaking Duration:** {duration_sec:.2f} seconds

        **SAFETY INSTRUCTIONS (IMMEDIATE ACTION REQUIRED):**
        --------------------------------
        1.  **EVACUATE:** If safe, move immediately to a designated assembly point.
        2.  **DUCK, COVER, AND HOLD:** If indoors, drop to the ground, take cover under a sturdy table or desk, and hold on.
        3.  **AVOID:** Stay clear of windows, bookshelves, tall furniture, and heavy objects.
        4.  **AFTERSHOCKS:** Be prepared for aftershocks, which can be as strong as the main event.

        This is an automated alert. Please follow your local emergency procedures.
        """
        send_alert_email(email_subject, email_body)
    
    plt.plot(signal, label=f"{filename} ({classification})", color=color, alpha=0.8)
    for o in on:
        plt.axvline(o, color='r', linestyle='--', linewidth=1)
    for f in off:
        plt.axvline(f, color='g', linestyle='--', linewidth=1)

if __name__ == "__main__":
    plt.figure(figsize=(14, 7))
    
    process_file("sensor_data_real.csv", "blue")
    process_file("sensor_data_fake.csv", "orange")

    plt.title("STA/LTA Earthquake Detection: Real vs Fake", fontsize=16)
    plt.xlabel("Sample Index", fontsize=12)
    plt.ylabel("Amplitude", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()