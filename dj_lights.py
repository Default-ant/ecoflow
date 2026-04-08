import time
import random
import sys

try:
    from gpiozero import LED
except ImportError:
    print("[Error] gpiozero not found. Running in dummy mode.")
    class LED:
        def __init__(self, p): self.pin = p
        def on(self): pass
        def off(self): pass

# ── Pin Mapping ──────────────────────────────────────────────────────────────
SIGNAL_PINS = [
    (17, 27, 22), # North
    (10, 9, 11),  # East
    (5, 6, 13),   # South
    (19, 26, 21)  # West
]

# Initialize all 12 LEDs
all_led_objs = []
for r, y, g in SIGNAL_PINS:
    all_led_objs.extend([LED(r), LED(y), LED(g)])

def all_off():
    for led in all_led_objs:
        led.off()

def all_on():
    for led in all_led_objs:
        led.on()

print("="*40)
print("   🎧 ECOFLOW AI - DJ LIGHT SHOW")
print("="*40)
print("Press Ctrl+C to stop the party.\n")

try:
    while True:
        # Pattern 1: The Fast Chase
        for _ in range(3):
            for led in all_led_objs:
                led.on()
                time.sleep(0.05)
                led.off()

        # Pattern 2: Lane Bounce
        for _ in range(4):
            for i in range(4): # North, East, South, West
                start = i * 3
                for j in range(start, start + 3):
                    all_led_objs[j].on()
                time.sleep(0.15)
                for j in range(start, start + 3):
                    all_led_objs[j].off()

        # Pattern 3: RANDOM DISCO
        for _ in range(60):
            idx = random.randint(0, 11)
            all_led_objs[idx].on()
            time.sleep(0.03)
            all_led_objs[idx].off()

        # Pattern 4: THE DROP (STROBE)
        for _ in range(12):
            all_on()
            time.sleep(0.04)
            all_off()
            time.sleep(0.04)

except KeyboardInterrupt:
    print("\n🛑 Party's over! Turning off lights...")
    all_off()
