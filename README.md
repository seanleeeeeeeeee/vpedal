A computer vision device that creates a virtual MIDI pedalboard below any piano.
<img width="1263" height="710" alt="Capture d’écran, le 2026-08-05 à 02 19 31" src="https://github.com/user-attachments/assets/30b79a33-7881-4c52-8de2-138e90e1a932" />
- Straightforward setup: tap corner points, keypress detection parameters adjusted semi-automatically.
- Runs with OpenCV on Raspberry Pi 5 with IMX219 and similar cameras (see picamera2 docs)

## Notes
- An external monitor, keyboard, and mouse are required for configuration

## Issues
- Background subtraction will not work if your footwear is a darker color than shadows. Use white socks for the best sensitivity.
