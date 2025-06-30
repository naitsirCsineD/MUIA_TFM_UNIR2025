import serial
import csv
import time
import sys
from serial.tools import list_ports

PORT = 'COM3'
BAUD = 115200

# 1) Lista y muestra puertos
print("Puertos detectados:")
for p in list_ports.comports():
    print(f"  {p.device} – {p.description}")
print(f"\nIntentando abrir {PORT} a {BAUD} baudios…")

# 2) Intenta abrir
try:
    ser = serial.Serial(PORT, BAUD, timeout=1)


    filename = f"datos_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    print(f"Grabando datos en: {filename}")
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        print(['Timestamp','MagX','MagY','MagZ','Lat','Lon'])
        writer.writerow(['Timestamp','MagX','MagY','MagZ','Lat','Lon'])
        try:
            while True:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                print(line)
                if line.startswith('Mag X:'):
                    parts = line.replace('Mag X:','')\
                                .replace('Y:','')\
                                .replace('Z:','').split()
                    magx, magy, magz = parts[:3]
                    gps_line = ser.readline().decode('utf-8', errors='ignore').strip()
                    if gps_line.startswith('Lat:'):
                        parts = gps_line.replace('Lat:','')\
                                        .replace('Lon:','').split()
                        lat, lon = parts[:2]
                        ts = time.strftime('%Y-%m-%d %H:%M:%S')
                        writer.writerow([ts, magx, magy, magz, lat, lon])
                        print([ts, magx, magy, magz, lat, lon])
                        csvfile.flush()
        except KeyboardInterrupt:
            pass
        finally:
            ser.close()
            print(f"\nCaptura finalizada. Archivo: {filename}")

except serial.SerialException as e:
    print(f"ERROR: no pude abrir {PORT}: {e}")
    sys.exit(1)

