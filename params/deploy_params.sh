# deploy all csv files in thfolder to the /params folder in the serial device on /dev/ttyUSB0
for f in *.csv; do
    echo "Pushing $f to target device"
    ampy --port /dev/ttyUSB0 put $f /params/$f
done
