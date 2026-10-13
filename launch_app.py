!python /home/cdsw/app/pii_datagen.py --ensure && uvicorn server:app --app-dir /home/cdsw/app --host 127.0.0.1 --port $CDSW_READONLY_PORT
