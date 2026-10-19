FROM python:3.12-slim
WORKDIR /app
RUN pip install --no-cache-dir numpy pandas requests py-clob-client==0.34.6
COPY polyweather /app/polyweather
# State (pilot.db, STOP kill file, response cache) lives on the /data volume.
WORKDIR /data
ENV PYTHONPATH=/app PYTHONUNBUFFERED=1 POLYWEATHER_CACHE=/data/cache \
    PILOT_DB=/data/pilot.db PILOT_KILL_FILE=/data/STOP
CMD ["python", "-m", "polyweather", "pilot"]
