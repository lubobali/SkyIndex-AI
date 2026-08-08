# Databricks notebook source
# MAGIC %md
# MAGIC # Scheduled refresh — harvest NWS narrative, then embed what is new
# MAGIC
# MAGIC This is the notebook the Databricks Workflow runs. It is a thin wrapper:
# MAGIC all the logic lives in `scripts/refresh_weather_index.py`, which also runs
# MAGIC as a plain CLI script. One implementation, two entry points.
# MAGIC
# MAGIC **Why a notebook rather than a Python-script task.** Installing
# MAGIC `psycopg2-binary` into a running kernel that has already loaded a
# MAGIC different build of the same native extension aborts the process with
# MAGIC SIGABRT the moment `import psycopg2` runs — not an exception, a crash.
# MAGIC The fix is to restart the Python interpreter after installing, and only a
# MAGIC notebook can do that mid-run. A `spark_python_task` has no way to.

# COMMAND ----------

# DBTITLE 1,Install dependencies
# MAGIC %pip uninstall -y psycopg2 psycopg2-binary
# MAGIC %pip install -q psycopg2-binary sentence-transformers requests databricks-sdk

# COMMAND ----------

# DBTITLE 1,Restart the interpreter so the freshly built extensions load cleanly
# This is the line whose absence crashed the job. Everything above installed
# into an interpreter that had already imported the old psycopg2; nothing below
# would work without a restart.
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Config
dbutils.widgets.text("user_agent", "(SkyIndex-AI, lubobali23@gmail.com)", "NWS User-Agent")
dbutils.widgets.text(
    "locations",
    "Chicago, IL;Austin, TX;Denver, CO;Miami, FL;Seattle, WA",
    "Locations (semicolon separated)",
)
dbutils.widgets.text("limit", "50", "Max documents per location per source")

USER_AGENT = dbutils.widgets.get("user_agent")
LOCATIONS = [part.strip() for part in dbutils.widgets.get("locations").split(";") if part.strip()]
LIMIT = int(dbutils.widgets.get("limit"))

# COMMAND ----------

# DBTITLE 1,Locate the project and run the refresh
import os
import sys

# The notebook lives in <project>/notebooks/, so the project root is its parent.
# Resolved from the notebook's own workspace path rather than __file__, which a
# Databricks notebook never defines.
_NOTEBOOK_PATH = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
)
PROJECT_ROOT = "/Workspace" + os.path.dirname(os.path.dirname(_NOTEBOOK_PATH))

for path in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "scripts")):
    if path not in sys.path:
        sys.path.insert(0, path)

# api.weather.gov rejects requests without a descriptive User-Agent, and the
# client refuses to start without one. weather_client reads the env var into a
# module constant at import time, so both are set.
os.environ["NWS_USER_AGENT"] = USER_AGENT
os.environ.setdefault("HF_HOME", "/tmp/.cache/huggingface")

import weather_client  # noqa: E402

weather_client.DEFAULT_USER_AGENT = USER_AGENT

from refresh_weather_index import refresh  # noqa: E402

print(f"project root : {PROJECT_ROOT}")
print(f"locations    : {LOCATIONS}")

summary = refresh(locations=LOCATIONS, limit=LIMIT, source_types=["alert", "forecast"])

# COMMAND ----------

# DBTITLE 1,Report
for key, value in summary.items():
    print(f"  {key:<20} {value}")

# Surfaces in the Workflow run output, so a run's result is visible without
# opening the driver logs.
dbutils.notebook.exit(str(summary))
