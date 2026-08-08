# Databricks notebook source
# MAGIC %md
# MAGIC # Scheduled refresh — harvest NWS narrative, then embed what is new
# MAGIC
# MAGIC This is the notebook the Databricks Workflow runs. It is a thin wrapper:
# MAGIC all the logic lives in `scripts/refresh_weather_index.py`, which also runs
# MAGIC as a plain CLI script. One implementation, two entry points.
# MAGIC
# MAGIC **Do not install `psycopg2-binary` here.** Databricks Runtime already
# MAGIC ships `psycopg2` 2.9.11, in `/databricks/python3/...`, which pip refuses
# MAGIC to uninstall because it sits outside the ephemeral environment:
# MAGIC
# MAGIC ```
# MAGIC Not uninstalling psycopg2 at /databricks/python3/lib/python3.12/site-packages,
# MAGIC outside environment
# MAGIC ```
# MAGIC
# MAGIC Installing `psycopg2-binary` on top therefore leaves two competing builds
# MAGIC of the same native extension in one interpreter, and `import psycopg2`
# MAGIC aborts the process with SIGABRT - a crash, not an exception, so no
# MAGIC try/except helps and restarting the kernel does not either, because both
# MAGIC copies are still present. The runtime's own psycopg2 works fine; the only
# MAGIC winning move is not to install a second one.
# MAGIC
# MAGIC **Why a notebook rather than a Python-script task.** `sentence-transformers`
# MAGIC still has to be installed at runtime, and any pip install into a live
# MAGIC kernel needs an interpreter restart afterwards. Only a notebook can do
# MAGIC that mid-run; a `spark_python_task` has no way to.

# COMMAND ----------

# DBTITLE 1,Install dependencies
# psycopg2 is deliberately absent - the runtime provides it. requests and
# databricks-sdk ship with the runtime too.
# MAGIC %pip install -q sentence-transformers

# COMMAND ----------

# DBTITLE 1,Restart the interpreter so the freshly installed packages load cleanly
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
