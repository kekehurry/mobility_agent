
# Init Environment

#### Create and activate a virtual environment
```
uv venv
source .venv/bin/activate  # On Mac/Linux
```

#### Install dependencies from pyproject.toml
```
uv pip install -e .
```

# Download POI Data

```
gdown --id 1ZMA_tNP6EapE6UED8hyi5_6uoNzTSvt0 -O data/geo/safegraph-sf-poi.csv

gdown --id 1JnOaM22HSrWC8SWVKm2x7kJHjl0oHoVt -O data/geo/safegraph-cambridge-poi.csv
```