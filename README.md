
# Init Environment

#### Create and activate a virtual environment
```
uv venv
source .venv/bin/activate
```

#### Install dependencies from pyproject.toml
```
uv sync
```

# Download POI Data

```
gdown 1ZMA_tNP6EapE6UED8hyi5_6uoNzTSvt0 -O data/geo/safegraph-sf-poi.csv
gdown 1JnOaM22HSrWC8SWVKm2x7kJHjl0oHoVt -O data/geo/safegraph-cambridge-poi.csv
```