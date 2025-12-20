from core.config import AppConfig
from core.catalog import rebuild_catalog

if __name__ == "__main__":
    cfg = AppConfig()
    items = rebuild_catalog(cfg.garments_dir, cfg.catalog_json)
    print(f"Catalog rebuilt. Garments found: {len(items)}")
