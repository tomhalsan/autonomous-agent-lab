import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    redis_url: str = "redis://localhost:6379/0"
    app_env: str = "development"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
            app_env=os.getenv("APP_ENV", "development"),
        )
