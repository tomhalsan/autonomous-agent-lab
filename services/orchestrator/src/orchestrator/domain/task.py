from dataclasses import dataclass


@dataclass(frozen=True)
class Task:
    prompt: str

    def __post_init__(self) -> None:
        if not self.prompt.strip():
            raise ValueError("Prompt must not be empty")
        if len(self.prompt) > 2000:
            raise ValueError("Prompt must not exceed 2000 characters")
