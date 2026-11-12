from dataclasses import dataclass, field
from urllib.parse import urlparse


@dataclass(frozen=True)
class McpServerConfig:
    """Configuration for connecting to an external MCP server process over stdio."""

    command: str
    args: list[str] = field(default_factory=list)
    env: dict[str, str] | None = None
    cwd: str | None = None
    timeout: float = 30.0

    def __post_init__(self) -> None:
        if not isinstance(self.command, str) or not self.command.strip():
            raise ValueError("command must be a non-empty string")
        if not isinstance(self.args, list) or not all(isinstance(a, str) for a in self.args):
            raise TypeError("args must be a list of strings")
        if self.env is not None and (
            not isinstance(self.env, dict)
            or not all(isinstance(k, str) and isinstance(v, str) for k, v in self.env.items())
        ):
            raise TypeError("env must be a dict of strings if provided")
        if self.cwd is not None and (not isinstance(self.cwd, str) or not self.cwd.strip()):
            raise ValueError("cwd must be a non-empty string if provided")
        if (
            type(self.timeout) is bool
            or not isinstance(self.timeout, (int, float))
            or self.timeout <= 0
        ):
            raise ValueError("timeout must be a positive number")
        object.__setattr__(self, "command", self.command.strip())
        object.__setattr__(self, "args", list(self.args))
        if self.env is not None:
            object.__setattr__(self, "env", dict(self.env))
        if self.cwd is not None:
            object.__setattr__(self, "cwd", self.cwd.strip())
        object.__setattr__(self, "timeout", float(self.timeout))


@dataclass(frozen=True)
class McpHttpClientConfig:
    """Configuration for connecting to an external MCP server over Streamable HTTP."""

    url: str
    headers: dict[str, str] | None = field(default=None, repr=False)
    timeout: float = 30.0
    terminate_on_close: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.url, str) or not self.url.strip():
            raise ValueError("url must be a non-empty string")
        parsed = urlparse(self.url.strip())
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise ValueError(f"url must be a valid http or https URL, got {self.url!r}")
        if self.headers is not None and (
            not isinstance(self.headers, dict)
            or not all(isinstance(k, str) and isinstance(v, str) for k, v in self.headers.items())
        ):
            raise TypeError("headers must be a dict of strings if provided")
        if (
            type(self.timeout) is bool
            or not isinstance(self.timeout, (int, float))
            or self.timeout <= 0
        ):
            raise ValueError("timeout must be a positive number")
        if not isinstance(self.terminate_on_close, bool):
            raise TypeError("terminate_on_close must be a boolean")

        object.__setattr__(self, "url", self.url.strip())
        if self.headers is not None:
            object.__setattr__(self, "headers", dict(self.headers))
        object.__setattr__(self, "timeout", float(self.timeout))
        object.__setattr__(self, "terminate_on_close", bool(self.terminate_on_close))
