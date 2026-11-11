from dataclasses import dataclass, field


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
