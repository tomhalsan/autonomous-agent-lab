import asyncio
import logging
import math
from collections.abc import Awaitable, Callable
from typing import Self

from agent_contracts.availability import AgentHeartbeat

logger = logging.getLogger(__name__)


def parse_heartbeat_interval(
    value: str | float | None, default: float = 10.0
) -> float:
    """Validate and parse a heartbeat interval in seconds.

    Must be a positive, finite number.
    """
    if value is None:
        return default
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return default
    try:
        val = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(
            f"Invalid heartbeat interval '{value}': must be a positive number"
        ) from exc
    if not math.isfinite(val) or val <= 0:
        raise ValueError(
            f"Invalid heartbeat interval {val}: must be a positive finite number"
        )
    return val


class HeartbeatRunner:
    """Periodic background heartbeat emitter that maintains runtime agent liveness."""

    def __init__(
        self,
        agent_id: str,
        publish_heartbeat: Callable[[AgentHeartbeat], Awaitable[None]],
        get_state: Callable[[], str] | None = None,
        interval_seconds: float = 10.0,
    ) -> None:
        if not agent_id or not agent_id.strip():
            raise ValueError("agent_id must not be empty")
        self.agent_id = agent_id.strip()
        self.interval_seconds = parse_heartbeat_interval(interval_seconds)
        self.publish_heartbeat = publish_heartbeat
        self.get_state = get_state or (lambda: "idle")
        self._task: asyncio.Task[None] | None = None
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running and self._task is not None and not self._task.done()

    def start(self) -> None:
        if self.is_running:
            return
        self._running = True
        self._task = asyncio.create_task(
            self._run_loop(), name=f"heartbeat-{self.agent_id}"
        )

    async def stop(self) -> None:
        self._running = False
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            finally:
                self._task = None

    async def publish_once(self) -> AgentHeartbeat:
        """Publish a single heartbeat immediately using current execution state."""
        raw_state = self.get_state()
        state = raw_state if isinstance(raw_state, str) else "idle"
        heartbeat = AgentHeartbeat.create(
            agent_id=self.agent_id,
            execution_state=state,
        )
        await self.publish_heartbeat(heartbeat)
        return heartbeat

    async def _run_loop(self) -> None:
        try:
            while self._running:
                await asyncio.sleep(self.interval_seconds)
                if not self._running:
                    break
                try:
                    await self.publish_once()
                except asyncio.CancelledError:
                    raise
                except (OSError, RuntimeError, TimeoutError, ValueError) as exc:
                    logger.warning(
                        "Heartbeat emission failed for agent %s: %s",
                        self.agent_id,
                        exc,
                    )
        except asyncio.CancelledError:
            pass
        finally:
            self._running = False

    async def __aenter__(self) -> Self:
        self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: object,
    ) -> None:
        await self.stop()
