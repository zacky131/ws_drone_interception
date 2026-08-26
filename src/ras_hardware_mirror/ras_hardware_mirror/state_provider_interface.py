"""Transport boundary for Gazebo now and a future, intentionally unimplemented RTK driver."""

from __future__ import annotations

from abc import ABC, abstractmethod
from .state_types import NavigationState


class StateProvider(ABC):
    @abstractmethod
    def latest(self) -> NavigationState | None:
        """Return a common map/ENU state, or None until the provider is ready."""


class RTKStateProvider(StateProvider):
    """Future interface stub; no physical RTK transport is implemented here."""

    def latest(self) -> NavigationState | None:
        raise NotImplementedError("RTKStateProvider requires later site/receiver integration")
