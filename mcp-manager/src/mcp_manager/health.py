#!/usr/bin/env python3
"""Health monitoring system for MCP Gateway."""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

from loguru import logger
from pydantic import BaseModel


@dataclass
class HealthMetric:
    """A single health metric."""

    name: str
    status: str  # 'healthy', 'degraded', 'unhealthy'
    value: Any
    timestamp: float
    details: Dict[str, Any] = field(default_factory=dict)


class ConnectionHealth(BaseModel):
    """Health status of a connection."""

    name: str
    transport: str
    is_connected: bool
    last_successful_check: Optional[float] = None
    last_failed_check: Optional[float] = None
    failure_count: int = 0
    consecutive_failures: int = 0
    response_time_ms: Optional[float] = None


class HealthMonitor:
    """Monitor health of MCP connections and gateway."""

    def __init__(self, check_interval: int = 60):
        """Initialize health monitor.

        Args:
            check_interval: Interval between health checks in seconds
        """
        self.check_interval = check_interval
        self.connections: Dict[str, ConnectionHealth] = {}
        self.metrics: Dict[str, HealthMetric] = {}
        self.is_monitoring = False
        self._monitor_task: Optional[asyncio.Task] = None
        logger.info(f"Health monitor initialized with {check_interval}s check interval")

    def register_connection(
        self, name: str, transport: str
    ) -> None:
        """Register a connection for monitoring.

        Args:
            name: Connection name
            transport: Transport type
        """
        self.connections[name] = ConnectionHealth(
            name=name,
            transport=transport,
            is_connected=False,
        )
        logger.info(f"Registered connection for monitoring: {name}")

    async def check_connection(self, name: str, check_func) -> bool:
        """Check health of a connection.

        Args:
            name: Connection name
            check_func: Async function to check connection

        Returns:
            True if connection is healthy, False otherwise
        """
        if name not in self.connections:
            return False

        health = self.connections[name]
        start_time = time.time()

        try:
            result = await asyncio.wait_for(check_func(), timeout=5.0)
            response_time = (time.time() - start_time) * 1000  # Convert to ms

            health.is_connected = result
            health.response_time_ms = response_time
            health.last_successful_check = time.time()
            health.consecutive_failures = 0

            if result:
                logger.debug(f"Connection {name} health check passed ({response_time:.1f}ms)")
            else:
                health.failure_count += 1
                health.consecutive_failures += 1
                health.last_failed_check = time.time()
                logger.warning(f"Connection {name} health check failed")

            return result
        except asyncio.TimeoutError:
            logger.warning(f"Connection {name} health check timed out")
            health.is_connected = False
            health.failure_count += 1
            health.consecutive_failures += 1
            health.last_failed_check = time.time()
            return False
        except Exception as e:
            logger.error(f"Connection {name} health check error: {e}")
            health.is_connected = False
            health.failure_count += 1
            health.consecutive_failures += 1
            health.last_failed_check = time.time()
            return False

    def get_connection_health(self, name: str) -> Optional[ConnectionHealth]:
        """Get health status of a connection.

        Args:
            name: Connection name

        Returns:
            Connection health or None if not found
        """
        return self.connections.get(name)

    def get_all_health(self) -> Dict[str, ConnectionHealth]:
        """Get health status of all connections.

        Returns:
            Dictionary of all connection health statuses
        """
        return self.connections.copy()

    def record_metric(
        self, name: str, status: str, value: Any, details: Optional[Dict] = None
    ) -> None:
        """Record a custom health metric.

        Args:
            name: Metric name
            status: Health status
            value: Metric value
            details: Additional details
        """
        self.metrics[name] = HealthMetric(
            name=name,
            status=status,
            value=value,
            timestamp=time.time(),
            details=details or {},
        )

    def get_metric(self, name: str) -> Optional[HealthMetric]:
        """Get a health metric.

        Args:
            name: Metric name

        Returns:
            Health metric or None if not found
        """
        return self.metrics.get(name)

    def get_all_metrics(self) -> Dict[str, HealthMetric]:
        """Get all health metrics.

        Returns:
            Dictionary of all metrics
        """
        return self.metrics.copy()

    def get_overall_status(self) -> str:
        """Get overall health status.

        Returns:
            'healthy', 'degraded', or 'unhealthy'
        """
        if not self.connections:
            return "unknown"

        healthy_count = sum(1 for h in self.connections.values() if h.is_connected)
        total_count = len(self.connections)

        if healthy_count == total_count:
            return "healthy"
        elif healthy_count > 0:
            return "degraded"
        else:
            return "unhealthy"

    def get_status_summary(self) -> Dict[str, Any]:
        """Get comprehensive health status summary.

        Returns:
            Dictionary with health status information
        """
        health_stats = self.get_all_health()
        healthy = sum(1 for h in health_stats.values() if h.is_connected)
        total = len(health_stats)

        return {
            "overall_status": self.get_overall_status(),
            "timestamp": datetime.now().isoformat(),
            "connections": {
                "total": total,
                "healthy": healthy,
                "unhealthy": total - healthy,
                "percentage": (healthy / total * 100) if total > 0 else 0,
            },
            "connection_details": {
                name: {
                    "is_connected": health.is_connected,
                    "transport": health.transport,
                    "response_time_ms": health.response_time_ms,
                    "failure_count": health.failure_count,
                    "consecutive_failures": health.consecutive_failures,
                    "last_successful_check": (
                        datetime.fromtimestamp(health.last_successful_check).isoformat()
                        if health.last_successful_check
                        else None
                    ),
                    "last_failed_check": (
                        datetime.fromtimestamp(health.last_failed_check).isoformat()
                        if health.last_failed_check
                        else None
                    ),
                }
                for name, health in health_stats.items()
            },
            "metrics": {
                name: {
                    "status": metric.status,
                    "value": str(metric.value),
                    "timestamp": datetime.fromtimestamp(metric.timestamp).isoformat(),
                }
                for name, metric in self.metrics.items()
            },
        }

    async def start_monitoring(self, check_functions: Dict[str, Any]) -> None:
        """Start continuous health monitoring.

        Args:
            check_functions: Dictionary mapping connection names to check functions
        """
        if self.is_monitoring:
            logger.warning("Monitoring already running")
            return

        self.is_monitoring = True
        logger.info("Starting health monitoring")

        async def monitor_loop():
            while self.is_monitoring:
                try:
                    tasks = []
                    for name, check_func in check_functions.items():
                        tasks.append(self.check_connection(name, check_func))

                    if tasks:
                        await asyncio.gather(*tasks, return_exceptions=True)

                    await asyncio.sleep(self.check_interval)
                except Exception as e:
                    logger.error(f"Monitoring error: {e}")
                    await asyncio.sleep(self.check_interval)

        self._monitor_task = asyncio.create_task(monitor_loop())

    async def stop_monitoring(self) -> None:
        """Stop continuous health monitoring."""
        self.is_monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        logger.info("Health monitoring stopped")

    def reset(self) -> None:
        """Reset all health monitoring data."""
        self.connections.clear()
        self.metrics.clear()
        logger.info("Health monitoring data reset")
