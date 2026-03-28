"""Prometheus metrics and monitoring."""

import logging
from prometheus_client import Counter, Histogram, Gauge

logger = logging.getLogger(__name__)

# ============================================================================
# API Request Metrics
# ============================================================================

# Requests per endpoint
http_requests_total = Counter(
    "http_requests_total",
    "Total HTTP requests",
    ["method", "endpoint", "status"],
)

# Request latency
http_request_duration_seconds = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
    buckets=(0.01, 0.05, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0),
)

# ============================================================================
# Rate Limiting Metrics
# ============================================================================

rate_limit_exceeded_total = Counter(
    "rate_limit_exceeded_total",
    "Total rate limit rejections",
    ["ip_address"],
)

# ============================================================================
# Authentication Metrics
# ============================================================================

auth_failures_total = Counter(
    "auth_failures_total",
    "Total authentication failures",
    ["reason"],  # "invalid_key", "expired_key", "missing_key"
)

api_keys_active = Gauge(
    "api_keys_active",
    "Number of active API keys",
)

# ============================================================================
# Research Module Metrics
# ============================================================================

research_queries_total = Counter(
    "research_queries_total",
    "Total research queries",
    ["status"],  # "success", "failure"
)

research_query_duration_seconds = Histogram(
    "research_query_duration_seconds",
    "Research query duration in seconds",
    ["status"],
    buckets=(0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0),
)

research_tools_used = Counter(
    "research_tools_used",
    "Research tool executions",
    ["tool_name", "status"],  # tavily, serpapi, duckduckgo, gpt-researcher
)

# ============================================================================
# Database Metrics
# ============================================================================

database_queries_total = Counter(
    "database_queries_total",
    "Total database queries",
    ["operation"],  # "select", "insert", "update", "delete"
)

database_connection_errors = Counter(
    "database_connection_errors",
    "Database connection errors",
)

# ============================================================================
# Quota & Cost Metrics
# ============================================================================

quota_exceeded_total = Counter(
    "quota_exceeded_total",
    "Total quota limit exceeded events",
    ["quota_type"],  # "daily", "monthly"
)

research_cost_units = Counter(
    "research_cost_units",
    "Total cost units consumed by research",
)


def record_request(method: str, endpoint: str, status: int, duration: float):
    """Record HTTP request metrics."""
    http_requests_total.labels(method=method, endpoint=endpoint, status=status).inc()
    http_request_duration_seconds.labels(method=method, endpoint=endpoint).observe(
        duration
    )


def record_research_query(status: str, duration: float):
    """Record research query metrics."""
    research_queries_total.labels(status=status).inc()
    research_query_duration_seconds.labels(status=status).observe(duration)


def record_tool_usage(tool_name: str, succeeded: bool):
    """Record tool execution."""
    status = "success" if succeeded else "failure"
    research_tools_used.labels(tool_name=tool_name, status=status).inc()


def record_rate_limit(ip_address: str):
    """Record rate limit rejection."""
    rate_limit_exceeded_total.labels(ip_address=ip_address).inc()


def record_auth_failure(reason: str):
    """Record authentication failure."""
    auth_failures_total.labels(reason=reason).inc()


def record_quota_exceeded(quota_type: str):
    """Record quota limit exceeded."""
    quota_exceeded_total.labels(quota_type=quota_type).inc()
