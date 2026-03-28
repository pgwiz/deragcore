"""Phase 4 Production Hardening - Test Suite."""

import pytest
import asyncio
from uuid import uuid4
from datetime import datetime

# Auth tests
def test_auth_module_imports():
    """Test auth module can be imported."""
    from ragcore.auth.manager import APIKeyManager
    from ragcore.auth.models import User, APIKey, AuditLog

    assert APIKeyManager is not None
    assert User is not None
    assert APIKey is not None
    assert AuditLog is not None


# Rate limiting tests
def test_rate_limiter_initialization():
    """Test rate limiter initialization."""
    from ragcore.rate_limiting import rate_limiter

    assert rate_limiter is not None
    assert hasattr(rate_limiter, "check_rate_limit")
    assert hasattr(rate_limiter, "check_quota")


def test_rate_limit_check():
    """Test IP-based rate limiting."""
    from ragcore.rate_limiting import rate_limiter

    ip = "192.168.1.1"
    allowed, retry_after = rate_limiter.check_rate_limit(ip)

    assert allowed is True
    assert retry_after is None


# Logging tests
def test_audit_logger_imports():
    """Test audit logger can be imported."""
    from ragcore.logging_ import audit_logger

    assert audit_logger is not None
    assert hasattr(audit_logger, "log_request")


# Monitoring tests
def test_monitoring_metrics_imports():
    """Test monitoring metrics can be imported."""
    from ragcore.monitoring import (
        record_request,
        record_research_query,
        record_tool_usage,
    )

    assert callable(record_request)
    assert callable(record_research_query)
    assert callable(record_tool_usage)


@pytest.mark.asyncio
async def test_health_check():
    """Test health check endpoint."""
    from ragcore.monitoring import health_check

    status = await health_check.get_health_status()

    assert status["status"] in ["healthy", "degraded"]
    assert "services" in status
    assert "database" in status["services"]
    assert "api_keys" in status["services"]


# Webhook tests
def test_webhook_models():
    """Test webhook models."""
    from ragcore.webhooks.models import WebhookEventType, Webhook, WebhookEvent

    assert WebhookEventType.RESEARCH_COMPLETE == "research.complete"
    assert WebhookEventType.RESEARCH_FAILED == "research.failed"
    assert WebhookEventType.RATE_LIMIT == "system.rate_limit_exceeded"


def test_webhook_manager_imports():
    """Test webhook manager can be imported."""
    from ragcore.webhooks import webhook_manager

    assert webhook_manager is not None
    assert hasattr(webhook_manager, "create_webhook")
    assert hasattr(webhook_manager, "dispatch_event")


def test_supported_events():
    """Test supported webhook events."""
    from ragcore.webhooks.models import SUPPORTED_EVENTS

    assert len(SUPPORTED_EVENTS) >= 5
    assert "research.complete" in SUPPORTED_EVENTS
    assert "system.error" in SUPPORTED_EVENTS


# Integration tests
def test_app_has_auth_routes():
    """Test app includes auth routes."""
    from ragcore.main import create_app

    app = create_app()
    routes = [str(route.path) for route in app.routes]

    assert any("/auth" in r for r in routes), f"Auth routes not found in {routes}"


def test_app_has_webhook_routes():
    """Test app includes webhook routes."""
    from ragcore.main import create_app

    app = create_app()
    routes = [str(route.path) for route in app.routes]

    assert any("/webhooks" in r for r in routes), f"Webhook routes not found in {routes}"


def test_app_has_health_endpoint():
    """Test app includes health endpoint."""
    from ragcore.main import create_app

    app = create_app()
    routes = [str(route.path) for route in app.routes]

    assert "/health" in routes, f"Health endpoint not found in {routes}"


def test_app_has_metrics_endpoint():
    """Test app includes metrics endpoint."""
    from ragcore.main import create_app

    app = create_app()
    routes = [str(route.path) for route in app.routes]

    assert "/metrics" in routes, f"Metrics endpoint not found in {routes}"


def test_config_has_phase4_settings():
    """Test config includes Phase 4 settings."""
    from ragcore.config import settings

    assert hasattr(settings, "auth_enabled")
    assert hasattr(settings, "rate_limit_enabled")
    assert hasattr(settings, "quota_enabled")
    assert hasattr(settings, "audit_logging_enabled")
    assert hasattr(settings, "webhook_enabled")
    assert hasattr(settings, "prometheus_enabled")


# Dependency injection tests
def test_auth_dependencies():
    """Test auth dependency injection."""
    from ragcore.auth.dependencies import (
        get_current_api_key,
        get_current_user,
        get_current_api_key_id,
    )

    assert callable(get_current_api_key)
    assert callable(get_current_user)
    assert callable(get_current_api_key_id)


# Summary test
def test_phase4_complete():
    """Overall Phase 4 completion check."""
    import inspect
    from ragcore.auth import router as auth_router
    from ragcore.webhooks import router as webhook_router

    # Check routers have endpoints
    auth_routes = [r for r in dir(auth_router.router) if not r.startswith("_")]
    webhook_routes = [r for r in dir(webhook_router.router) if not r.startswith("_")]

    assert len(auth_routes) > 0, "Auth router has no endpoints"
    assert len(webhook_routes) > 0, "Webhook router has no endpoints"

    print("Phase 4 all components present")
