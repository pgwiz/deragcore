"""Rate limiting and quota system."""

import logging
from datetime import datetime
from typing import Optional
from collections import defaultdict

from ragcore.config import settings
from ragcore.auth.models import APIKey

logger = logging.getLogger(__name__)


class RateLimiter:
    """Track and enforce rate limits per IP and API key."""

    def __init__(self):
        """Initialize rate limiter with in-memory tracking."""
        # IP-based rate limiting: {ip: {timestamp: count}}
        self.ip_requests = defaultdict(lambda: {})

        # Daily quota tracking: {api_key_id: {date: count}}
        self.daily_usage = defaultdict(lambda: {})

        # Monthly quota tracking: {api_key_id: {month: count}}
        self.monthly_usage = defaultdict(lambda: {})

    def check_rate_limit(self, ip_address: str) -> tuple[bool, Optional[int]]:
        """
        Check if IP has exceeded rate limit.

        Args:
            ip_address: Client IP address

        Returns:
            Tuple of (is_allowed, retry_after_seconds)
        """
        if not settings.rate_limit_enabled:
            return True, None

        current_time = datetime.utcnow()
        minute_key = current_time.strftime("%Y-%m-%d %H:%M")

        # Cleanup old entries (older than 2 minutes)
        self.ip_requests[ip_address] = {
            ts: count
            for ts, count in self.ip_requests[ip_address].items()
            if ts >= datetime.utcnow().strftime("%Y-%m-%d %H:%M")
        }

        # Get current minute request count
        current_count = self.ip_requests[ip_address].get(minute_key, 0)

        if current_count >= settings.rate_limit_requests_per_minute:
            # Rate limit exceeded
            retry_after = 60  # Retry after 60 seconds
            logger.warning(
                f"Rate limit exceeded for IP {ip_address}: "
                f"{current_count}/{settings.rate_limit_requests_per_minute} requests/min"
            )
            return False, retry_after

        # Increment counter
        self.ip_requests[ip_address][minute_key] = current_count + 1
        return True, None

    def check_quota(
        self, api_key: Optional[APIKey]
    ) -> tuple[bool, Optional[dict]]:
        """
        Check if API key has exceeded quotas.

        Args:
            api_key: API key object

        Returns:
            Tuple of (is_allowed, quota_info)
        """
        if not settings.quota_enabled or not api_key:
            return True, None

        key_id = str(api_key.id)
        today = datetime.utcnow().strftime("%Y-%m-%d")
        month = datetime.utcnow().strftime("%Y-%m")

        # Get current counts
        daily_count = self.daily_usage[key_id].get(today, 0)
        monthly_count = self.monthly_usage[key_id].get(month, 0)

        # Check limits
        daily_exceeded = daily_count >= settings.quota_daily_limit
        monthly_exceeded = monthly_count >= settings.quota_monthly_limit

        if daily_exceeded or monthly_exceeded:
            quota_info = {
                "daily": {
                    "used": daily_count,
                    "limit": settings.quota_daily_limit,
                    "exceeded": daily_exceeded,
                },
                "monthly": {
                    "used": monthly_count,
                    "limit": settings.quota_monthly_limit,
                    "exceeded": monthly_exceeded,
                },
            }
            logger.warning(f"Quota exceeded for API key {key_id}: {quota_info}")
            return False, quota_info

        # Increment counters
        self.daily_usage[key_id][today] = daily_count + 1
        self.monthly_usage[key_id][month] = monthly_count + 1

        return True, None

    def record_research_cost(self, api_key: Optional[APIKey], tokens: int = 0) -> int:
        """
        Calculate and record cost for research query.

        Args:
            api_key: API key for tracking
            tokens: Approximate tokens used

        Returns:
            Cost in units
        """
        cost = settings.research_query_cost
        if tokens > 0:
            cost += int(tokens * settings.research_cost_token_multiplier)

        logger.debug(f"Research query cost: {cost} units (tokens={tokens})")
        return cost


# Global rate limiter instance
rate_limiter = RateLimiter()
