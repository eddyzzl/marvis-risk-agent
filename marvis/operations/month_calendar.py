"""Calendar-month windows; fixed seconds never masquerade as calendar months."""

from dataclasses import dataclass
from datetime import UTC, datetime
from zoneinfo import ZoneInfo


@dataclass(frozen=True)
class CalendarMonthCalendar:
    anchor_at: datetime
    interval_months: int
    timezone: str
    schema_version: str = "operations.calendar.months.v1"

    def __post_init__(self):
        if (
            self.anchor_at.tzinfo is None
            or type(self.interval_months) is not int
            or not 1 <= self.interval_months <= 120
        ):
            raise ValueError(
                "calendar month anchor/timezone and bounded positive interval required"
            )
        zone = ZoneInfo(self.timezone)
        local = self.anchor_at.astimezone(zone)
        if local.day != 1 or (
            local.hour,
            local.minute,
            local.second,
            local.microsecond,
        ) != (0, 0, 0, 0):
            raise ValueError("calendar month anchor must be local first day midnight")
        if self.schema_version != "operations.calendar.months.v1":
            raise ValueError("unsupported monthly calendar schema")
        object.__setattr__(self, "anchor_at", self.anchor_at.astimezone(UTC))

    def boundary(self, index):
        if type(index) is not int:
            raise ValueError("integer month boundary index required")
        local = self.anchor_at.astimezone(ZoneInfo(self.timezone))
        total = local.year * 12 + local.month - 1 + index * self.interval_months
        return datetime(
            total // 12, total % 12 + 1, 1, tzinfo=ZoneInfo(self.timezone)
        ).astimezone(UTC)

    def period(self, index):
        from marvis.operations.contracts import CalendarPeriod

        if type(index) is not int or index < 0:
            raise ValueError("non-negative monthly period index required")
        start, end = self.boundary(index), self.boundary(index + 1)
        return CalendarPeriod(
            f"{self.schema_version}:{self.timezone}:{start.isoformat()}/{end.isoformat()}",
            index,
            start,
            end,
        )

    def due_periods(self, *, active_from, now, limit):
        if (
            active_from.tzinfo is None
            or now.tzinfo is None
            or active_from < self.anchor_at
            or type(limit) is not int
            or limit < 0
        ):
            raise ValueError(
                "aware calendar activation/clock and bounded budget required"
            )
        local = active_from.astimezone(ZoneInfo(self.timezone))
        anchor = self.anchor_at.astimezone(ZoneInfo(self.timezone))
        index = max(
            0,
            ((local.year - anchor.year) * 12 + local.month - anchor.month)
            // self.interval_months,
        )
        if self.boundary(index) < active_from:
            index += 1
        result = []
        for number in range(index, index + limit):
            period = self.period(number)
            if period.ends_at > now:
                break
            result.append(period)
        return tuple(result)

    def to_dict(self):
        return {
            "schema_version": self.schema_version,
            "anchor_at": self.anchor_at.isoformat(),
            "interval_months": self.interval_months,
            "timezone": self.timezone,
        }

    @classmethod
    def from_dict(cls, payload):
        return cls(
            datetime.fromisoformat(payload["anchor_at"].replace("Z", "+00:00")),
            payload["interval_months"],
            payload["timezone"],
            payload["schema_version"],
        )
