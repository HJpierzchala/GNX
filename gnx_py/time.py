from datetime import timedelta, datetime
from typing import Union, List
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Iterable, Tuple, Union, List

def arange_datetime(start_datetime, end_datetime, step_minutes=2):
    """
        Generates a list of epochs in the form of datetime objects with a 2-minute interval between two dates.

        Parameters:
        - start_datetime: start date and time as a datetime object.
        - end_datetime: end date and time as a datetime object.
        - step_minutes: interval between epochs in minutes (default 2).

        Returns:
        - List of datetime objects.
        """
    current_datetime = start_datetime
    datetime_list = []

    while current_datetime <= end_datetime:
        datetime_list.append(current_datetime)
        current_datetime += timedelta(minutes=step_minutes)

    return datetime_list


def datetime2doy(dates):
    if isinstance(dates, datetime):
        dates = [dates]

    doy_list = [date.timetuple().tm_yday for date in dates]

    return doy_list if len(doy_list) > 1 else doy_list[0]





# Table of second jumps (UTC -> GPS), effective from 00:00:00 UTC on a given day
# The value is the total number of second jumps applicable from that date.
_LEAP_TABLE = [
    (datetime(1981, 7, 1, tzinfo=timezone.utc), 1),
    (datetime(1982, 7, 1, tzinfo=timezone.utc), 2),
    (datetime(1983, 7, 1, tzinfo=timezone.utc), 3),
    (datetime(1985, 7, 1, tzinfo=timezone.utc), 4),
    (datetime(1988, 1, 1, tzinfo=timezone.utc), 5),
    (datetime(1990, 1, 1, tzinfo=timezone.utc), 6),
    (datetime(1991, 1, 1, tzinfo=timezone.utc), 7),
    (datetime(1992, 7, 1, tzinfo=timezone.utc), 8),
    (datetime(1993, 7, 1, tzinfo=timezone.utc), 9),
    (datetime(1994, 7, 1, tzinfo=timezone.utc), 10),
    (datetime(1996, 1, 1, tzinfo=timezone.utc), 11),
    (datetime(1997, 7, 1, tzinfo=timezone.utc), 12),
    (datetime(1999, 1, 1, tzinfo=timezone.utc), 13),
    (datetime(2006, 1, 1, tzinfo=timezone.utc), 14),
    (datetime(2009, 1, 1, tzinfo=timezone.utc), 15),
    (datetime(2012, 7, 1, tzinfo=timezone.utc), 16),
    (datetime(2015, 7, 1, tzinfo=timezone.utc), 17),
    (datetime(2017, 1, 1, tzinfo=timezone.utc), 18),
]
_GPS_EPOCH = datetime(1980, 1, 6, tzinfo=timezone.utc)


# def _to_timestamp_utc(t) -> datetime:
#     """Convert various time types to datetime in UTC."""
#     # pandas.Timestamp
#     if pd is not None and isinstance(t, pd.Timestamp):
#         if t.tzinfo is None:
#             t = t.tz_localize("UTC")
#         else:
#             t = t.tz_convert("UTC")
#         t = t.round("us")
#         return t.to_pydatetime()
#     # numpy.datetime64
#     if np is not None and isinstance(t, np.datetime64):
#         if pd is not None:
#             ts = pd.Timestamp(t)
#             if ts.tzinfo is None:
#                 ts = ts.tz_localize("UTC")
#             else:
#                 ts = ts.tz_convert("UTC")
#             return ts.to_pydatetime()
#         epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
#         ns = int(str(t).split('[')[0])
#         return epoch + timedelta(microseconds=ns / 1000.0)
#
#     if isinstance(t, datetime):
#         if t.tzinfo is None:
#             return t.replace(tzinfo=timezone.utc)
#         return t.astimezone(timezone.utc)
#
#     raise TypeError(f"Unsupported time type: {type(t)!r}")


_EPOCH_UTC = datetime(1970, 1, 1, tzinfo=timezone.utc)
from functools import lru_cache

@lru_cache(maxsize=200_000)
def _to_timestamp_utc_cached_key(ns_or_us: int, scale: str) -> datetime:
    # scale: "us" or "ns"
    if scale == "ns":
        us = (ns_or_us + 500) // 1000
    else:
        us = ns_or_us
    return _EPOCH_UTC + timedelta(microseconds=us)

def _to_timestamp_utc(t) -> datetime:
    if pd is not None and isinstance(t, pd.Timestamp):
        ts = t.tz_localize("UTC") if t.tzinfo is None else t.tz_convert("UTC")
        return _to_timestamp_utc_cached_key(int(ts.value), "ns")

    if np is not None and isinstance(t, np.datetime64):
        us = int(t.astype("datetime64[us]").astype("int64"))
        return _to_timestamp_utc_cached_key(us, "us")

    if isinstance(t, datetime):
        if t.tzinfo is None:
            # datetime bez tz -> może się powtarzać; cache po "microseconds since epoch" byłby trudniejszy
            return t.replace(tzinfo=timezone.utc)
        return t.astimezone(timezone.utc)

    raise TypeError(f"Unsupported time type: {type(t)!r}")




def _gps_utc_offset_seconds(dt_utc: datetime) -> int:
    """Returns the number of seconds (GPS-UTC) applicable to the moment dt_utc (UTC)."""
    offs = 0
    for since, val in _LEAP_TABLE:
        if dt_utc >= since:
            offs = val
        else:
            break
    return offs


def _utc_to_gps_datetime(dt_utc: datetime) -> datetime:
    """UTC → GPS: add (GPS-UTC) seconds."""
    return dt_utc + timedelta(seconds=_gps_utc_offset_seconds(dt_utc))


def _datetime_to_gps_week_tow(dt_gps: datetime) -> Tuple[int, float]:
    """Based on GPS time, return (week, tow_seconds)."""
    delta = dt_gps - _GPS_EPOCH
    total_seconds = delta.total_seconds()
    week = int(total_seconds // (7 * 24 * 3600))
    tow = total_seconds - week * 7 * 24 * 3600
    return week, tow


def datetime2toc(
    t: Union[datetime, "np.datetime64", "pd.Timestamp", Iterable],
    return_week: bool = False,
    time_scale: str = "GPS",
):
    """
        Convert date/time to TOC (seconds-of-week in GPS time).
        - If `return_week=True`, returns (week, tow). In case of a list: a list of tuples.
    - If `return_week=False`, returns only seconds-of-week (float). In case of a list: a list of floats.
        - `time_scale`:
             * 'UTC' (default): input interpreted as UTC → conversion to GPS time using leap seconds.
             * 'GPS'           : input is already in GPS scale (no additional correction).

        Parameters
        ---------
        t : datetime | numpy.datetime64 | pandas.Timestamp | Iterable of these types
        return_week : bool
        time_scale : {'UTC','GPS'}

        Returns
        ------
        float | (int, float) | List[float] | List[(int, float)]
        """
    def _one(x):
        dt_utc = _to_timestamp_utc(x)
        dt_gps = _utc_to_gps_datetime(dt_utc) if time_scale.upper() == "UTC" else dt_utc
        w, tow = _datetime_to_gps_week_tow(dt_gps)
        return (w, tow) if return_week else tow

    if isinstance(t, (list, tuple)) or (np is not None and isinstance(t, np.ndarray)):
        return [_one(x) for x in t]
    return _one(t)



def toc2datetime(gps_week: int, seconds_of_week: Union[int, List[int]]) -> Union[datetime, List[datetime]]:
    gps_epoch = datetime(1980, 1, 6, 0, 0, 0)

    base_datetime = gps_epoch + timedelta(weeks=gps_week)

    def convert_seconds_to_datetime(seconds: int) -> datetime:
        return base_datetime + timedelta(seconds=seconds)

    if isinstance(seconds_of_week, list):
        return list(map(convert_seconds_to_datetime, seconds_of_week))
    else:
        return convert_seconds_to_datetime(seconds_of_week)


def doy_to_datetime(year: int, doy: int) -> datetime:
    """
        Converts (year, day_of_year) → UTC datetime at 00:00:00.

        Parameters
        ---------
        year : int
            Calendar year (e.g. 2025).
        doy : int
            Day of the year in the range 1-366 (takes leap years into account).

        Returns
        -------
        datetime.datetime
            Object with the date and time set to 00:00:00.
    """
    if doy < 1:
        raise ValueError("doy musi być ≥ 1")
    first_day = datetime(year, 1, 1)
    return first_day + timedelta(days=doy - 1)


def doy2fixed_time(year, doy, hour, minute, second):
    date = doy_to_datetime(year, doy)
    date_with_time = date.replace(hour=hour, minute=minute, second=second)
    return date_with_time


def gpst2utc(gpst_time):
    """
        Converts GPST time to UTC time.

        :param gpst_time: A datetime object representing the time in GPST.
        :return: A datetime object representing the time in UTC.
        """
    GPST_UTC_DIFF_SECONDS = 18
    # Dodaj różnicę (w sekundy) aby uzyskać UTC
    utc_time = gpst_time - timedelta(seconds=GPST_UTC_DIFF_SECONDS)
    return utc_time


def local2gsow(local_time):
    """
        Convert local time to GPS seconds of the week.

        Args:
        local_time (datetime): Local time as a datetime object.

        Returns:
        float: GPS seconds of the week.
        """
    # GPS epoch start time
    gps_epoch = datetime(1980, 1, 6, 0, 0, 0)

    # Calculate the total seconds since the GPS epoch
    delta = local_time - gps_epoch
    total_seconds = delta.total_seconds()

    # Calculate the GPS week number
    gps_week = int(total_seconds // (7 * 86400))

    # Calculate the seconds of the current GPS week
    gps_seconds_of_week = total_seconds % (7 * 86400)

    return gps_seconds_of_week
