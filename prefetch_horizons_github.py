#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.time import Time
from astroquery.jplhorizons import Horizons

MJDREF = 2400000.5

# Keep each Horizons request below the ~90,024 row server limit.
MAX_ROWS_PER_QUERY = 80000


def safe_slug(value: str) -> str:
    value = str(value).strip()
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    value = value.strip("._")
    return value if value else "target"


def read_choose(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Choose file not found: {path}")

    objects = []
    seen = set()

    for raw in path.read_text(encoding="utf-8").splitlines():
        obj = raw.strip()
        if not obj or obj.startswith("#"):
            continue
        if obj not in seen:
            seen.add(obj)
            objects.append(obj)

    return objects


def object_csv_path(input_dir: Path, obj: str):
    """
    Find the object's CSV recursively anywhere under input_dir.

    Supported filename conventions, in this order:

        Rubin / EDP2:
            2006 GQ59 -> 2006_GQ59.csv
            99414     -> 99414.csv

        Asteroid Institute:
            permanent designation   -> permid_99414_ALL.csv
            provisional designation -> provid_2015_BB627_ALL.csv

    Both Asteroid Institute prefixes are tried automatically, so the caller does
    not need to know whether the object is stored as a permanent or provisional ID.
    """
    slug = safe_slug(obj)

    filenames = [
        f"{slug}.csv",
        f"permid_{slug}_ALL.csv",
        f"provid_{slug}_ALL.csv",
    ]

    # Prefer direct matches under input_dir, preserving the priority above.
    for filename in filenames:
        direct = input_dir / filename
        if direct.is_file():
            return direct

    # Then search recursively, again preserving the same priority.
    for filename in filenames:
        matches = sorted(p for p in input_dir.rglob(filename) if p.is_file())

        if not matches:
            continue

        if len(matches) > 1:
            print(
                f"[WARNING] Multiple CSVs found for {obj} using pattern "
                f"{filename}; using {matches[0]}",
                file=sys.stderr,
                flush=True,
            )
            for extra in matches[1:]:
                print(
                    f"          also found: {extra}",
                    file=sys.stderr,
                    flush=True,
                )

        return matches[0]

    return None


def read_mjd_range(csv_path: Path) -> tuple[float, float]:
    """
    Read the observation time range from either supported CSV format.

    Rubin / EDP2:
        midpointMjdTai
        midPointMjdTai   (legacy capitalization)

    Asteroid Institute:
        mjd

    Fallback:
        obs_time         (ISO/date-time string, converted to MJD)
    """
    wanted = {"midpointMjdTai", "midPointMjdTai", "mjd", "obs_time"}

    df = pd.read_csv(csv_path, usecols=lambda c: c in wanted)

    if "midpointMjdTai" in df.columns:
        col = "midpointMjdTai"
        mjd = pd.to_numeric(df[col], errors="coerce").dropna()

    elif "midPointMjdTai" in df.columns:
        col = "midPointMjdTai"
        mjd = pd.to_numeric(df[col], errors="coerce").dropna()

    elif "mjd" in df.columns:
        col = "mjd"
        mjd = pd.to_numeric(df[col], errors="coerce").dropna()

    elif "obs_time" in df.columns:
        col = "obs_time"
        obs_time = pd.to_datetime(df[col], utc=True, errors="coerce").dropna()

        if obs_time.empty:
            mjd = pd.Series(dtype=float)
        else:
            # Unix epoch 1970-01-01 00:00:00 UTC corresponds to MJD 40587.0.
            unix_seconds = obs_time.astype("int64") / 1e9
            mjd = unix_seconds / 86400.0 + 40587.0

    else:
        raise ValueError(
            f"{csv_path} has no supported time column. "
            "Expected one of: midpointMjdTai, midPointMjdTai, mjd, obs_time."
        )

    if len(mjd) == 0:
        raise ValueError(
            f"No valid observation times found in {csv_path} "
            f"using column {col}."
        )

    return float(mjd.min()), float(mjd.max())


def mjd_to_iso_utc(mjd: float) -> str:
    return Time(mjd, format="mjd", scale="utc").isot


def build_cache_path(
    output_dir: Path,
    target_id: str,
    location: str,
    step_minutes: int,
    pad_minutes: int,
    start_mjd: float,
    stop_mjd: float,
) -> Path:
    return output_dir / (
        f"{safe_slug(target_id)}"
        f"__loc_{safe_slug(location)}"
        f"__step_{step_minutes}m"
        f"__pad_{pad_minutes}m"
        f"__{start_mjd:.6f}_{stop_mjd:.6f}.csv"
    )


def cache_is_valid(path: Path) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False

    try:
        df = pd.read_csv(path, nrows=3)
    except Exception:
        return False

    required = {
        "jd",
        "mjd",
        "pred_V",
        "r_au",
        "delta_au",
        "alpha_deg",
    }

    return len(df) >= 2 and required.issubset(df.columns)


def estimate_rows(
    start_mjd: float,
    stop_mjd: float,
    step_minutes: int,
) -> int:
    span_minutes = max(0.0, (stop_mjd - start_mjd) * 1440.0)
    return int(math.floor(span_minutes / step_minutes)) + 1


def make_chunks(
    start_mjd: float,
    stop_mjd: float,
    step_minutes: int,
    max_rows: int = MAX_ROWS_PER_QUERY,
) -> list[tuple[float, float]]:
    """
    Split a long query into smaller Horizons requests.

    Adjacent chunks overlap at one endpoint. Duplicates are removed after
    concatenation, preventing gaps between chunks.
    """
    estimated = estimate_rows(start_mjd, stop_mjd, step_minutes)

    if estimated <= max_rows:
        return [(start_mjd, stop_mjd)]

    max_span_days = ((max_rows - 1) * step_minutes) / 1440.0

    chunks = []
    current = start_mjd

    while current < stop_mjd:
        end = min(current + max_span_days, stop_mjd)
        chunks.append((current, end))

        if end >= stop_mjd:
            break

        current = end

    return chunks


def horizons_table_to_dataframe(table) -> pd.DataFrame:
    eph = table.to_pandas()

    out = pd.DataFrame(
        {
            "jd": pd.to_numeric(
                eph.get("datetime_jd", np.nan),
                errors="coerce",
            ),
            "pred_V": pd.to_numeric(
                eph.get("V", np.nan),
                errors="coerce",
            ),
            "r_au": pd.to_numeric(
                eph.get("r", np.nan),
                errors="coerce",
            ),
            "delta_au": pd.to_numeric(
                eph.get("delta", np.nan),
                errors="coerce",
            ),
            "alpha_deg": pd.to_numeric(
                eph.get("alpha", np.nan),
                errors="coerce",
            ),
            "RA_deg": pd.to_numeric(
                eph.get("RA", np.nan),
                errors="coerce",
            ),
            "DEC_deg": pd.to_numeric(
                eph.get("DEC", np.nan),
                errors="coerce",
            ),
        }
    )

    out = out.dropna(subset=["jd"]).copy()
    out["mjd"] = out["jd"] - MJDREF

    return out


def query_one_chunk(
    target_id: str,
    location: str,
    start_mjd: float,
    stop_mjd: float,
    step_minutes: int,
    retries: int,
    sleep_seconds: float,
    chunk_number: int,
    chunk_count: int,
) -> pd.DataFrame:
    epochs = {
        "start": mjd_to_iso_utc(start_mjd),
        "stop": mjd_to_iso_utc(stop_mjd),
        "step": f"{step_minutes}m",
    }

    last_exc = None

    for attempt in range(1, retries + 1):
        try:
            print(
                f"[QUERY] {target_id} | "
                f"step={step_minutes}m | "
                f"chunk={chunk_number}/{chunk_count} | "
                f"rows~{estimate_rows(start_mjd, stop_mjd, step_minutes)} | "
                f"attempt={attempt}/{retries}",
                flush=True,
            )

            table = Horizons(
                id=target_id,
                location=location,
                epochs=epochs,
            ).ephemerides()

            out = horizons_table_to_dataframe(table)

            if len(out) < 2:
                raise RuntimeError(
                    f"Horizons returned only {len(out)} usable rows "
                    f"for {target_id}, "
                    f"chunk {chunk_number}/{chunk_count}"
                )

            if sleep_seconds > 0:
                time.sleep(sleep_seconds)

            return out

        except Exception as exc:
            last_exc = exc

            print(
                f"[ERROR] {target_id} "
                f"step={step_minutes}m "
                f"chunk={chunk_number}/{chunk_count} "
                f"attempt={attempt}: {exc}",
                file=sys.stderr,
                flush=True,
            )

            if attempt < retries:
                wait = max(
                    sleep_seconds * (2 ** (attempt - 1)),
                    2.0,
                )
                print(
                    f"[WAIT] retrying in {wait:.1f}s",
                    flush=True,
                )
                time.sleep(wait)

    raise RuntimeError(
        f"Failed Horizons query for {target_id}, "
        f"step={step_minutes}m, "
        f"chunk={chunk_number}/{chunk_count} "
        f"after {retries} attempts: {last_exc}"
    )


def query_and_save(
    target_id: str,
    csv_path: Path,
    output_dir: Path,
    location: str,
    step_minutes: int,
    pad_minutes: int,
    retries: int,
    sleep_seconds: float,
) -> Path:

    mjd_min, mjd_max = read_mjd_range(csv_path)

    # The cache filename represents the requested photometric interval plus the
    # normal padding used by the downstream period codes.
    start_mjd = mjd_min - pad_minutes / 1440.0
    stop_mjd = mjd_max + pad_minutes / 1440.0

    # Query Horizons one full sampling step wider on both sides. Horizons returns
    # a discrete cadence grid, so a query that starts/ends exactly at start_mjd /
    # stop_mjd can otherwise return its first/last sample slightly inside those
    # boundaries. The wider query guarantees that the saved cache data covers
    # the interval encoded in the filename.
    step_days = step_minutes / 1440.0
    query_start_mjd = start_mjd - step_days
    query_stop_mjd = stop_mjd + step_days

    out_path = build_cache_path(
        output_dir,
        target_id,
        location,
        step_minutes,
        pad_minutes,
        start_mjd,
        stop_mjd,
    )

    if cache_is_valid(out_path):
        print(
            f"[SKIP cached] {target_id} "
            f"step={step_minutes}m -> {out_path.name}",
            flush=True,
        )
        return out_path

    chunks = make_chunks(
        query_start_mjd,
        query_stop_mjd,
        step_minutes,
    )

    total_estimated = estimate_rows(
        query_start_mjd,
        query_stop_mjd,
        step_minutes,
    )

    if len(chunks) > 1:
        print(
            f"[CHUNK] {target_id} "
            f"step={step_minutes}m "
            f"would produce ~{total_estimated} rows; "
            f"splitting into {len(chunks)} JPL requests",
            flush=True,
        )

    frames = []

    for i, (chunk_start, chunk_stop) in enumerate(
        chunks,
        start=1,
    ):
        frame = query_one_chunk(
            target_id=target_id,
            location=location,
            start_mjd=chunk_start,
            stop_mjd=chunk_stop,
            step_minutes=step_minutes,
            retries=retries,
            sleep_seconds=sleep_seconds,
            chunk_number=i,
            chunk_count=len(chunks),
        )

        frames.append(frame)

    out = pd.concat(frames, ignore_index=True)

    out = (
        out.drop_duplicates(subset=["jd"])
        .sort_values("mjd")
        .reset_index(drop=True)
    )

    if len(out) < 2:
        raise RuntimeError(
            f"Only {len(out)} usable rows remained "
            f"after merging for {target_id}"
        )

    # Verify the actual returned Horizons rows cover the interval represented by
    # the cache filename. This catches an endpoint/cadence problem before the
    # cache is committed to the repository.
    tol = 1e-6
    actual_start = float(out["mjd"].min())
    actual_stop = float(out["mjd"].max())

    if actual_start > start_mjd + tol or actual_stop < stop_mjd - tol:
        raise RuntimeError(
            "Horizons result does not cover the requested cache interval: "
            f"requested={start_mjd:.6f}..{stop_mjd:.6f}, "
            f"returned={actual_start:.6f}..{actual_stop:.6f}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    tmp_path = out_path.with_suffix(".tmp.csv")
    out.to_csv(tmp_path, index=False)
    tmp_path.replace(out_path)

    print(
        f"[OK] {target_id} "
        f"step={step_minutes}m "
        f"rows={len(out)} "
        f"chunks={len(chunks)} -> "
        f"{out_path.name}",
        flush=True,
    )

    return out_path


def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--choose-file",
        type=Path,
        default=Path("choose_horizons.txt"),
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("parquet"),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("HORIZONS_CACHE"),
    )

    parser.add_argument(
        "--location",
        default="X05",
    )

    parser.add_argument(
        "--step-minutes",
        type=int,
        nargs="+",
        default=[1, 10],
    )

    parser.add_argument(
        "--pad-minutes",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--batch-index",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--num-batches",
        type=int,
        required=True,
    )

    parser.add_argument(
        "--retries",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=1.5,
    )

    args = parser.parse_args()

    if args.num_batches < 1:
        raise SystemExit(
            "--num-batches must be >= 1"
        )

    if not 0 <= args.batch_index < args.num_batches:
        raise SystemExit(
            "--batch-index must be between "
            "0 and num-batches-1"
        )

    if not args.input_dir.is_dir():
        raise SystemExit(
            f"Input directory does not exist: "
            f"{args.input_dir}"
        )

    objects = read_choose(
        args.choose_file
    )

    batch_objects = [
        obj
        for index, obj in enumerate(objects)
        if index % args.num_batches
        == args.batch_index
    ]

    print(
        f"Total objects: {len(objects)}",
        flush=True,
    )

    print(
        f"Batch {args.batch_index + 1}/"
        f"{args.num_batches}: "
        f"{len(batch_objects)} objects",
        flush=True,
    )

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    failures = []
    successful_objects = 0

    for number, obj in enumerate(
        batch_objects,
        start=1,
    ):
        print(
            f"\n=== Batch "
            f"{args.batch_index + 1}/"
            f"{args.num_batches} | "
            f"Object {number}/"
            f"{len(batch_objects)}: "
            f"{obj} ===",
            flush=True,
        )

        csv_path = object_csv_path(
            args.input_dir,
            obj,
        )

        if csv_path is None:
            slug = safe_slug(obj)
            msg = (
                f"{obj}: none of "
                f"{slug}.csv, "
                f"permid_{slug}_ALL.csv, or "
                f"provid_{slug}_ALL.csv "
                f"were found anywhere under "
                f"{args.input_dir}"
            )

            print(
                f"[MISSING] {msg}",
                file=sys.stderr,
                flush=True,
            )

            failures.append(msg)
            continue

        print(
            f"[INPUT] {obj} -> {csv_path}",
            flush=True,
        )

        object_ok = True

        for step in args.step_minutes:
            try:
                query_and_save(
                    target_id=obj,
                    csv_path=csv_path,
                    output_dir=args.output_dir,
                    location=args.location,
                    step_minutes=step,
                    pad_minutes=args.pad_minutes,
                    retries=args.retries,
                    sleep_seconds=args.sleep_seconds,
                )

            except Exception as exc:
                object_ok = False

                msg = (
                    f"{obj} "
                    f"step={step}m: {exc}"
                )

                failures.append(msg)

                print(
                    f"[FAILED] {msg}",
                    file=sys.stderr,
                    flush=True,
                )

        if object_ok:
            successful_objects += 1

    print(
        "\n================ SUMMARY ================",
        flush=True,
    )

    print(
        f"Batch: "
        f"{args.batch_index + 1}/"
        f"{args.num_batches}",
        flush=True,
    )

    print(
        f"Objects assigned: "
        f"{len(batch_objects)}",
        flush=True,
    )

    print(
        f"Objects fully successful: "
        f"{successful_objects}",
        flush=True,
    )

    print(
        f"Failures: {len(failures)}",
        flush=True,
    )

    if failures:
        print(
            "\nFailure list:",
            file=sys.stderr,
            flush=True,
        )

        for item in failures:
            print(
                f"  - {item}",
                file=sys.stderr,
                flush=True,
            )

        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
