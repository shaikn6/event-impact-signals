"""CLI entry point.

Usage:
    eis poll              # one ingest/classify/store cycle, then exit
    eis serve             # run the live scheduler + API server together
    eis serve --no-live   # just the API, reading whatever is already in the DB
"""

from __future__ import annotations

import argparse
import logging
import threading

from eis.store import DEFAULT_DB_PATH


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="event-impact-signals CLI")
    parser.add_argument("command", choices=["poll", "serve"])
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--interval-minutes", type=int, default=15)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--no-live", action="store_true", help="serve only; don't run the background poller"
    )
    parser.add_argument(
        "--include-reddit",
        action="store_true",
        help="also poll Reddit (best-effort, often blocked without OAuth — see ingest/reddit.py)",
    )
    args = parser.parse_args()

    if args.command == "poll":
        from eis.scheduler import poll_once
        from eis.store import SignalStore

        store = SignalStore(args.db_path)
        inserted = poll_once(store, include_reddit=args.include_reddit)
        print(f"Inserted {inserted} new signal(s) into {args.db_path}")
        store.close()

    elif args.command == "serve":
        import uvicorn

        from eis import api
        from eis.scheduler import run_scheduler

        if not args.no_live:
            thread = threading.Thread(
                target=run_scheduler,
                kwargs={"interval_minutes": args.interval_minutes, "db_path": args.db_path},
                daemon=True,
            )
            thread.start()

        uvicorn.run(api.app, host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
