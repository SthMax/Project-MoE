import argparse
import shutil
from pathlib import Path


def _delete_npz_in_dir(dir_path: Path) -> int:
    if not dir_path.exists():
        return 0
    count = 0
    for p in dir_path.glob("*.npz"):
        p.unlink(missing_ok=True)
        count += 1
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Clean E1.1 outputs for a run/domain to ensure a clean start. "
            "Defaults to deleting only partial .npz files."
        )
    )
    parser.add_argument("--run_dir", required=True, help=".../outputs/runs/<run_id>")
    parser.add_argument(
        "--domain",
        default=None,
        help="If set, only clean this domain. If omitted, clean all domains under the run.",
    )
    parser.add_argument(
        "--delete_run_dir",
        action="store_true",
        help="Delete the entire run_dir (dangerous).",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Required to actually delete files.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        print(f"[E1.1] Nothing to clean; run_dir does not exist: {run_dir}")
        return

    if not args.yes:
        raise RuntimeError(
            "Refusing to delete without --yes. "
            "Re-run with --yes after checking --run_dir/--domain."
        )

    if args.delete_run_dir:
        shutil.rmtree(run_dir)
        print(f"[E1.1] Deleted run_dir: {run_dir}")
        return

    domains: list[Path]
    if args.domain is not None:
        domains = [run_dir / args.domain]
    else:
        domains = [p for p in run_dir.iterdir() if p.is_dir()]

    total_deleted = 0
    for domain_dir in domains:
        partials_dir = domain_dir / "partials"
        deleted = _delete_npz_in_dir(partials_dir)
        total_deleted += deleted
        if deleted:
            print(f"[E1.1] Deleted {deleted} partial(s) in {partials_dir}")

    print(f"[E1.1] Done. Deleted {total_deleted} partial(s) total.")


if __name__ == "__main__":
    main()
