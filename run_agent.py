import argparse
from agentic_data_scientist import AgenticDataScientist


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the offline Agentic Data Scientist.")
    parser.add_argument("--data", required=True, help="Path to CSV dataset")
    parser.add_argument("--target", required=True, help="Target column name or 'auto'")
    parser.add_argument("--output_root", default="outputs", help="Outputs folder")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split fraction")
    parser.add_argument("--max_replans", type=int, default=1, help="Max replans (simple loop)")
    parser.add_argument("--quiet", action="store_true", help="Reduce logs")
    args = parser.parse_args()

    agent = AgenticDataScientist(verbose=not args.quiet)
    out_dir = agent.run(
        data_path=args.data,
        target=args.target,
        output_root=args.output_root,
        seed=args.seed,
        test_size=args.test_size,
        max_replans=args.max_replans,
    )
    print(out_dir)


if __name__ == "__main__":
    main()
