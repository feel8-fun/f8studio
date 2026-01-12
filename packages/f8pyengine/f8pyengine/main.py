import argparse
from f8pyengine.service_host import ServiceHostRegistry

import json
from f8pysdk import F8ServiceDescribe


def _main() -> int:
    """F8PyStudio main entry point."""
    parser = argparse.ArgumentParser(description="F8PyStudio Main")
    parser.add_argument(
        "--describe",
        action="store_true",
        help="Output the service description in JSON format",
    )
    args = parser.parse_args()

    if args.describe:
        describe = F8ServiceDescribe(
            service=ServiceHostRegistry.instance().service_spec(),
            operators=ServiceHostRegistry.instance().operator_specs(),
        ).model_dump(mode="json")

        print(json.dumps(describe, ensure_ascii=False, indent=1))
        raise SystemExit(0)


if __name__ == "__main__":
    _main()
