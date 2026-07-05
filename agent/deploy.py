"""Deploys contracts/diplomatic_interpreter.py to a running GenLayer node.

    python -m agent.deploy --chain studionet
    python -m agent.deploy --chain testnet_asimov --private-key 0x...

The GenLayer CLI (`npm install -g genlayer`, then `genlayer deploy
--contract contracts/diplomatic_interpreter.py`) is the documented,
officially supported deploy path -- prefer it if you have Node available.
This script is a minimal genlayer-py-only alternative.

deploy_contract() returns a transaction hash, not an address; on localnet
and studionet the deployed contract's address is surfaced consistently in
the receipt's `recipient` / `data.contract_address` fields (verified on
live studionet deploys in sibling repos). The full simplified receipt is
printed so you can confirm.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
CONTRACT_PATH = REPO_ROOT / "contracts" / "diplomatic_interpreter.py"


def main(argv: list[str] | None = None) -> None:
    from genlayer_py import create_account, create_client
    from genlayer_py.chains import localnet, studionet, testnet_asimov, testnet_bradbury
    from genlayer_py.types import TransactionStatus

    chains = {
        "localnet": localnet,
        "testnet_asimov": testnet_asimov,
        "testnet_bradbury": testnet_bradbury,
        "studionet": studionet,
    }

    parser = argparse.ArgumentParser(description="Deploy DiplomaticInterpreter.")
    parser.add_argument("--chain", default="localnet", choices=sorted(chains))
    parser.add_argument("--private-key", default=os.environ.get("GENLAYER_PRIVATE_KEY"))
    parser.add_argument("--contract-path", default=str(CONTRACT_PATH))
    args = parser.parse_args(argv)

    account = create_account(args.private_key) if args.private_key else create_account()
    client = create_client(chain=chains[args.chain], account=account)

    # studionet shares localnet's chain id 61999, so fund_account works on both.
    if args.chain in ("localnet", "studionet"):
        try:
            client.fund_account(address=account.address, amount=10**18)
        except Exception as exc:
            print(f"fund_account skipped: {exc}")

    code = Path(args.contract_path).read_text()
    print(f"Deploying {args.contract_path} to {args.chain} as {account.address} ...")

    tx_hash = client.deploy_contract(code=code, account=account, args=[])
    receipt = client.wait_for_transaction_receipt(
        transaction_hash=tx_hash,
        status=TransactionStatus.ACCEPTED,
        interval=3000,
        retries=40,
    )

    address = (
        receipt.get("recipient")
        or receipt.get("to_address")
        or (receipt.get("data") or {}).get("contract_address")
    )

    print(json.dumps(receipt, indent=2, default=str))
    if address:
        print(f"\nDeployed contract address (verify against the receipt above): {address}")
        print(
            "Run training against it with:\n"
            f"  python -m agent.train --env genlayer --chain {args.chain} --address {address}"
        )
    else:
        print(
            "\nCould not find an obvious address field in the receipt above. "
            "Inspect it manually, or use `genlayer deploy` / GenLayer Studio instead."
        )


if __name__ == "__main__":
    main()
