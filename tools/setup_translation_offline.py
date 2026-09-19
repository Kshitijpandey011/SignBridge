"""One-time offline translation package downloader for Argos Translate.

NOTE: This is the ONLY script in the entire project permitted to connect to the internet.
It downloads offline model packages (~100MB per language pair) to local storage.
Once downloaded, the entire translation pipeline operates 100% offline.
"""

from __future__ import annotations

import argparse
import sys
from typing import List


def download_argos_packages(languages: List[str]) -> None:
    """Download and install Argos translation packages from English to target languages."""
    try:
        import argostranslate.package  # type: ignore
        import argostranslate.translate  # type: ignore
    except ImportError:
        print("Error: 'argostranslate' is not installed in the current environment.")
        print("Install it using: pip install argostranslate")
        sys.exit(1)

    print("Updating Argos package index from remote repository (one-time setup)...")
    try:
        argostranslate.package.update_package_index()
    except Exception as e:
        print(f"Failed to update package index (Check internet connection): {e}")
        return

    available_packages = argostranslate.package.get_available_packages()
    print(f"Discovered {len(available_packages)} packages in registry.")

    for target_lang in languages:
        target_lang = target_lang.lower().strip()
        if target_lang == "en":
            continue

        print(f"\nSearching offline package for pair: en -> {target_lang}...")
        package_to_install = next(
            (p for p in available_packages if p.from_code == "en" and p.to_code == target_lang),
            None,
        )

        if package_to_install is None:
            print(f"Warning: No package found in registry for en -> {target_lang}.")
            continue

        download_path = package_to_install.download()
        argostranslate.package.install_from_path(download_path)
        print(f"Successfully installed offline package: {package_to_install}")

    print("\nOffline translation setup complete! You can now run offline translation.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download offline Argos translation models")
    parser.add_argument(
        "--languages",
        "-l",
        nargs="+",
        default=["es", "hi", "fr"],
        help="List of target ISO language codes to download (e.g. es, hi, fr, de)",
    )
    args = parser.parse_args()
    download_argos_packages(args.languages)


if __name__ == "__main__":
    main()
