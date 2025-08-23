#!/usr/bin/env python3
"""
Script to build embedded libraries with progress tracking

This script pre-builds all embeddings for findings in a library, providing
progress bars with ETA for long-running operations.
"""

import argparse
import sys
from pathlib import Path
from typing import Optional, List
import time
from datetime import datetime, timedelta

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from efi_core.types import ChunkerSpec
from efi_analyser.chunkers import SentenceChunker
from efi_analyser.embedders import SentenceTransformerEmbedder
from efi_library.embedded.embedded_library import EmbeddedLibrary


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Build embedded libraries with progress tracking"
    )
    parser.add_argument(
        "--library",
        type=Path,
        default=Path("libraries/crea_publications"),
        help="Path to library directory (default: libraries/crea_publications)"
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("workspace"),
        help="Path to workspace directory (default: workspace)"
    )
    parser.add_argument(
        "--list-libraries",
        action="store_true",
        help="List available libraries and exit"
    )
    
    args = parser.parse_args()
    
    # List available libraries if requested
    if args.list_libraries:
        libraries_dir = Path("libraries")
        if libraries_dir.exists():
            print("Available libraries:")
            for lib_dir in sorted(libraries_dir.iterdir()):
                if lib_dir.is_dir():
                    findings_file = lib_dir / "findings.json"
                    index_file = lib_dir / "index.json"
                    if findings_file.exists() or index_file.exists():
                        print(f"  {lib_dir.name}/")
        else:
            print("No libraries directory found")
        return 0
    
    # Validate paths
    if not args.library.exists():
        print(f"Error: Library path does not exist: {args.library}")
        print("Use --list-libraries to see available libraries")
        return 1
    
    if not args.workspace.exists():
        print(f"Creating workspace directory: {args.workspace}")
        args.workspace.mkdir(parents=True, exist_ok=True)
    
    # Check if library has findings
    findings_file = args.library / "findings.json"
    index_file = args.library / "index.json"
    
    if not findings_file.exists() and not index_file.exists():
        print(f"Error: Library {args.library.name} does not contain findings data")
        print("Expected either findings.json or index.json")
        return 1
    
    embedded_library = EmbeddedLibrary(
        library_path=args.library,
        workspace_path=args.workspace,
        chunker=SentenceChunker(),
        embedder=SentenceTransformerEmbedder()
    )
    
    # Get library info
    info = embedded_library.get_library_info()
    print(f"Library contains {info['document_count']} documents with "
          f"{info['total_findings_count']} total findings")
    
    embedded_library.build_all()
    return 0


if __name__ == "__main__":
    exit(main())
