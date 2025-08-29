#!/usr/bin/env python3
"""
Run all corpus configs in configs/air_quality/ sequentially, or run a single config file.
This script will process each country's config one by one, or a single specified config.
"""

import sys
from pathlib import Path
from typing import List, Dict, Any
import time

from efi_corpus.manager import run_config


def find_configs(config_dir: Path) -> List[Path]:
    """Find all YAML/JSON config files in the given directory"""
    configs = []
    for ext in ['*.yaml', '*.yml', '*.json']:
        configs.extend(config_dir.glob(ext))
    return sorted(configs)


def run_single_config(config_path: Path) -> Dict[str, Any]:
    """Run a single config file"""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    print(f"Running single config: {config_path.name}")
    print(f"Config: {config_path}")
    print("-" * 40)
    
    start_time = time.time()
    summary = run_config(config_path)
    end_time = time.time()
    
    duration = end_time - start_time
    print(f"✅ Completed successfully in {duration:.1f}s")
    print(f"Summary: {summary}")
    
    return {
        "config": config_path.name,
        "status": "success",
        "summary": summary,
        "duration": duration,
        "retries": 0
    }


def run_all_configs(config_dir: str = "configs/air_quality", delay_seconds: int = 60, max_retries: int = 2):
    """Run all configs in the specified directory with delays between runs and retry logic"""
    config_path = Path(config_dir)
    
    if not config_path.exists():
        print(f"Error: Config directory not found: {config_path}")
        sys.exit(1)
    
    configs = find_configs(config_path)
    
    if not configs:
        print(f"No config files found in {config_path}")
        sys.exit(1)
    
    print(f"Found {len(configs)} config files to run:")
    for i, config in enumerate(configs, 1):
        print(f"  {i}. {config.name}")
    
    print(f"\nWill run with {delay_seconds}s delay between configs")
    print(f"Failed configs will be retried up to {max_retries} times")
    print("=" * 60)
    
    results = []
    failed_configs = []  # Track configs that need retrying
    
    # First pass: run all configs
    for i, config_file in enumerate(configs, 1):
        print(f"\n[{i}/{len(configs)}] Running: {config_file.name}")
        print(f"Config: {config_file}")
        print("-" * 40)
        
        try:
            start_time = time.time()
            summary = run_config(config_file)
            end_time = time.time()
            
            result = {
                "config": config_file.name,
                "status": "success",
                "summary": summary,
                "duration": end_time - start_time,
                "retries": 0
            }
            
            print(f"✅ Completed successfully in {result['duration']:.1f}s")
            print(f"Summary: {summary}")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            result = {
                "config": config_file.name,
                "status": "failed",
                "error": str(e),
                "duration": 0,
                "retries": 0
            }
            failed_configs.append((config_file, result))
        
        results.append(result)
        
        # Add delay between runs (except for the last one)
        if i < len(configs):
            print(f"\nWaiting {delay_seconds}s before next config...")
            time.sleep(delay_seconds)
    
    # Retry failed configs
    if failed_configs and max_retries > 0:
        print(f"\n{'='*60}")
        print(f"RETRYING FAILED CONFIGS (up to {max_retries} attempts)")
        print(f"{'='*60}")
        
        for retry_attempt in range(1, max_retries + 1):
            if not failed_configs:
                break
                
            print(f"\n--- Retry attempt {retry_attempt}/{max_retries} ---")
            still_failed = []
            
            for config_file, result in failed_configs:
                print(f"\n🔄 Retrying: {config_file.name} (attempt {retry_attempt})")
                
                try:
                    start_time = time.time()
                    summary = run_config(config_file)
                    end_time = time.time()
                    
                    # Update the original result
                    result["status"] = "success"
                    result["summary"] = summary
                    result["duration"] = end_time - start_time
                    result["retries"] = retry_attempt
                    
                    print(f"✅ Retry successful in {result['duration']:.1f}s")
                    print(f"Summary: {summary}")
                    
                except Exception as e:
                    print(f"❌ Retry failed: {e}")
                    result["error"] = str(e)
                    result["retries"] = retry_attempt
                    still_failed.append((config_file, result))
                
                # Small delay between retries
                time.sleep(10)
            
            failed_configs = still_failed
            
            if failed_configs:
                print(f"\n{len(failed_configs)} configs still failed after retry {retry_attempt}")
                if retry_attempt < max_retries:
                    print(f"Waiting 30s before next retry attempt...")
                    time.sleep(30)
    
    # Print final summary
    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    
    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "failed"]
    
    print(f"Total configs: {len(configs)}")
    print(f"Successful: {len(successful)}")
    print(f"Failed: {len(failed)}")
    
    if successful:
        print(f"\nSuccessful runs:")
        for r in successful:
            retry_info = f" (retried {r['retries']}x)" if r['retries'] > 0 else ""
            print(f"  ✅ {r['config']} ({r['duration']:.1f}s){retry_info}")
    
    if failed:
        print(f"\nFailed runs:")
        for r in failed:
            retry_info = f" (retried {r['retries']}x)" if r['retries'] > 0 else ""
            print(f"  ❌ {r['config']}: {r['error']}{retry_info}")
    
    return results


def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run corpus configs sequentially or a single config file")
    parser.add_argument("--config-dir", default="configs/air_quality", 
                       help="Directory containing config files (default: configs/air_quality)")
    parser.add_argument("--config-file", 
                       help="Single config file to run (overrides --config-dir)")
    parser.add_argument("--delay", type=int, default=60,
                       help="Delay between config runs in seconds (default: 60)")
    parser.add_argument("--max-retries", type=int, default=2,
                       help="Maximum retry attempts for failed configs (default: 2)")
    
    args = parser.parse_args()
    
    try:
        if args.config_file:
            # Run single config file
            config_path = Path(args.config_file)
            result = run_single_config(config_path)
            results = [result]
        else:
            # Run all configs in directory
            results = run_all_configs(args.config_dir, args.delay, args.max_retries)
        
        # Exit with error code if any configs failed
        failed_count = len([r for r in results if r["status"] == "failed"])
        if failed_count > 0:
            print(f"\n⚠️  {failed_count} config(s) failed")
            sys.exit(1)
        else:
            print("\n🎉 All configs completed successfully!")
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
