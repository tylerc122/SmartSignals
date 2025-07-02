#!/usr/bin/env python3
"""
Smart Traffic Signals - Multi-Scale Analysis Runner

Quick script to run complete multi-scale validation and generate visualizations.
Use this for demonstrations or to reproduce results.
"""

import subprocess
import sys
import os
from datetime import datetime

def run_command(cmd, description):
    """Run a command and handle output."""
    print(f"\n🔄 {description}")
    print("=" * 60)
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, cwd='.')
        
        if result.returncode == 0:
            print(f"✅ {description} completed successfully!")
            if result.stdout:
                print("\nOutput:")
                print(result.stdout)
            return True
        else:
            print(f"❌ {description} failed!")
            if result.stderr:
                print("Error:")
                print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Exception during {description}: {e}")
        return False

def main():
    """Run complete multi-scale analysis."""
    print("🚦 SMART TRAFFIC SIGNALS - MULTI-SCALE ANALYSIS")
    print("=" * 70)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Check if we're in the right directory
    if not os.path.exists('src/evaluation/multi_scale_comparison.py'):
        print("❌ Error: Please run this script from the Smart_Signals project root directory")
        sys.exit(1)
    
    # Step 1: Run multi-scale comparison
    success = run_command(
        "python src/evaluation/multi_scale_comparison.py --scales short medium long",
        "Multi-scale controller comparison"
    )
    
    if not success:
        print("\n❌ Multi-scale comparison failed. Cannot proceed with visualizations.")
        sys.exit(1)
    
    # Step 2: Generate clean visualizations
    success = run_command(
        "python src/evaluation/create_clean_charts.py",
        "Clean multi-scale visualization generation"
    )
    
    if not success:
        print("\n⚠️  Visualizations failed, but data was collected successfully.")
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 MULTI-SCALE ANALYSIS COMPLETED")
    print("=" * 70)
    
    print("\nResults available in:")
    print("  results/phase_1_multi_scale_validation/data/multi_scale_comparison_*.json (detailed data)")
    print("  results/phase_1_multi_scale_validation/data/multi_scale_comparison_*_summary.csv (summary table)")
    print("  results/phase_1_multi_scale_validation/visualizations/ (clean charts)")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()