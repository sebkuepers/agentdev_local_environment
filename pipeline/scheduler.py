#!/usr/bin/env python3
"""
Pipeline Scheduler

This module schedules and runs the data pipeline at regular intervals.
It uses the schedule library to manage scheduling.
"""

import os
import sys
import time
import schedule
import logging
from datetime import datetime
from dotenv import load_dotenv

# Import pipeline
from pipeline.pipeline import run_pipeline

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log")
    ]
)
logger = logging.getLogger(__name__)

# Configuration
CRAWL_INTERVAL = int(os.getenv("CRAWL_INTERVAL", "3600"))  # Default: 1 hour

def job():
    """Job to run the pipeline."""
    logger.info(f"Scheduled pipeline run starting at {datetime.now().isoformat()}")
    try:
        run_pipeline()
        logger.info(f"Scheduled pipeline run completed at {datetime.now().isoformat()}")
    except Exception as e:
        logger.error(f"Scheduled pipeline run failed: {e}")

def main():
    """Main function that sets up and runs the scheduler."""
    logger.info(f"Starting pipeline scheduler. Interval: {CRAWL_INTERVAL} seconds")
    
    # Run once immediately
    job()
    
    # Schedule regular runs
    if CRAWL_INTERVAL > 0:
        # Convert seconds to hours and minutes for better readability in logs
        hours = CRAWL_INTERVAL // 3600
        minutes = (CRAWL_INTERVAL % 3600) // 60
        seconds = CRAWL_INTERVAL % 60
        
        interval_str = ""
        if hours > 0:
            interval_str += f"{hours} hour(s) "
        if minutes > 0:
            interval_str += f"{minutes} minute(s) "
        if seconds > 0:
            interval_str += f"{seconds} second(s)"
            
        logger.info(f"Scheduling pipeline to run every {interval_str}")
        
        # Schedule the job
        schedule.every(CRAWL_INTERVAL).seconds.do(job)
        
        # Keep the script running
        try:
            while True:
                schedule.run_pending()
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Scheduler stopped by user.")
    else:
        logger.info("No recurring schedule set (CRAWL_INTERVAL <= 0). Exiting after initial run.")

if __name__ == "__main__":
    main()