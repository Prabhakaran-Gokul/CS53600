"""Run all assignment 1 tasks: ping and traceroute measurements."""

from loguru import logger

from cs536.assignment_1.ping import run as run_ping
from cs536.assignment_1.traceroute_dennis import run as run_traceroute


def main():
    """Execute both ping and traceroute measurements."""
    logger.info("=" * 60)
    logger.info("STARTING ASSIGNMENT 1: NETWORK MEASUREMENTS")
    logger.info("=" * 60)

    # Run ping measurements
    logger.info("\n" + "=" * 60)
    logger.info("TASK 1: PING MEASUREMENTS")
    logger.info("=" * 60)
    try:
        run_ping()
        logger.success("Ping measurements completed successfully")
    except Exception as e:
        logger.error(f"Ping measurements failed: {e}")

    # Run traceroute measurements
    logger.info("\n" + "=" * 60)
    logger.info("TASK 2: TRACEROUTE MEASUREMENTS")
    logger.info("=" * 60)
    try:
        run_traceroute()
        logger.success("Traceroute measurements completed successfully")
    except Exception as e:
        logger.error(f"Traceroute measurements failed: {e}")

    logger.info("\n" + "=" * 60)
    logger.info("ASSIGNMENT 1 COMPLETED")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
