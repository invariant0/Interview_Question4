from typing import Optional
import logging
from attr import dataclass
import os 
from econ_models.io.file_utils import load_json_file
from econ_models.config.economic_params import EconomicParams
import sys

logger = logging.getLogger(__name__)

@dataclass
class BondsConfig():
    """
    Configuration for state space boundaries.

    Attributes:
        capital_min: Minimum capital value for sampling.
        capital_max: Maximum capital value for sampling.
        productivity_min: Minimum productivity value for sampling.
        productivity_max: Maximum productivity value for sampling.
        debt_min: Minimum debt value for sampling (risky model).
        debt_max: Maximum debt value for sampling (risky model).
    """
    # Domain boundaries

    @staticmethod
    def validate_and_load(
        bounds_file: str,
        current_params: EconomicParams
    ) -> dict:
        """
        Load bounds and validate parameter consistency.

        Args:
            bounds_file: Path to the bounds JSON file.
            current_params: Current economic parameters to validate against.

        Returns:
            Dictionary of validated boundary values.

        Raises:
            SystemExit: If validation fails or file is missing.
        """
        if not os.path.exists(bounds_file):
            logger.error(
                f"Boundary file '{bounds_file}' missing. "
                "Run 'solve_vfi.py --auto-bounds' first."
            )
            sys.exit(1)

        data = load_json_file(bounds_file)

        if "source_params" not in data or "bounds" not in data:
            logger.error(
                "Invalid boundary file format. "
                "Please re-run 'solve_vfi.py --auto-bounds'."
            )
            sys.exit(1)
        BondsConfig._check_parameter_consistency(
            data["source_params"],
            current_params.__dict__
        )

        logger.info("Boundary validation successful.")
        return data["bounds"]

    @staticmethod
    def _check_parameter_consistency(
        stored_params: dict,
        current_params: dict
    ) -> None:
        """Check for mismatches between stored and current parameters."""
        mismatches = []

        for key, val in current_params.items():
            if key in stored_params:
                if stored_params[key] != val:
                    mismatches.append(
                        f"{key}: stored={stored_params[key]}, current={val}"
                    )
            else:
                mismatches.append(f"{key} missing in stored bounds")

        if mismatches:
            logger.critical(
                "Parameter mismatch detected between current config "
                "and pre-computed bounds."
            )
            for m in mismatches:
                logger.error(f"  - {m}")
            logger.error(
                "Action required: Re-run 'solve_vfi.py --auto-bounds' "
                "with the current parameter file."
            )
            sys.exit(1)